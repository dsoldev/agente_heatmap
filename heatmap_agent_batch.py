import base64, json, asyncio, aiofiles, os
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
load_dotenv(find_dotenv())  # Carrega o arquivo .env local
from pydantic import BaseModel, Field
from typing import List
import cv2
from pathlib import Path


COMPONENTS = {
    "duck": ["cabeça", "corpo", "asa/cauda"],                 # pato
    "driller": ["empunhadura", "corpo", "cabeçote"],
    "cat": ["cabeça", "corpo", "patas/cauda"],            # gato
}

class ImageAgentOutput(BaseModel):
    component: List[str] = Field(
        description='Lista de componentes mais quentes no heatmap da imagem. Retorna uma lista vazia se não houver componentes quentes. Exemplo: ["cabeça", "corpo"]'
    )

SYSTEM_TEMPLATE = """
Você é um especialista em mapas de calor (heatmaps) de visão computacional.

A imagem que receberá SEMPRE tem:
• lado esquerdo → foto original
• lado direito  → a mesma foto com heatmap JET (vermelho = quente, azul = frio)
  sobreposto com transparência ≈50 %.

Seu trabalho:
1. Observe SOMENTE o lado direito (heatmap).
2. Encontre onde estão os componentes nomeados abaixo.
3. Julgue visualmente quais componentes mostraram as cores quentes (vermelho).
4. Retorne **apenas** uma lista com o(s) nome(s) desses componentes,
   usando exatamente os nomes abaixo:

{component_list}

Se o calor estiver distribuído uniformemente e nenhum componente se destacar,
retorne uma lista vazia: [].
"""

STRUCTURED_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ImageAgentOutput",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["component"],
            "additionalProperties": False
        }
    }
}

def encode_image(path):
    """Codifica imagem para base64 com redimensionamento."""
    img = cv2.imread(path)
    # print(path)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    _, img_encoded = cv2.imencode('.png', img)
    img_b64 = base64.b64encode(img_encoded).decode()
    return img_b64

def create_batch_request(image_path, obj_type, custom_id=None):
    """Cria uma requisição individual para o batch."""
    if custom_id is None:
        custom_id = f"request_{os.path.basename(image_path)}_{obj_type}"
    
    # Codifica a imagem
    img_b64 = encode_image(image_path)
    
    # Monta as mensagens
    system_content = SYSTEM_TEMPLATE.format(
        component_list=", ".join(COMPONENTS[obj_type])
    )
    
    batch_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5-mini",
            "temperature": 1,
            "seed": 42,  # Seed fixo para maior determinismo
            "response_format": STRUCTURED_OUTPUT_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ]
        }
    }
    
    return batch_request

def generate_batch_file(image_paths, obj_types, output_file="batch_requests.jsonl"):
    """Gera arquivo JSONL com todas as requisições para o batch."""
    
    # Garantir que image_paths e obj_types tenham o mesmo tamanho
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if isinstance(obj_types, str):
        obj_types = [obj_types]
    
    if len(obj_types) == 1 and len(image_paths) > 1:
        obj_types = obj_types * len(image_paths)
    
    batch_requests = []

    for i, (image_path, obj_type) in tqdm(enumerate(zip(image_paths, obj_types)), total=len(image_paths)):
        name = str(image_path).replace('\\','-')
        custom_id = f"request_{i:04d}_{name}_{obj_type}"
        request = create_batch_request(image_path, obj_type, custom_id)
        batch_requests.append(request)
    
    # Escreve o arquivo JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    print(f"Arquivo {output_file} gerado com {len(batch_requests)} requisições.")
    return output_file

async def analyse_image(path, obj_type):
    """Função original mantida para compatibilidade (modo não-batch)."""
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    
    # modelo com visão
    llm = ChatOpenAI(
        model_name="gpt-4.1-mini", 
        temperature=0.0,
        model_kwargs={"seed": 42}  # Seed fixo para maior determinismo
    )
    llm = llm.with_structured_output(ImageAgentOutput)

    # 3.1 lê e codifica em base64
    img_b64 = encode_image(path)

    # 3.2 monta mensagens
    system_msg = SystemMessage(
        content=SYSTEM_TEMPLATE.format(
            component_list=", ".join(COMPONENTS[obj_type])
        )
    )
    human_msg = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {           
                "url": f"data:image/png;base64,{img_b64}"
            }
        }
    ])
    # 3.3 chamada à API
    response = await llm.ainvoke([system_msg, human_msg])
    # converte string → lista Python
    print(f"Resposta: {response}")
    return response.component

if __name__ == "__main__":
    object_name = "duck"
    image_files = sorted(Path(object_name).rglob("*.png"))
    print(image_files[0])

    print(f"Found {len(image_files)} images in '{object_name}' directory.")

    # Gerar arquivo JSONL para batch API
    generate_batch_file(image_files, [object_name], f"{object_name}_analysis_batch.jsonl")

    # fname = "000588.png"
    # print("=== TESTE 1 ===")
    # result1 = asyncio.run(analyse_image(fname, object))
    
    # print("\n=== TESTE 2 ===")
    # result2 = asyncio.run(analyse_image(fname, object))
    
    # print(f"\nResultado 1: {result1}")
    # print(f"Resultado 2: {result2}")
    # print(f"Consistente: {result1 == result2}")