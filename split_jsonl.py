import os
import sys
from typing import TextIO

MB = 1024 * 1024
DEFAULT_MAX_MB = 200  # limite padrão

def _open_new_part(base_dir: str, base_name: str, ext: str, part_idx: int) -> TextIO:
    """Cria um novo arquivo de parte e devolve o handle."""
    path = os.path.join(base_dir, f"{base_name}_{part_idx}{ext}")
    return open(path, "w", encoding="utf-8")


def split_jsonl(path: str, max_mb: int = DEFAULT_MAX_MB) -> None:
    """Divide *path* em partes menores que *max_mb* MB e imprime log por linhas."""

    max_bytes = max_mb * MB
    if max_bytes <= 0:
        raise ValueError("max_mb precisa ser > 0")

    # Conta linhas totais em uma passada rápida
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    base_dir, filename = os.path.split(os.path.abspath(path))
    base_name, ext = os.path.splitext(filename)

    part_idx = 1
    current_size = 0
    lines_in_part = 0
    start_line_idx = 0  # linha inicial da parte (0‑based)
    global_line_idx = 0

    out_file = _open_new_part(base_dir, base_name, ext, part_idx)

    def _log_part(idx: int, start: int, end: int, n_lines: int):
        """Imprime log: Arquivo N (X linhas) – A até B de T"""
        print(
            f"Arquivo {idx} ({n_lines} linhas) – "
            f"{start} até {end} de {total_lines}"
        )

    # Segunda passada: divide o arquivo
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line_size = len(line.encode("utf-8"))

            # Se estourar limite → fecha parte
            if current_size + line_size > max_bytes and current_size > 0:
                out_file.close()
                _log_part(part_idx, start_line_idx, global_line_idx, lines_in_part)

                part_idx += 1
                start_line_idx = global_line_idx
                out_file = _open_new_part(base_dir, base_name, ext, part_idx)
                current_size = 0
                lines_in_part = 0

            out_file.write(line)
            current_size += line_size
            lines_in_part += 1
            global_line_idx += 1

    # fecha última parte
    out_file.close()
    _log_part(part_idx, start_line_idx, global_line_idx, lines_in_part)
    print(f"✅ Total de partes: {part_idx}")

if __name__ == "__main__":
    jsonl_path = 'duck_analysis_batch.jsonl'
    max_mb = 190

    split_jsonl(jsonl_path, max_mb)
