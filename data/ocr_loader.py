"""
OCRBench dataset loader — Hugging Face Parquet format.

Source: https://huggingface.co/datasets/echo840/OCRBench

HF parquet schema:
    dataset, question, question_type, answer (list[str]), image (bytes)
"""

import os
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from utils.batch import hf_image_to_tempfile


def load_ocrbench(
    max_samples: Optional[int] = None,
    task_types: Optional[list[str]] = None,
) -> list[dict]:
    from configs.config import OCR_DATA_DIR

    files = sorted(
        os.path.join(OCR_DATA_DIR, f) for f in os.listdir(OCR_DATA_DIR)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {OCR_DATA_DIR}")

    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    df = table.to_pandas()

    data = []
    for idx, row in df.iterrows():
        task_type = str(row.get("question_type", ""))
        if task_types and task_type not in task_types:
            continue

        answer = ""
        raw = row.get("answer")
        if raw is not None:
            try:
                answer = str(raw[0]) if len(raw) > 0 else ""
            except TypeError:
                answer = str(raw)

        data.append({
            "id": int(idx),
            "image": hf_image_to_tempfile(row.get("image")),
            "question": str(row.get("question", "")),
            "answer": answer,
            "task_type": task_type,
            "dataset_name": str(row.get("dataset", "")),
        })

    if max_samples is not None:
        data = data[:max_samples]

    return data
