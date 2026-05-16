"""
VQA-RAD dataset loader — Hugging Face Parquet format.

Source: https://huggingface.co/datasets/flaviagiammarino/vqa-rad

HF parquet schema:
    image (bytes), question (str), answer (str)

Reference:
    Lau et al., "A dataset of clinically generated visual questions and
    answers about radiology images," Scientific Data, 2018.
    https://doi.org/10.1038/sdata.2018.251
"""

import os
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from utils.batch import hf_image_to_tempfile


def load_vqarad(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> list[dict]:
    from configs.config import VQARAD_DATA_DIR

    split_dir = os.path.join(VQARAD_DATA_DIR, split)
    if not os.path.isdir(split_dir):
        split_dir = VQARAD_DATA_DIR

    files = sorted(
        os.path.join(split_dir, f) for f in os.listdir(split_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")

    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    df = table.to_pandas()

    data = []
    for idx, row in df.iterrows():
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        answer_type = "closed" if answer.lower() in {"yes", "no"} else "open"

        data.append({
            "id": int(idx),
            "image": hf_image_to_tempfile(row.get("image")),
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
        })

    if max_samples is not None:
        data = data[:max_samples]

    return data
