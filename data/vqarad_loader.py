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

from utils.batch import hf_image_to_tempfile, read_parquet_dir


def _answer_type(answer: str) -> str:
    return "closed" if answer.lower() in {"yes", "no"} else "open"


def load_vqarad(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> list[dict]:
    """Load VQA-RAD by parquet filename prefix (reads 'test-*' by default)."""
    from configs.config import VQARAD_DATA_DIR

    df = read_parquet_dir(VQARAD_DATA_DIR, prefix=split).to_pandas()

    if max_samples is not None and max_samples < len(df):
        df = df.iloc[:max_samples]

    data = []
    for idx, row in df.iterrows():
        answer = str(row.get("answer", ""))
        data.append({
            "id": int(idx),
            "image": hf_image_to_tempfile(row.get("image")),
            "question": str(row.get("question", "")),
            "answer": answer,
            "answer_type": _answer_type(answer),
        })
    return data