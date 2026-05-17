"""
POPE dataset loader — Hugging Face Parquet format.

Source: https://huggingface.co/datasets/lmms-lab/POPE

HF parquet schema (default config):
    id, question_id, question, answer, image_source, image (bytes), category
    category ∈ {random, popular, adversarial}
"""

import random
from typing import Optional

import pyarrow.compute as pc

from utils.batch import hf_image_to_tempfile, read_parquet_dir


def load_pope_by_split(
    split: str = "random",
    data_dir: Optional[str] = None,
    max_samples: int = 1000,
    seed: int = 42,
) -> list[dict]:
    from configs.config import POPE_DATA_DIR

    data_dir = data_dir or POPE_DATA_DIR

    table = read_parquet_dir(data_dir)
    table = table.filter(pc.equal(table["category"], split))
    df = table.to_pandas()

    if max_samples < len(df):
        indices = sorted(random.Random(seed).sample(range(len(df)), max_samples))
        df = df.iloc[indices]

    return [
        {
            "question_id": str(row.get("question_id", row.get("id", ""))),
            "image": hf_image_to_tempfile(row.get("image")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")).strip().lower(),
        }
        for row in df.to_dict("records")
    ]
