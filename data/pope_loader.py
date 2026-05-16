"""
POPE dataset loader — Hugging Face Parquet format.

Source: https://huggingface.co/datasets/lmms-lab/POPE

HF parquet schema (default config):
    id, question_id, question, answer, image_source, image (bytes), category
    category ∈ {random, popular, adversarial}
"""

import os
import random
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from utils.batch import hf_image_to_tempfile


def _read_parquet(data_dir: str) -> pa.Table:
    files = sorted(
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def load_pope_by_split(
    split: str = "random",
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> list[dict]:
    from configs.config import POPE_DATA_DIR, POPE_DEFAULT_MAX_SAMPLES, POPE_SAMPLE_SEED

    data_dir = data_dir or POPE_DATA_DIR
    if max_samples is None:
        max_samples = POPE_DEFAULT_MAX_SAMPLES
    if sample_seed is None:
        sample_seed = POPE_SAMPLE_SEED

    table = _read_parquet(data_dir)
    df = table.to_pandas()
    df = df[df["category"] == split]

    if max_samples and max_samples < len(df):
        indices = sorted(random.Random(sample_seed).sample(range(len(df)), max_samples))
        df = df.iloc[indices]

    return [{
        "question_id": str(row.get("question_id", row.get("id", ""))),
        "image": hf_image_to_tempfile(row.get("image")),
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")).strip().lower(),
    } for _, row in df.iterrows()]
