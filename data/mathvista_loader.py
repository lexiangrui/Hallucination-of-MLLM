"""
MathVista dataset loader — Hugging Face Parquet format.

Source: https://huggingface.co/datasets/AI4Math/MathVista

HF parquet schema:
    pid, question, image (filename), decoded_image (bytes), choices, unit,
    precision, answer, question_type, answer_type, metadata, query
"""

import math
import os
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from utils.batch import hf_image_to_tempfile


def load_mathvista(
    max_samples: Optional[int] = None,
    question_types: Optional[list[str]] = None,
    split: str = "testmini",
) -> list[dict]:
    """
    Load MathVista dataset from Hugging Face parquet files.

    Args:
        max_samples: Maximum number of samples to load
        question_types: Filter by question types
        split: Dataset split to load ("testmini" or "test"). Default is "testmini" (1000 samples).
    """
    from configs.config import MATHVISTA_DATA_DIR

    files = sorted(
        os.path.join(MATHVISTA_DATA_DIR, f) for f in os.listdir(MATHVISTA_DATA_DIR)
        if f.endswith(".parquet") and f.startswith(split)
    )
    if not files:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' in {MATHVISTA_DATA_DIR}"
        )

    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    df = table.to_pandas()

    has_decoded = "decoded_image" in df.columns
    data = []

    for _, row in df.iterrows():
        q_type = row["question_type"]
        if question_types and q_type not in question_types:
            continue

        image_path = hf_image_to_tempfile(row.get("decoded_image")) if has_decoded else ""

        choices = row.get("choices")
        if choices is not None and hasattr(choices, "tolist"):
            choices = choices.tolist()
        elif choices is not None and not isinstance(choices, (list, tuple)):
            choices = [choices]

        query = row.get("query", "")
        question = str(query) if query else str(row.get("question", ""))

        precision = row.get("precision")
        if precision is not None and hasattr(precision, "item"):
            precision = precision.item()
        if precision is not None and isinstance(precision, float) and math.isnan(precision):
            precision = None

        unit = row.get("unit")
        if unit is not None and str(unit) in ("None", "nan", ""):
            unit = None
        elif unit is not None:
            unit = str(unit)

        data.append({
            "pid": int(row["pid"]) if str(row["pid"]).isdigit() else str(row["pid"]),
            "image": image_path,
            "question": question,
            "answer": str(row["answer"]) if row["answer"] else "",
            "question_type": q_type,
            "answer_type": str(row.get("answer_type", "text")),
            "precision": precision,
            "unit": unit,
            "choices": choices,
        })

    if max_samples is not None:
        data = data[:max_samples]

    return data
