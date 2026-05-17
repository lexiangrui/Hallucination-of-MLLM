"""
MathVista dataset loader — Hugging Face Parquet format.

Source: https://huggingface.co/datasets/AI4Math/MathVista

HF parquet schema:
    pid, question, image (filename), decoded_image (bytes), choices, unit,
    precision, answer, question_type, answer_type, metadata, query
"""

import math
from typing import Optional

from utils.batch import hf_image_to_tempfile, normalize_choices, read_parquet_dir


def load_mathvista(
    max_samples: Optional[int] = None,
    question_types: Optional[list[str]] = None,
    split: str = "testmini",
) -> list[dict]:
    from configs.config import MATHVISTA_DATA_DIR

    df = read_parquet_dir(MATHVISTA_DATA_DIR, prefix=split).to_pandas()

    has_decoded = "decoded_image" in df.columns
    data = []

    for row in df.to_dict("records"):
        if max_samples is not None and len(data) >= max_samples:
            break

        q_type = row["question_type"]
        if question_types and q_type not in question_types:
            continue

        image_path = hf_image_to_tempfile(row.get("decoded_image")) if has_decoded else ""
        choices = normalize_choices(row.get("choices"))

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

    return data
