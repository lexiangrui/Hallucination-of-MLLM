"""
MathVista MLLM Judge evaluation runner.
"""

import logging
from typing import Optional

from data import load_mathvista
from evaluation.judge import run_mllm_judge

LOGGER = logging.getLogger(__name__)


def run_mathvista(
    model_name: str,
    response_file: str,
    max_samples: Optional[int] = None,
    workers: int = 1,
) -> dict:
    LOGGER.info("[1/4] Loading MathVista dataset...")
    samples = load_mathvista(max_samples=max_samples)
    LOGGER.info("  Loaded %s samples", len(samples))

    return run_mllm_judge(
        model_name=model_name,
        dataset_name="mathvista",
        response_file=response_file,
        samples=samples,
        id_key="pid",
        workers=workers,
        extra_detail_fields=lambda s: {"question_type": s.get("question_type", "")},
    )
