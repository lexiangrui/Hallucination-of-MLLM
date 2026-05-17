"""
VQA-RAD GPT Judge evaluation runner.
"""

import logging
from typing import Optional

from data import load_vqarad
from evaluation.judge import run_gpt_judge
from evaluation.metrics import compute_gpt_judge_summary

LOGGER = logging.getLogger(__name__)


def _compute_vqarad_summary(details: list[dict], total: int) -> tuple[dict, list[int]]:
    metrics, hallucination_flags = compute_gpt_judge_summary(details, total)

    answer_type_stats = {"closed": {"n": 0, "n_hall": 0}, "open": {"n": 0, "n_hall": 0}}
    for detail in details:
        has_h = detail.get("has_hallucination")
        atype = detail.get("answer_type", "open")
        if has_h is not None and atype in answer_type_stats:
            flag = 1 if has_h else 0
            answer_type_stats[atype]["n"] += 1
            answer_type_stats[atype]["n_hall"] += flag

    closed = answer_type_stats["closed"]
    open_ = answer_type_stats["open"]
    metrics["closed_hallucination_rate"] = closed["n_hall"] / closed["n"] if closed["n"] else 0.0
    metrics["open_hallucination_rate"] = open_["n_hall"] / open_["n"] if open_["n"] else 0.0
    metrics["answer_type_stats"] = answer_type_stats

    return metrics, hallucination_flags


def _log_vqarad_extra(metrics: dict) -> None:
    LOGGER.info(
        "  Closed-ended HR:    %.4f  Open-ended HR: %.4f",
        metrics["closed_hallucination_rate"],
        metrics["open_hallucination_rate"],
    )


def run_vqarad(
    model_name: str,
    response_file: str,
    max_samples: Optional[int] = None,
    workers: int = 1,
) -> dict:
    LOGGER.info("[1/4] Loading VQA-RAD dataset...")
    samples = load_vqarad(max_samples=max_samples)
    LOGGER.info("  Loaded %s samples", len(samples))

    return run_gpt_judge(
        model_name=model_name,
        dataset_name="vqarad",
        response_file=response_file,
        samples=samples,
        id_key="id",
        workers=workers,
        compute_summary=_compute_vqarad_summary,
        extra_detail_fields=lambda s: {"answer_type": s.get("answer_type", "open")},
        log_extra=_log_vqarad_extra,
    )
