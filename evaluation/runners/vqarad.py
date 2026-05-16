"""
VQA-RAD GPT Judge evaluation runner.
"""

import logging
import os
from typing import Optional

from configs import config
from data import load_vqarad
from evaluation.metrics import hallucination_rate
from utils.batch import load_existing_details, load_response_subset, run_resumable_batch, save_json_atomic


LOGGER = logging.getLogger(__name__)


def _compute_vqarad_summary(details: list[dict], total: int) -> tuple[dict, list[int]]:
    scores = []
    hallucination_flags = []
    type_counts = {"faithfulness": 0, "factuality": 0, "logical": 0, "none": 0, "error": 0}
    answer_type_stats = {"closed": {"n": 0, "n_hall": 0}, "open": {"n": 0, "n_hall": 0}}

    for detail in details:
        score = detail.get("score")
        has_h = detail.get("has_hallucination")
        htype = detail.get("hallucination_type", "error")
        atype = detail.get("answer_type", "open")

        if htype not in type_counts:
            htype = "error"
        if score is not None:
            scores.append(score)
        if has_h is not None:
            flag = 1 if has_h else 0
            hallucination_flags.append(flag)
            if atype in answer_type_stats:
                answer_type_stats[atype]["n"] += 1
                answer_type_stats[atype]["n_hall"] += flag
        type_counts[htype] = type_counts.get(htype, 0) + 1

    closed_hr = (
        answer_type_stats["closed"]["n_hall"] / answer_type_stats["closed"]["n"]
        if answer_type_stats["closed"]["n"] > 0 else 0.0
    )
    open_hr = (
        answer_type_stats["open"]["n_hall"] / answer_type_stats["open"]["n"]
        if answer_type_stats["open"]["n"] > 0 else 0.0
    )

    metrics = {
        "total_samples": total,
        "completed_samples": len(details),
        "hallucination_rate": hallucination_rate(hallucination_flags) if hallucination_flags else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "n_hallucinated": sum(hallucination_flags),
        "n_valid": len(hallucination_flags),
        "type_counts": type_counts,
        "closed_hallucination_rate": closed_hr,
        "open_hallucination_rate": open_hr,
        "answer_type_stats": answer_type_stats,
    }
    return metrics, hallucination_flags


def _save_vqarad_snapshot(
    *,
    details_by_id: dict[str, dict],
    sample_ids: list[str],
    total: int,
    model_name: str,
    out_path: str,
) -> None:
    details = [dict(details_by_id[sid]) for sid in sample_ids if sid in details_by_id]
    metrics, _ = _compute_vqarad_summary(details, total)
    save_json_atomic({
        "model_name": model_name,
        "dataset": "vqarad",
        "method": "gpt-judge",
        "metrics": metrics,
        "details": details,
    }, out_path)


def _judge_one(
    sample: dict,
    responses: dict[str, str],
    judge,
) -> tuple[str, dict]:
    sid = str(sample.get("id", ""))
    response = responses.get(sid, "")
    gt_answer = sample.get("answer", "")

    judge_result = judge.judge(
        image_path=sample.get("image", ""),
        question=sample["question"],
        model_response=response,
        ground_truth=gt_answer,
        dataset="vqarad",
    )

    detail = {
        "id": sid,
        "question": sample["question"][:200],
        "answer_type": sample.get("answer_type", "open"),
        "gt_answer": gt_answer,
        "model_response": response[:300],
        "score": judge_result["score"],
        "has_hallucination": judge_result["has_hallucination"],
        "hallucination_type": judge_result["hallucination_type"],
        "reason": judge_result.get("reason", ""),
    }
    return sid, detail


def run_vqarad(
    model_name: str,
    response_file: str,
    max_samples: Optional[int] = None,
    workers: int = 1,
) -> dict:
    """
    Run GPT Judge hallucination detection on VQA-RAD.
    """
    LOGGER.info("[1/4] Loading VQA-RAD dataset...")
    samples = load_vqarad(max_samples=max_samples)
    total = len(samples)
    LOGGER.info("  Loaded %s samples", total)

    LOGGER.info("[2/4] Loading model responses for %s...", model_name)
    responses, samples = load_response_subset(
        path=response_file,
        samples=samples,
        id_key="id",
        dataset="vqarad",
    )
    total = len(samples)

    LOGGER.info("[3/4] Initializing GPT Judge...")
    from evaluation.detectors.gpt_judge import GPTJudge

    judge = GPTJudge(
        model=config.GPT_JUDGE_MODEL,
        temperature=config.GPT_JUDGE_TEMPERATURE,
    )
    LOGGER.info("  GPT Judge ready (model: %s, api_method: %s)", judge.model, judge.client.api_method)

    LOGGER.info("[4/4] Running GPT Judge (%s samples)...", total)
    out_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_vqarad.json")
    sample_ids = [str(sample.get("id", "")) for sample in samples]
    completed_details = load_existing_details(out_path, "id", set(sample_ids))

    completed_details = run_resumable_batch(
        items=samples,
        item_id=lambda sample: str(sample.get("id", "")),
        completed=completed_details,
        process_one=lambda sample: _judge_one(sample, responses, judge),
        save_completed=lambda details_by_id: _save_vqarad_snapshot(
            details_by_id=details_by_id,
            sample_ids=sample_ids,
            total=total,
            model_name=model_name,
            out_path=out_path,
        ),
        workers=workers,
        label="judge results",
    )

    _save_vqarad_snapshot(
        details_by_id=completed_details,
        sample_ids=sample_ids,
        total=total,
        model_name=model_name,
        out_path=out_path,
    )

    details = [dict(completed_details[sid]) for sid in sample_ids if sid in completed_details]
    metrics, hallucination_flags = _compute_vqarad_summary(details, total)

    LOGGER.info("\n%s", "=" * 60)
    LOGGER.info("Results: %s on VQA-RAD (GPT Judge)", model_name)
    LOGGER.info("=" * 60)
    LOGGER.info(
        "  Hallucination Rate: %.4f (%s/%s)",
        metrics["hallucination_rate"],
        sum(hallucination_flags),
        len(hallucination_flags),
    )
    LOGGER.info("  Average Score:      %.2f", metrics["avg_score"])
    LOGGER.info(
        "  Closed-ended HR:    %.4f  Open-ended HR: %.4f",
        metrics["closed_hallucination_rate"],
        metrics["open_hallucination_rate"],
    )
    type_counts = metrics["type_counts"]
    LOGGER.info(
        "  Type breakdown:     faith=%s, fact=%s, logic=%s, none=%s, error=%s",
        type_counts["faithfulness"],
        type_counts["factuality"],
        type_counts["logical"],
        type_counts["none"],
        type_counts["error"],
    )

    LOGGER.info("\nSaved to: %s", out_path)
    return {"model_name": model_name, "dataset": "vqarad", "metrics": metrics}
