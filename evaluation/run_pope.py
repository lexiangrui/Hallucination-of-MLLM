"""
POPE rule-based evaluation runner.
"""

import logging
import os
from typing import Optional

from configs import config
from data import load_pope_by_split
from evaluation.metrics import compute_pope_metrics
from evaluation.rule_based import detect_pope_hallucination
from utils.batch import load_response_subset, save_json_atomic

LOGGER = logging.getLogger(__name__)


def run_pope(
    model_name: str,
    response_file: str,
    max_samples: Optional[int] = None,
    pope_split: str = "random",
) -> dict:
    LOGGER.info("[1/2] Loading POPE dataset (%s)...", pope_split)
    samples = load_pope_by_split(split=pope_split, max_samples=max_samples)
    LOGGER.info("  Loaded %s samples", len(samples))

    LOGGER.info("[2/2] Running rule-based detection...")
    responses, samples = load_response_subset(
        path=response_file,
        samples=samples,
        id_key="question_id",
        dataset=f"pope/{pope_split}",
    )
    total = len(samples)

    label_list = []
    pred_list = []
    details = []

    for sample in samples:
        sid = str(sample.get("question_id", ""))
        response = responses.get(sid, "")
        gt_answer = sample.get("answer", "")

        result = detect_pope_hallucination(response, gt_answer)
        predicted = result.get("predicted")

        label_list.append(gt_answer)
        pred_list.append(predicted)

        details.append({
            "question_id": sid,
            "question": sample["question"][:200],
            "gt_answer": gt_answer,
            "model_response": response[:200],
            "predicted": predicted,
            "is_correct": result["is_correct"],
            "is_object_hallucination": result["is_hallucination"],
            "detail": result["detail"],
        })

    metrics = compute_pope_metrics(label_list, pred_list)
    cm = metrics["confusion_matrix"]

    LOGGER.info("\n%s", "=" * 60)
    LOGGER.info("Results: %s on POPE-%s (规则判断法)", model_name, pope_split)
    LOGGER.info("=" * 60)
    LOGGER.info("  TP/FP/TN/FN:        %s/%s/%s/%s", cm["TP"], cm["FP"], cm["TN"], cm["FN"])
    LOGGER.info("  Accuracy:           %.4f", metrics["accuracy"])
    LOGGER.info("  Precision:          %.4f", metrics["precision"])
    LOGGER.info("  Recall:             %.4f", metrics["recall"])
    LOGGER.info("  F1 Score:           %.4f", metrics["f1"])
    LOGGER.info("  Yes Ratio:          %.4f", metrics["yes_ratio"])
    LOGGER.info(
        "  Object Halluc. Rate:%.4f (%s/%s)",
        metrics["object_hallucination_rate"],
        metrics["n_object_hallucinated"],
        total,
    )

    out_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_pope_{pope_split}.json")
    save_json_atomic({
        "model_name": model_name,
        "dataset": "pope",
        "split": pope_split,
        "method": "rule-based",
        "metrics": metrics,
        "details": details,
    }, out_path)
    LOGGER.info("\nSaved to: %s", out_path)

    return {
        "model_name": model_name,
        "dataset": "pope",
        "split": pope_split,
        "metrics": metrics,
    }
