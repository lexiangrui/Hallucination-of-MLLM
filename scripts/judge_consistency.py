#!/usr/bin/env python3
"""
E1: Judge model consistency experiment.

Compares GPT-5.5 Judge vs Claude Opus 4.7 Judge on the same 100 MathVista samples.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.config import MODELS
from data import load_mathvista
from evaluation.judge import run_mllm_judge
from evaluation.metrics import cohens_kappa, compute_mllm_judge_summary
from utils.batch import load_response_subset, save_json_atomic, load_json_object

LOGGER = logging.getLogger(__name__)

SAMPLES = 100
SAMPLE_SEED = 42
JUDGE_MODEL = "claude-opus-4-7"
OUTPUT_DIR = "results/errors_analysis/judge_consistency"


def _get_mathvista_results(model_name: str) -> str:
    return f"results/{model_name}_mathvista.json"


def _get_mathvista_responses(model_name: str) -> str:
    return f"responses/{model_name}_mathvista.json"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Judge consistency experiment with Claude Opus.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1: sample 100 from MathVista, align with responses
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 1: Sampling %d MathVista samples (seed=%d)", SAMPLES, SAMPLE_SEED)
    LOGGER.info("=" * 60)

    mathvista_samples = load_mathvista()
    random.seed(SAMPLE_SEED)
    indices = sorted(random.sample(range(len(mathvista_samples)), SAMPLES))
    base_samples = [mathvista_samples[i] for i in indices]

    gpt_details_by_model = {}
    all_samples = {}
    all_responses = {}

    for model_name in MODELS:
        resp_path = _get_mathvista_responses(model_name)
        if not os.path.exists(resp_path):
            LOGGER.warning("  SKIP: response file not found %s", resp_path)
            continue

        responses, aligned = load_response_subset(
            path=resp_path, samples=base_samples, id_key="pid", dataset="mathvista",
        )
        if not aligned:
            LOGGER.warning("  SKIP: no aligned samples for %s", model_name)
            continue

        sampled_responses = {str(s["pid"]): responses.get(str(s["pid"]), "") for s in aligned}
        resp_out = os.path.join(OUTPUT_DIR, f"{model_name}_responses.json")
        save_json_atomic(sampled_responses, resp_out)

        all_samples[model_name] = aligned
        all_responses[model_name] = resp_out

        # Extract and save GPT-5.5 results for these 100 samples
        gpt_result = load_json_object(_get_mathvista_results(model_name))
        gpt_details = gpt_result.get("details", [])
        sampled_ids = {str(s["pid"]) for s in aligned}
        filtered = [d for d in gpt_details if str(d.get("pid", "")) in sampled_ids]
        if filtered:
            gpt_metrics, _ = compute_mllm_judge_summary(filtered, len(filtered))
            gpt_out = os.path.join(OUTPUT_DIR, f"{model_name}_gpt.json")
            save_json_atomic({"model_name": model_name, "dataset": "mathvista",
                               "metrics": gpt_metrics, "details": filtered}, gpt_out)
            gpt_details_by_model[model_name] = filtered

        LOGGER.info("  %s: %d samples aligned", model_name, len(aligned))

    # Phase 2: run Claude Opus Judge
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Phase 2: Running Claude Opus Judge on %d models", len(all_samples))
    LOGGER.info("=" * 60)

    claude_details_by_model = {}
    for model_name, samples in all_samples.items():
        LOGGER.info("[%s]", model_name)
        result = run_mllm_judge(
            model_name=model_name,
            dataset_name="mathvista",
            response_file=all_responses[model_name],
            samples=samples,
            id_key="pid",
            workers=args.workers,
            judge_model=JUDGE_MODEL,
            output_dir=OUTPUT_DIR,
            output_name=f"{model_name}_claude.json",
        )
        claude_details_by_model[model_name] = result.get("details", [])

    # Phase 3: compare GPT-5.5 vs Claude
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Phase 3: Comparison Summary")
    LOGGER.info("=" * 60)

    print("")
    header = f"{'Model':<30} {'GPT HR':<10} {'Claude HR':<12} {'ΔHR':<10} {'GPT Avg':<10} {'Claude Avg':<12} {'Kappa':<8}"
    print(header)
    print("-" * 92)

    all_gpt_scores = []
    all_claude_scores = []
    all_gpt_h = []
    all_claude_h = []

    for model_name in MODELS:
        gpt_details = gpt_details_by_model.get(model_name)
        claude_details = claude_details_by_model.get(model_name)

        if not gpt_details or not claude_details:
            print(f"{model_name:<30} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<8}")
            continue

        gpt_scores = [d.get("score", 0) for d in gpt_details]
        claude_scores = [d.get("score", 0) for d in claude_details]
        gpt_h = [int(d.get("has_hallucination", False)) for d in gpt_details]
        claude_h = [int(d.get("has_hallucination", False)) for d in claude_details]

        gpt_hr = sum(gpt_h) / len(gpt_h)
        claude_hr = sum(claude_h) / len(claude_h)
        gpt_avg = sum(gpt_scores) / len(gpt_scores)
        claude_avg = sum(claude_scores) / len(claude_scores)
        kappa = cohens_kappa(gpt_h, claude_h)

        print(f"{model_name:<30} {gpt_hr:<10.4f} {claude_hr:<12.4f} {claude_hr - gpt_hr:<+10.4f} "
              f"{gpt_avg:<10.2f} {claude_avg:<12.2f} {kappa:<8.4f}")

        all_gpt_scores.extend(gpt_scores)
        all_claude_scores.extend(claude_scores)
        all_gpt_h.extend(gpt_h)
        all_claude_h.extend(claude_h)

    print("-" * 92)
    if all_gpt_scores:
        r, p = pearsonr(all_gpt_scores, all_claude_scores)
        overall_kappa = cohens_kappa(all_gpt_h, all_claude_h)
        print(f"{'Overall (Pearson r)':<30} {r:<10.4f}  (p={p})")
        print(f"{'Overall (Kappa)':<30} {overall_kappa:<10.4f}")


if __name__ == "__main__":
    main()
