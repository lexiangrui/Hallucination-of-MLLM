#!/usr/bin/env python3
"""
E2/E3/E4: Error analysis experiments (zero API cost).

E2: Threshold sensitivity — vary hallucination cutoff 0-6, plot HR curves
E3: Length bias — correlation between answer length and Judge score
E4: Cross-model consistency — how many models agree on each sample
"""

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.config import MODELS
from utils.batch import save_json_atomic, load_json_object

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = "results/errors_analysis"

DATASETS = {
    "pope_random": {"result": lambda m: f"results/{m}_pope_random.json", "id_key": "question_id"},
    "pope_popular": {"result": lambda m: f"results/{m}_pope_popular.json", "id_key": "question_id"},
    "pope_adversarial": {"result": lambda m: f"results/{m}_pope_adversarial.json", "id_key": "question_id"},
    "mathvista": {"result": lambda m: f"results/{m}_mathvista.json", "id_key": "pid"},
    "mathvista_cot": {"result": lambda m: f"results/{m}-cot_mathvista.json", "id_key": "pid"},
    "vqarad": {"result": lambda m: f"results/{m}_vqarad.json", "id_key": "id"},
}

RESPONSE_FILES = {
    "mathvista": lambda m: f"responses/{m}_mathvista.json",
    "mathvista_cot": lambda m: f"responses/{m}_mathvista_cot.json",
}


def _load_cache(path: str, cache: dict) -> dict:
    if path not in cache:
        cache[path] = load_json_object(path)
    return cache[path]


# ---------------------------------------------------------------------------
# E2: Threshold sensitivity
# ---------------------------------------------------------------------------


def run_threshold_analysis(cache: dict) -> dict:
    LOGGER.info("=" * 60)
    LOGGER.info("E2: Threshold sensitivity analysis")
    LOGGER.info("=" * 60)

    thresholds = list(range(7))
    all_curves = {}
    for model in MODELS:
        for dset, cfg in DATASETS.items():
            data = _load_cache(cfg["result"](model), cache)
            details = data.get("details", [])
            if not details:
                continue

            key = f"{model}@{dset}"
            scores = [d.get("score") for d in details if d.get("score") is not None]
            if not scores:
                continue

            n = len(scores)
            curve = {str(t): sum(1 for s in scores if s < t) / n for t in thresholds}
            all_curves[key] = {"n": n, "hr_curve": curve}

    out_path = os.path.join(OUTPUT_DIR, "threshold_sensitivity.json")
    save_json_atomic(all_curves, out_path)
    LOGGER.info("Saved to %s", out_path)
    return all_curves


# ---------------------------------------------------------------------------
# E3: Length bias
# ---------------------------------------------------------------------------


def run_length_bias(cache: dict) -> dict:
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("E3: Length bias analysis (answer length vs score)")
    LOGGER.info("=" * 60)

    results = {}

    for model in MODELS:
        for dset in ["mathvista", "mathvista_cot"]:
            cfg = DATASETS[dset]
            result_data = _load_cache(cfg["result"](model), cache)
            responses = load_json_object(RESPONSE_FILES[dset](model))

            details = result_data.get("details", [])
            if not details or not responses:
                continue

            id_key = cfg["id_key"]
            pairs = []
            for d in details:
                sid = str(d.get(id_key, ""))
                resp = responses.get(sid, "")
                score = d.get("score")
                if resp and score is not None:
                    pairs.append({"sample_id": sid, "response": resp, "score": score,
                                  "length": len(resp.split()), "has_h": d.get("has_hallucination", False)})

            if not pairs:
                continue

            lengths = [p["length"] for p in pairs]
            scores = [p["score"] for p in pairs]
            rho, p_val = spearmanr(lengths, scores)

            # Group by length quartiles
            sorted_pairs = sorted(pairs, key=lambda x: x["length"])
            n = len(sorted_pairs)
            q1 = sorted_pairs[:n // 4]
            q4 = sorted_pairs[3 * n // 4:]

            key = f"{model}@{dset}"
            results[key] = {
                "n": n,
                "spearman_rho": rho,
                "spearman_p": p_val,
                "median_length": sorted(lengths)[n // 2],
                "median_score": sorted(scores)[n // 2],
                "q1_avg_score": sum(p["score"] for p in q1) / len(q1) if q1 else 0,
                "q4_avg_score": sum(p["score"] for p in q4) / len(q4) if q4 else 0,
            }

            print(f"  {key:<35} n={n:<4} ρ={rho:<+8.4f} (p={p_val:.4f})  "
                  f"Q1_avg={results[key]['q1_avg_score']:.2f}  "
                  f"Q4_avg={results[key]['q4_avg_score']:.2f}")

    out_path = os.path.join(OUTPUT_DIR, "length_bias.json")
    save_json_atomic(results, out_path)
    LOGGER.info("Saved to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# E4: Cross-model consistency
# ---------------------------------------------------------------------------


def run_cross_model_consistency(cache: dict) -> dict:
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("E4: Cross-model hallucination consistency")
    LOGGER.info("=" * 60)

    results = {}

    for dset, cfg in DATASETS.items():
        sample_flags = defaultdict(list)
        id_key = cfg["id_key"]

        for model in MODELS:
            data = _load_cache(cfg["result"](model), cache)
            for d in data.get("details", []):
                sid = str(d.get(id_key, ""))
                h = d.get("has_hallucination", False)
                if isinstance(h, str):
                    h = h.lower() in ("true", "1", "yes")
                sample_flags[sid].append(int(h))

        overlapping = {sid: flags for sid, flags in sample_flags.items() if len(flags) == 4}
        if not overlapping:
            continue

        overlap_dist = Counter(sum(flags) for flags in overlapping.values())
        n_total = len(overlapping)

        results[dset] = {
            "n_samples": n_total,
            "overlap_distribution": {str(k): v for k, v in sorted(overlap_dist.items())},
            "hard_ratio": sum(1 for f in overlapping.values() if sum(f) >= 3) / n_total,
            "easy_ratio": sum(1 for f in overlapping.values() if sum(f) == 0) / n_total,
        }

        od = results[dset]["overlap_distribution"]
        print(f"  {dset:<25} n={n_total:<5} "
              f"0/4={od.get('0', 0):<5} 1/4={od.get('1', 0):<5} "
              f"2/4={od.get('2', 0):<5} 3/4={od.get('3', 0):<5} 4/4={od.get('4', 0):<5}")

    out_path = os.path.join(OUTPUT_DIR, "cross_model_consistency.json")
    save_json_atomic(results, out_path)
    LOGGER.info("Saved to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cache = {}
    results = {
        "threshold": run_threshold_analysis(cache),
        "length_bias": run_length_bias(cache),
        "cross_model": run_cross_model_consistency(cache),
    }

    LOGGER.info("\nAll analyses complete. Results in %s", OUTPUT_DIR)
    return results


if __name__ == "__main__":
    main()
