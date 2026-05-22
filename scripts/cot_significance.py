#!/usr/bin/env python3
"""
Paired significance tests for the Direct vs. CoT comparison on MathVista.

For each model, load Direct and CoT result JSONs, pair samples by `pid`,
compute the 2x2 contingency table of hallucination flags, and run a McNemar
test plus a Wald CI on ΔHR.

Also report Wilson 95% CIs for the raw HR in each condition.

Output: results/errors_analysis/cot_significance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import (
    mcnemar_test,
    paired_diff_ci,
    paired_flip_counts,
    wilson_ci,
)

RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = REPO_ROOT / "results" / "errors_analysis" / "cot_significance.json"

MODELS = [
    ("GPT-5.4-mini", "gpt-5.4-mini"),
    ("Gemini 2.5 Flash", "gemini-2.5-flash"),
    ("Qwen3.5-35B", "Qwen3.5-35B-A3B"),
    ("Qwen3-VL-235B", "Qwen3-VL-235B-A22B-Instruct"),
]


def _flags_from_results(path: Path) -> dict[str, int]:
    """Return pid → 0/1 hallucination flag, skipping samples without a flag."""
    with open(path) as f:
        data = json.load(f)
    flags: dict[str, int] = {}
    for item in data.get("details", []):
        has_h = item.get("has_hallucination")
        if has_h is None:
            continue
        flags[str(item["pid"])] = 1 if has_h else 0
    return flags


def analyze_model(display: str, slug: str) -> dict:
    direct_path = RESULTS_DIR / f"{slug}_mathvista.json"
    cot_path = RESULTS_DIR / f"{slug}-cot_mathvista.json"
    if not direct_path.exists() or not cot_path.exists():
        return {"model": display, "error": f"missing result files for {slug}"}

    direct = _flags_from_results(direct_path)
    cot = _flags_from_results(cot_path)
    a, b, c, d, pids = paired_flip_counts(direct, cot)
    n = len(pids)

    hr_direct = (c + d) / n if n else 0.0
    hr_cot = (b + d) / n if n else 0.0
    delta = hr_cot - hr_direct

    direct_lo, direct_hi = wilson_ci(c + d, n)
    cot_lo, cot_hi = wilson_ci(b + d, n)
    diff_lo, diff_hi = paired_diff_ci(b, c, n)
    mc = mcnemar_test(b, c)

    return {
        "model": display,
        "slug": slug,
        "n_paired": n,
        "contingency": {
            "a_both_clean": a,
            "b_cot_only_h": b,
            "c_direct_only_h": c,
            "d_both_h": d,
        },
        "hr_direct": hr_direct,
        "hr_direct_ci": [direct_lo, direct_hi],
        "hr_cot": hr_cot,
        "hr_cot_ci": [cot_lo, cot_hi],
        "delta_hr": delta,
        "delta_hr_ci": [diff_lo, diff_hi],
        "mcnemar": mc,
    }


def _fmt_pct(x: float, lo: float, hi: float) -> str:
    return f"{x*100:.1f}%  [{lo*100:.1f}, {hi*100:.1f}]"


def _fmt_delta(x: float, lo: float, hi: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x)*100:.1f}pp  [{lo*100:+.1f}, {hi*100:+.1f}]"


def main() -> None:
    rows = [analyze_model(display, slug) for display, slug in MODELS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"models": rows}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}\n")
    header = (
        f"{'Model':22} {'N':>5}  "
        f"{'Direct HR [95% CI]':>26}  {'CoT HR [95% CI]':>26}  "
        f"{'ΔHR [95% CI]':>24}  {'McNemar':>22}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['model']:22}  ERROR: {row['error']}")
            continue
        mc = row["mcnemar"]
        direct_cell = _fmt_pct(row["hr_direct"], *row["hr_direct_ci"])
        cot_cell = _fmt_pct(row["hr_cot"], *row["hr_cot_ci"])
        delta_cell = _fmt_delta(row["delta_hr"], *row["delta_hr_ci"])
        mc_cell = f"χ²={mc['chi2']:.2f}, p={mc['p_value']:.3g}"
        print(f"{row['model']:22} {row['n_paired']:>5}  "
              f"{direct_cell:>26}  {cot_cell:>26}  {delta_cell:>24}  {mc_cell:>22}")


if __name__ == "__main__":
    main()
