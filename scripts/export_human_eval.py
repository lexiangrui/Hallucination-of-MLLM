#!/usr/bin/env python3
"""
Export a blind human-alignment annotation sheet for MathVista COT.

Sampling strategy (分层抽样):
- 模型: gpt-5.4-mini, gemini-2.5-flash, Qwen3.5-35B-A3B, Qwen3-VL-235B-A22B (×4)
- GPT Judge 判定: 有幻觉 / 无幻觉 (×2)
- 每组每模型抽样数可配 (默认 5)，覆盖全部 3 类幻觉类型 (faithfulness / factuality / logical)
- seed=42 固定可复现

Outputs:
- results/errors_analysis/human_alignment/samples.csv      blind annotation sheet
- results/errors_analysis/human_alignment/annotations.csv  editable copy for annotators
- results/errors_analysis/human_alignment/meta.json        GPT Judge predictions keyed by sample_id
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.config import MODELS

# Default sources derived from config.MODELS — add "-cot" suffix for the COT evaluation variant.
DEFAULT_RESULTS = [
    (f"{m}-cot", REPO_ROOT / "results" / f"{m}-cot_mathvista.json")
    for m in MODELS
]
DEFAULT_RESPONSES = {
    f"{m}-cot": REPO_ROOT / "responses" / f"{m}_mathvista_cot.json"
    for m in MODELS
}
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "errors_analysis" / "human_alignment"
HAL_TYPES = {"faithfulness", "factuality", "logical"}

SAMPLE_FIELDS = [
    "sample_id",
    "model",
    "pid",
    "image",
    "question",
    "gt_answer",
    "model_response",
    "human_label",
    "human_type",
]


def _safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_response_map(path: Path) -> dict[str, str]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Response file must be a JSON object: {path}")
    return {str(k): "" if v is None else str(v) for k, v in data.items()}


def _load_mathvista_lookup() -> dict[str, dict[str, Any]]:
    from data import load_mathvista

    return {str(sample["pid"]): sample for sample in load_mathvista()}


def _parse_result_source(entry: str) -> tuple[str, Path]:
    if "=" not in entry:
        raise ValueError(f"Invalid --result entry {entry!r}; expected model=path.json")
    model, path = entry.split("=", 1)
    model = model.strip()
    path = path.strip()
    if not model or not path:
        raise ValueError(f"Invalid --result entry {entry!r}; expected model=path.json")
    return model, Path(path)


def _flatten_records(
    sources: list[tuple[str, Path]],
    response_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    mathvista_by_pid = _load_mathvista_lookup()
    records = []

    for model, result_path in sources:
        data = _load_json(result_path)
        if model not in response_paths:
            raise ValueError(f"Missing response path for model: {model}")
        responses = _load_response_map(response_paths[model])
        for detail in data.get("details", []):
            gpt_type = str(detail.get("hallucination_type", "error"))
            if gpt_type == "error":
                continue

            pid = str(detail.get("pid", ""))
            if pid not in mathvista_by_pid:
                raise ValueError(f"Missing MathVista sample for pid={pid}")
            if pid not in responses:
                raise ValueError(f"Missing model response for {model} pid={pid}")

            sample = mathvista_by_pid[pid]
            sample_id = f"{_safe_model_name(model)}__{pid}"
            gpt_has_h = detail.get("has_hallucination")

            records.append({
                "sample_id": sample_id,
                "model": model,
                "pid": pid,
                "image": sample["image"],
                "question": sample["question"],
                "gt_answer": sample["answer"],
                "model_response": responses[pid],
                "gpt_has_h": None if gpt_has_h is None else int(bool(gpt_has_h)),
                "gpt_type": gpt_type if gpt_type in HAL_TYPES else "none",
                "gpt_score": detail.get("score"),
            })

    return records


def _sample_with_type_coverage(
    records: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    types: set[str],
) -> list[dict[str, Any]]:
    """Sample n records while maximizing coverage of the given types."""
    if n >= len(records):
        return list(records)

    # Prioritise one sample per type
    chosen: list[dict[str, Any]] = []
    seen_ids = set()
    for t in sorted(types):
        candidates = [r for r in records if r["gpt_type"] == t and r["sample_id"] not in seen_ids]
        if candidates:
            pick = rng.choice(candidates)
            chosen.append(pick)
            seen_ids.add(pick["sample_id"])

    # Fill remainder with random from untouched records
    remaining = [r for r in records if r["sample_id"] not in seen_ids]
    needed = n - len(chosen)
    if needed > 0:
        chosen.extend(rng.sample(remaining, min(needed, len(remaining))))

    return chosen[:n]


def _sample_records(
    records: list[dict[str, Any]],
    per_cell: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Stratified sampling by (model, gpt_has_h) with type coverage."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["gpt_has_h"])].append(record)

    rng = random.Random(seed)
    selected = []
    for key in sorted(grouped.keys()):
        model, has_h = key
        group = sorted(grouped[key],
                       key=lambda row: int(row["pid"]) if row["pid"].isdigit() else row["pid"])
        if len(group) < per_cell:
            raise ValueError(
                f"Not enough samples for stratum {key}: {len(group)} < {per_cell}"
            )
        if has_h == 1:
            # Type-aware sampling for hallucination group
            chosen = _sample_with_type_coverage(group, per_cell, rng, HAL_TYPES)
        else:
            chosen = rng.sample(group, per_cell)
        selected.extend(chosen)

    return sorted(selected, key=lambda row: (row["model"], row["gpt_has_h"],
                                              int(row["pid"]) if row["pid"].isdigit() else row["pid"]))


def _csv_cell(value: Any) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def _write_samples_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for record in records:
            row = {field: _csv_cell(record.get(field, "")) for field in SAMPLE_FIELDS}
            row["human_label"] = ""
            row["human_type"] = ""
            writer.writerow(row)


def _write_meta_json(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        record["sample_id"]: {
            "model": record["model"],
            "pid": record["pid"],
            "gpt_score": record["gpt_score"],
            "gpt_has_h": record["gpt_has_h"],
            "gpt_type": record["gpt_type"],
        }
        for record in records
    }
    with open(path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blind human-eval samples for MathVista COT.")
    parser.add_argument("--result", action="append", default=None,
                        help="Result source in model=path.json form; can be repeated")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--per-cell", type=int, default=5,
                        help="Samples per (model, has_hallucination) stratum")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.per_cell <= 0:
        raise ValueError("--per-cell must be positive")

    sources = (
        [_parse_result_source(item) for item in args.result]
        if args.result
        else DEFAULT_RESULTS
    )
    out_dir = Path(args.out_dir)

    records = _flatten_records(sources, DEFAULT_RESPONSES)
    selected = _sample_records(records, per_cell=args.per_cell, seed=args.seed)

    samples_path = out_dir / "samples.csv"
    annotations_path = out_dir / "annotations.csv"
    meta_path = out_dir / "meta.json"

    _write_samples_csv(selected, samples_path)
    _write_samples_csv(selected, annotations_path)
    _write_meta_json(selected, meta_path)

    n_models = len(set(r["model"] for r in selected))
    print(f"Models:       {n_models}")
    print(f"Total samples:{len(selected)}")
    print(f"Samples  →   {samples_path}")
    print(f"Annotations → {annotations_path}")
    print(f"Meta     →   {meta_path}")


if __name__ == "__main__":
    main()
