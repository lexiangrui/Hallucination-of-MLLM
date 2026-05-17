#!/usr/bin/env python3
"""
Analyze human-vs-GPT-Judge alignment for the MathVista human-eval sample.

Inputs:
- results/errors_analysis/human_alignment/annotations.csv
- results/errors_analysis/human_alignment/meta.json

Output:
- results/errors_analysis/human_alignment/report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import cohens_kappa, confusion_matrix

DEFAULT_EVAL_DIR = REPO_ROOT / "results" / "errors_analysis" / "human_alignment"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "errors_analysis" / "human_alignment"
TYPE_ORDER = ["faithfulness", "factuality", "logical"]


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_annotations(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_int(value: str, field: str, sample_id: str) -> int | None:
    value = str(value or "").strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} for {sample_id} must be an integer, got {value!r}") from exc


def _join_rows(rows: list[dict[str, str]], meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    joined = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        if sample_id not in meta:
            raise ValueError(f"Missing sample_id in meta.json: {sample_id}")

        human_label = _parse_int(row.get("human_label", ""), "human_label", sample_id)
        human_type = str(row.get("human_type", "")).strip()

        if human_label is None and human_type == "":
            continue
        if human_label not in (0, 1):
            raise ValueError(f"human_label for {sample_id} must be 0 or 1")

        record = dict(row)
        record.update(meta[sample_id])
        record["human_label"] = human_label
        record["human_type"] = human_type if human_type in TYPE_ORDER else "none"
        record["gpt_has_h"] = int(record["gpt_has_h"])
        joined.append(record)

    return joined


def _alignment_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    human_labels = [int(row["human_label"]) for row in records]
    gpt_labels = [int(row["gpt_has_h"]) for row in records]

    cm = confusion_matrix(human_labels, gpt_labels)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    kappa = cohens_kappa(human_labels, gpt_labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohens_kappa": kappa,
    }


def _format_metrics_table(title: str, rows: list[tuple[str, list[dict[str, Any]]]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| 子集 | N | Accuracy | Precision | Recall | F1 | Cohen's Kappa |",
        "|------|---:|---------:|----------:|-------:|---:|--------------:|",
    ]
    for name, subset in rows:
        metrics = _alignment_metrics(subset)
        if not metrics:
            lines.append(f"| {name} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            "| {name} | {n} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {cohens_kappa:.4f} |".format(
                name=name,
                n=len(subset),
                **metrics,
            )
        )
    lines.append("")
    return lines


def _type_confusion(records: list[dict[str, Any]]) -> tuple[list[list[int]], float]:
    matrix = [[0 for _ in TYPE_ORDER] for _ in TYPE_ORDER]
    total = 0
    agree = 0
    for row in records:
        human_type = row["human_type"]
        gpt_type = row["gpt_type"]
        if human_type not in TYPE_ORDER or gpt_type not in TYPE_ORDER:
            continue
        i = TYPE_ORDER.index(human_type)
        j = TYPE_ORDER.index(gpt_type)
        matrix[i][j] += 1
        total += 1
        if i == j:
            agree += 1
    return matrix, (agree / total if total else 0.0)


def _format_type_confusion(records: list[dict[str, Any]]) -> list[str]:
    matrix, agreement = _type_confusion(records)
    lines = [
        "### 幻觉类型一致性",
        "",
        f"Type agreement: {agreement:.4f} （仅限人和 GPT 均判定有幻觉的样本）",
        "",
        "| Human \\ GPT | faithfulness | factuality | logical |",
        "|-------------|-------------:|----------:|--------:|",
    ]
    for human_type, row in zip(TYPE_ORDER, matrix):
        lines.append(f"| {human_type} | {row[0]} | {row[1]} | {row[2]} |")
    lines.append("")
    return lines


def _format_mismatches(records: list[dict[str, Any]]) -> list[str]:
    mismatches = [
        row for row in records
        if int(row["human_label"]) != int(row["gpt_has_h"])
    ]
    lines = [
        "### Mismatch 清单",
        "",
        f"Total mismatches: {len(mismatches)}",
        "",
        "| sample_id | model | pid | human_label | gpt_has_h | human_type | gpt_type | notes |",
        "|-----------|-------|-----|------------:|----------:|------------|----------|-------|",
    ]
    for row in mismatches:
        notes = str(row.get("notes", "")).replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {row['sample_id']} | {row['model']} | {row['pid']} | "
            f"{row['human_label']} | {row['gpt_has_h']} | {row['human_type']} | "
            f"{row['gpt_type']} | {notes} |"
        )
    lines.append("")
    return lines


def _format_breakdowns(records: list[dict[str, Any]]) -> list[str]:
    lines = []
    by_model = defaultdict(list)
    by_model_type = defaultdict(list)
    by_model_gpt_type = defaultdict(list)

    for row in records:
        by_model[row["model"]].append(row)
        by_model_type[(row["model"], row["human_type"])].append(row)
        by_model_gpt_type[(row["model"], row["gpt_type"])].append(row)

    lines.extend(_format_metrics_table(
        "整体与按模型指标",
        [("overall", records)] + [(model, by_model[model]) for model in sorted(by_model)],
    ))

    lines.extend(_format_metrics_table(
        "按模型 × Human Type",
        [
            (f"{model} / {htype}", by_model_type[(model, htype)])
            for model in sorted(by_model)
            for htype in TYPE_ORDER
            if by_model_type[(model, htype)]
        ],
    ))

    lines.extend(_format_metrics_table(
        "按模型 × GPT Type",
        [
            (f"{model} / {gtype}", by_model_gpt_type[(model, gtype)])
            for model in sorted(by_model)
            for gtype in TYPE_ORDER
            if by_model_gpt_type[(model, gtype)]
        ],
    ))
    return lines


def _format_error_mode_template() -> list[str]:
    return [
        "### 错误模式归因表",
        "",
        "该表用于人工复核 mismatch 后填写。",
        "",
        "| 模式 | 样本数 | 说明 | 典型样例 ID |",
        "|------|-------:|------|------------|",
        "| GPT Judge 漏检 |  |  |  |",
        "| GPT Judge 过判 |  |  |  |",
        "| 类型错位 |  |  |  |",
        "| 图像/问题歧义 |  |  |  |",
        "| 其他 |  |  |  |",
        "",
    ]


def _write_report(records: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Human-as-Judge 对齐实验报告",
        "",
        f"有效人工标注样本数：{len(records)}",
        "",
    ]
    lines.extend(_format_breakdowns(records))
    lines.extend(_format_type_confusion(records))
    lines.extend(_format_mismatches(records))
    lines.extend(_format_error_mode_template())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human alignment annotations.")
    parser.add_argument("--annotations", default=str(DEFAULT_EVAL_DIR / "annotations.csv"))
    parser.add_argument("--meta", default=str(DEFAULT_EVAL_DIR / "meta.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "report.md"))
    args = parser.parse_args()

    annotations = _load_annotations(Path(args.annotations))
    meta = _load_json(Path(args.meta))
    records = _join_rows(annotations, meta)
    if not records:
        raise ValueError("No completed annotations found.")

    _write_report(records, Path(args.output))
    print(f"Wrote report for {len(records)} annotated samples to {args.output}")


if __name__ == "__main__":
    main()
