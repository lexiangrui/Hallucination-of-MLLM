#!/usr/bin/env python3
"""
Hallucination detection for MLLMs.

Dataset → Method mapping:
  POPE      → 规则判断法
  MathVista → MLLM Judge
  VQA-RAD   → MLLM Judge
"""

import argparse
import logging
import os
from typing import Optional

from configs import config
from evaluation import run_mathvista, run_pope, run_vqarad


METHOD_MAP = {"pope": "规则判断法", "mathvista": "mllm-judge", "vqarad": "mllm-judge"}
LOGGER = logging.getLogger(__name__)


def _parse_response_files(entries: Optional[list[str]]) -> dict[str, str]:
    """Parse response mappings in the single supported form: model:dataset=path."""
    mapping = {}
    paths_seen = {}

    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --response-files entry {entry!r}; expected model:dataset=path"
            )
        key, path = entry.split("=", 1)
        key = key.strip()
        path = path.strip()
        if ":" not in key or not path:
            raise ValueError(
                f"Invalid --response-files entry {entry!r}; expected model:dataset=path"
            )
        model_name, dataset = [part.strip() for part in key.split(":", 1)]
        if not model_name or dataset not in METHOD_MAP:
            raise ValueError(
                f"Invalid --response-files key {key!r}; dataset must be pope, mathvista, or vqarad"
            )

        normalized_key = f"{model_name}:{dataset}"
        normalized_path = os.path.abspath(path)
        if normalized_key in mapping:
            raise ValueError(f"Duplicate --response-files key: {normalized_key}")
        if normalized_path in paths_seen:
            raise ValueError(
                f"Response file {path} is reused by both {paths_seen[normalized_path]} "
                f"and {normalized_key}. Provide a separate response file for each run."
            )

        mapping[normalized_key] = path
        paths_seen[normalized_path] = normalized_key

    return mapping


def _response_file_for(
    response_files: dict[str, str],
    model_name: str,
    dataset: str,
) -> str:
    key = f"{model_name}:{dataset}"
    if key not in response_files:
        raise ValueError(
            f"No response file mapping for {key}. "
            f"Use --response-files {key}=path.json"
        )
    return response_files[key]


def _run_one(
    *,
    model_name: str,
    dataset: str,
    response_file: str,
    max_samples: Optional[int],
    workers: int,
    pope_split: Optional[str] = None,
) -> dict:
    if dataset == "pope":
        return run_pope(
            model_name=model_name,
            response_file=response_file,
            max_samples=max_samples,
            pope_split=pope_split or "random",
        )
    if dataset == "mathvista":
        return run_mathvista(
            model_name=model_name,
            response_file=response_file,
            max_samples=max_samples,
            workers=workers,
        )
    if dataset == "vqarad":
        return run_vqarad(
            model_name=model_name,
            response_file=response_file,
            max_samples=max_samples,
            workers=workers,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def _summary_key(
    model_name: str,
    dataset: str,
    pope_split: Optional[str] = None,
) -> str:
    if dataset == "pope":
        return f"{model_name}@pope/{pope_split or 'random'}"
    return f"{model_name}@{dataset}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="MLLM Hallucination Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset → Method:
  POPE      → 规则判断法
  MathVista → MLLM Judge
  VQA-RAD   → MLLM Judge

Examples:
  python main.py --dataset pope --model gpt-5.4 --response-files gpt-5.4:pope=responses/gpt54_pope_random.json
  python main.py --dataset mathvista --model gpt-5.4 --response-files gpt-5.4:mathvista=responses/gpt54_mathvista.json --workers 4
  python main.py --dataset vqarad --model gpt-5.4 --response-files gpt-5.4:vqarad=responses/gpt54_vqarad.json --workers 4
  python main.py --dataset all --response-files gpt-5.4:pope=responses/gpt54_pope_random.json gpt-5.4:mathvista=responses/gpt54_mathvista.json gpt-5.4:vqarad=responses/gpt54_vqarad.json --max-samples 100
        """,
    )
    parser.add_argument("--dataset", default="pope",
                        choices=["pope", "mathvista", "vqarad", "all"])
    parser.add_argument("--model", default=None,
                        help="Model to evaluate (default: all in config.MODELS)")
    parser.add_argument("--response-files", nargs="+", required=True,
                        help="Response mappings in the form model:dataset=path.json")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="每个数据集最多使用的样本数（默认 1000，seed=42 随机采样）",
    )
    parser.add_argument("--pope-split", default="random",
                        choices=["random", "popular", "adversarial"],
                        help="POPE sampling setting to evaluate")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent MLLM Judge requests for MathVista")
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    models = [args.model] if args.model else config.MODELS
    datasets = [args.dataset] if args.dataset != "all" else ["pope", "mathvista", "vqarad"]
    response_files = _parse_response_files(args.response_files)

    all_results = {}
    for dataset in datasets:
        for model_name in models:
            pope_split = args.pope_split if dataset == "pope" else None
            key = _summary_key(model_name, dataset, pope_split)
            LOGGER.info("\n%s", "#" * 60)
            LOGGER.info("# %s  [%s]", key, METHOD_MAP[dataset])
            LOGGER.info("#" * 60)

            result = _run_one(
                model_name=model_name,
                dataset=dataset,
                response_file=_response_file_for(response_files, model_name, dataset),
                max_samples=args.max_samples,
                pope_split=pope_split,
                workers=args.workers,
            )
            if result:
                all_results[key] = result

    if len(all_results) > 1:
        LOGGER.info("\n%s", "=" * 60)
        LOGGER.info("SUMMARY")
        LOGGER.info("=" * 60)
        for key, result in all_results.items():
            metrics = result["metrics"]
            if result["dataset"] == "pope":
                LOGGER.info(
                    "  %s: F1=%.4f, Acc=%.4f, Yes Ratio=%.4f, HR=%.4f",
                    key,
                    metrics.get("f1", 0),
                    metrics.get("accuracy", 0),
                    metrics.get("yes_ratio", 0),
                    metrics.get("object_hallucination_rate", 0),
                )
            else:
                LOGGER.info(
                    "  %s: HR=%.4f, Avg Score=%.2f",
                    key,
                    metrics.get("hallucination_rate", 0),
                    metrics.get("avg_score", 0),
                )


if __name__ == "__main__":
    main()
