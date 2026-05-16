#!/usr/bin/env python3
"""
Generate model responses for POPE, MathVista, or VQA-RAD.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data import load_mathvista, load_pope_by_split, load_vqarad
from utils.api import ModelClient, call_vision_model_with_retries, create_model_client
from utils.batch import load_text_map, run_resumable_batch, save_json_atomic, sort_json_keys

LOGGER = logging.getLogger(__name__)

ID_KEYS = {"pope": "question_id", "mathvista": "pid", "vqarad": "id"}


def _default_output_path(dataset: str, model: str, pope_split: str) -> str:
    suffix = f"pope_{pope_split}" if dataset == "pope" else dataset
    return str(Path("responses") / f"{model}_{suffix}.json")


def _save_json(data: dict[str, str], path: str) -> None:
    save_json_atomic({k: data[k] for k in sort_json_keys(data.keys())}, path)


def _mathvista_format_instruction(sample: dict) -> str:
    question_type = sample.get("question_type", "")
    answer_type = sample.get("answer_type", "text")
    precision = sample.get("precision")
    unit = sample.get("unit")

    if question_type == "multi_choice":
        return "Answer with the option letter only."

    unit_suffix = f" Include the unit '{unit}'." if unit else ""

    if answer_type == "integer":
        return f"Answer with an integer.{unit_suffix}"
    if answer_type == "float":
        prec = int(precision) if precision is not None else 2
        return f"Answer with a number rounded to {prec} decimal place(s).{unit_suffix}"

    return "Keep your answer concise."


def _build_prompt(sample: dict, dataset: str, prompt_mode: str) -> str:
    question = sample["question"]

    if dataset == "pope":
        return f"{question}\nPlease answer YES or NO without an explanation."

    if dataset == "vqarad":
        return (
            "You are an expert radiologist.\n\n"
            f"{question}\n"
            "Please provide a detailed answer based on what you observe in the medical image."
        )

    choices = sample.get("choices")
    choices_text = ""
    if choices is not None:
        if hasattr(choices, "tolist"):
            choices = choices.tolist()
        elif not isinstance(choices, (list, tuple)):
            choices = [choices]
        choices = [c for c in choices if c is not None]
        if choices:
            choices_text = "\nChoices: " + ", ".join(str(c) for c in choices)

    format_instruction = _mathvista_format_instruction(sample)
    if prompt_mode == "cot":
        return (
            "Please generate a step-by-step answer, base your reasoning "
            "strictly on what is visible in the image, and avoid speculating "
            "about details that are unclear or not present. "
            "End with the final "
            "answer on a separate line starting with 'Final answer:'.\n\n"
            f"Question: {question}\n\n"
            f"{format_instruction}{choices_text}"
        )
    return f"Question: {question}\n\n{format_instruction}{choices_text}"


def _load_samples(dataset: str, pope_split: str, max_samples: Optional[int]) -> list[dict]:
    if dataset == "pope":
        return load_pope_by_split(split=pope_split, max_samples=max_samples)
    if dataset == "vqarad":
        return load_vqarad(max_samples=max_samples)
    return load_mathvista(max_samples=max_samples)


def _generate_one(args, client: ModelClient, dataset: str, id_key: str, sample: dict) -> tuple[str, str]:
    sid = str(sample[id_key])
    image_path = sample.get("image", "")
    if image_path and not os.path.exists(image_path):
        LOGGER.warning("  image not found: %s  %s", image_path, sid)
        return sid, ""

    prompt = _build_prompt(sample, dataset, args.prompt_mode)
    t0 = time.time()
    answer = call_vision_model_with_retries(
        client=client,
        prompt=prompt,
        image_path=sample["image"],
        temperature=args.temperature,
        retries=args.retries,
        timeout=args.timeout,
        max_tokens=client.max_tokens,
    )
    LOGGER.info("  [%s] %.1fs  %s", dataset, time.time() - t0, sid)
    return sid, answer


def generate_responses(args) -> None:
    client = create_model_client(args.model)
    output_path = args.output or _default_output_path(args.dataset, args.model, args.pope_split)
    samples = _load_samples(args.dataset, args.pope_split, args.max_samples)
    responses = load_text_map(output_path)
    id_key = ID_KEYS[args.dataset]

    LOGGER.info("Dataset: %s | Model: %s | Samples: %s", args.dataset, args.model, len(samples))
    LOGGER.info("Output: %s", output_path)

    responses = run_resumable_batch(
        items=samples,
        item_id=lambda sample: str(sample[id_key]),
        completed=responses,
        process_one=lambda sample: _generate_one(args, client, args.dataset, id_key, sample),
        save_completed=lambda completed: _save_json(completed, output_path),
        workers=args.workers,
        label="responses",
    )
    LOGGER.info("Done. Saved %s responses to %s", len(responses), output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate VLM responses for evaluation.")
    parser.add_argument("--dataset", required=True, choices=["pope", "mathvista", "vqarad"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pope-split", default="random", choices=["random", "popular", "adversarial"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt-mode", default="direct", choices=["direct", "cot"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()
    generate_responses(args)


if __name__ == "__main__":
    main()
