"""
GPT Judge hallucination detection.

Uses GPT-5.5 as an automated evaluator following the MMHal-Bench protocol:
  Sun et al., "Aligning Large Multimodal Models with Factually Augmented
  RLHF," ACL Findings 2024. https://arxiv.org/abs/2309.14525

Key design:
- 0-6 Likert scale jointly measuring informativeness and hallucination
- score < 3  → with hallucination
- score >= 3 → no hallucination
- Hallucination Rate = N_{score<3} / N_total
- Always classifies hallucination into faithfulness / factuality / logical types
- Raises on API/parse errors — callers handle exceptions, not error data
"""

import json
from typing import Literal, Optional

from utils.api import call_vision_model_with_retries, create_model_client


HALLUCINATION_TYPES = {"faithfulness", "factuality", "logical", "none", "error"}

_JUDGE_FEWSHOT_AND_REMINDER = """## Few-Shot Examples

**Example 1 — Hallucination (Score 2):**
Image shows two cats. Question: "How many cats are in the image?" Model response: "There is one cat in the image." → The response is informative but miscounts a visible object. Score: 2 (Very informative, WITH hallucination).

**Example 2 — No Hallucination (Score 5):**
Image shows a panda sitting on a tree branch with green leaves. Question: "What is the panda doing?" Model response: "The panda is sitting on a tree branch, surrounded by leaves." → The response is accurate and makes no false claims about the image. Score: 5 (Very informative, NO hallucination).

**Example 3 — Hallucination (Score 2):**
Image shows a dog holding an umbrella, with no water in sight. Question: "Describe the scene." Model response: "A dog holding an umbrella, positioned near the water." → The response fabricates "water" which is not in the image. Score: 2 (Very informative, WITH hallucination).

**Example 4 — No Hallucination (Score 3):**
Image shows a complex diagram or chart. Question: "Explain the economic trend." Model response: "I cannot determine the specific trend from this image. The chart appears to show some data over time." → The response honestly admits uncertainty without making false claims. Score: 3 (Not informative, NO hallucination).

**Example 5 — No Hallucination (Score 6):**
Image shows industrial smokestacks. Question: "What environmental concerns does this image raise?" Model response provides a detailed analysis of potential emissions, air quality impacts, and relevant environmental regulations, all grounded in visible evidence. → The detailed reasoning is based on what is visible and makes no false claims. Score: 6 (Very informative with good analysis, NO hallucination).

## Important Reminder
The standard human-generated answer may only contain factual information without detailed analysis, and may not be fully comprehensive. A model response that provides detailed reasoning or analysis grounded in the visual evidence should be encouraged, not penalized. Only penalize claims that are clearly inconsistent with or absent from the image. If the model provides extra detail that is factually correct and consistent with the image, do NOT treat it as hallucination."""

_RATING_SCALE = """You are an impartial evaluator for multimodal language models. Your task is to rate a model's response to a visual question on a 0-6 scale, jointly considering informativeness and hallucination.

## Rating Scale (0-6)
- **6**: Very informative, with good analysis/reasoning, NO hallucination
- **5**: Very informative, NO hallucination
- **4**: Somewhat informative, NO hallucination
- **3**: Not informative (e.g., "I don't know"), but NO hallucination
- **2**: Very informative, WITH hallucination
- **1**: Somewhat informative, WITH hallucination
- **0**: Not informative, WITH hallucination

## Hallucination Definition
Hallucination refers to situations where the response includes information not present or implied in the image, or makes claims inconsistent with the visual content. A response that honestly admits inability to answer (e.g., "I cannot determine") has NO hallucination (score 3).

"""

_JUDGE_TYPE_EXTRA = """## Hallucination Types (only when score < 3)
- **faithfulness**: Visual inconsistency — describes objects/attributes/relations not in the image
- **factuality**: Contradiction with established world knowledge
- **logical**: Reasoning error — conclusion does not follow from evidence or self-contradiction

## Output Format
Reply in JSON only:
{
    "score": 0-6,
    "has_hallucination": true/false,
    "hallucination_type": "faithfulness" | "factuality" | "logical" | "none",
    "reason": "Brief explanation in English"
}"""

JUDGE_SYSTEM_PROMPT = _RATING_SCALE + _JUDGE_FEWSHOT_AND_REMINDER + "\n\n" + _JUDGE_TYPE_EXTRA


def build_judge_prompt(
    question: str,
    model_response: str,
    ground_truth: str,
    dataset: Literal["pope", "mathvista", "ocr"],
) -> str:
    """Build the evaluation prompt."""
    if dataset == "pope":
        task_desc = "The question is a Yes/No object existence query."
    elif dataset == "mathvista":
        task_desc = "The question requires mathematical visual reasoning (chart, diagram, or geometry)."
    else:
        task_desc = (
            "The question requires optical character recognition (OCR) "
            "from an image — reading text in scene images, documents, "
            "or handwritten content. Carefully check whether the model "
            "correctly reads the text visible in the image."
        )

    return f"""{task_desc}

Question: {question}
Model Response: {model_response}
Ground Truth Answer: {ground_truth}

Please rate the model's response on the 0-6 scale."""


# ==================== GPT Judge Client ====================

class GPTJudge:
    """GPT-5.5 based hallucination judge following MMHal-Bench protocol."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        hallucination_threshold: int = 3,
    ):
        from configs.config import (
            GPT_JUDGE_MODEL, GPT_JUDGE_TEMPERATURE,
        )

        self.model = model if model is not None else GPT_JUDGE_MODEL
        self.client = create_model_client(self.model)
        self.temperature = temperature if temperature is not None else GPT_JUDGE_TEMPERATURE
        self.hallucination_threshold = hallucination_threshold

    def judge(
        self,
        image_path: str,
        question: str,
        model_response: str,
        ground_truth: str,
        dataset: Literal["pope", "mathvista", "ocr"],
        retries: int = 3,
    ) -> dict:
        """
        Evaluate a model response using MMHal-Bench 0-6 scoring.

        Returns:
            Dict with: score, has_hallucination, hallucination_type, reason.

        Raises:
            ValueError on API or parse failure — never returns error data.
        """
        prompt = build_judge_prompt(question, model_response, ground_truth, dataset)

        raw = call_vision_model_with_retries(
            client=self.client,
            prompt=prompt,
            image_path=image_path,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=self.temperature,
            require_image=False,
            retries=retries,
        )
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> dict:
        """Parse JSON response, with fallback for markdown code fences."""
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                start = raw.index("{")
                end = raw.rindex("}")
                candidate = raw[start:end + 1]
                parsed = json.loads(candidate)
            except (ValueError, json.JSONDecodeError):
                raise ValueError(f"Failed to parse judge response as JSON: {raw[:200]}")

        if not parsed:
            raise ValueError(f"Empty judge response: {raw[:200]}")

        score = parsed.get("score", None)
        if score is not None and isinstance(score, (int, float)):
            has_hallucination = score < self.hallucination_threshold
        else:
            has_hallucination = parsed.get("has_hallucination", None)
            if isinstance(has_hallucination, str):
                has_hallucination = has_hallucination.lower() in ("true", "1", "yes")

        hallucination_type = parsed.get("hallucination_type", "error")
        if has_hallucination is False:
            hallucination_type = "none"
        if hallucination_type not in HALLUCINATION_TYPES:
            hallucination_type = "error"

        return {
            "score": score,
            "has_hallucination": has_hallucination,
            "hallucination_type": hallucination_type,
            "reason": parsed.get("reason", raw[:200]),
            "raw_response": raw,
        }
