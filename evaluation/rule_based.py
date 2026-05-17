"""
Rule-based POPE hallucination detection.
"""


def normalize_pope_answer_rule(text: str) -> str:
    """
    Normalize a POPE answer with a rule adapted from RUCAIBox/POPE evaluate.py.

    Rule:
    - keep only the first sentence before "."
    - remove commas
    - split by spaces
    - if one of {"No", "not", "no"} appears, predict "no"; otherwise "yes"
    """
    if "." in text:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split()
    if any(w.lower() in {"no", "not"} for w in words):
        return "no"
    return "yes"


def detect_pope_hallucination(model_response: str, ground_truth: str) -> dict:
    """
    Detect object hallucination in POPE.

    POPE object hallucination is a false positive: the model answers "yes"
    while the ground truth label is "no".
    """
    predicted = normalize_pope_answer_rule(model_response)
    is_correct = predicted == ground_truth
    is_hallucination = predicted == "yes" and ground_truth == "no"

    return {
        "is_hallucination": is_hallucination,
        "is_correct": is_correct,
        "predicted": predicted,
        "method": "rule-based",
        "detail": f"Predicted '{predicted}', ground truth '{ground_truth}'",
    }
