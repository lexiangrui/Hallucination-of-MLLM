"""
Evaluation metrics for hallucination detection.

Metrics:
- Accuracy  — (TP + TN) / Total
- Precision — TP / (TP + FP)
- Recall    — TP / (TP + FN)
- F1        — 2 * P * R / (P + R)
- Hallucination Rate  — proportion flagged as hallucinatory by GPT Judge

Hallucination detection is treated as a binary classification:
- Positive class = hallucination detected
- Ground truth label comes from:
  - Rule-based: comparison with dataset ground truth
  - Human annotation: gold labels
- For GPT Judge vs. ground truth comparison, we treat GPT Judge
  predictions as the detection result and ground truth as the label.
"""

def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """
    Compute confusion matrix elements.
    y_true[i] = 1 if sample i IS a hallucination (ground truth)
    y_pred[i] = 1 if detector flags sample i as hallucination
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Accuracy = (TP + TN) / Total."""
    cm = confusion_matrix(y_true, y_pred)
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    if total == 0:
        return 0.0
    return (cm["TP"] + cm["TN"]) / total


def precision(y_true: list[int], y_pred: list[int]) -> float:
    """Precision = TP / (TP + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FP"]
    if denom == 0:
        return 0.0
    return cm["TP"] / denom


def recall(y_true: list[int], y_pred: list[int]) -> float:
    """Recall = TP / (TP + FN)."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FN"]
    if denom == 0:
        return 0.0
    return cm["TP"] / denom


def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    """F1 = 2 * Precision * Recall / (Precision + Recall)."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_pope_metrics(label_list: list[str], pred_list: list[str]) -> dict:
    """
    Compute POPE official binary-classification metrics.

    Positive class = "yes" (object exists). In POPE, object hallucinations are
    false positives: the model answers "yes" when the label is "no".
    """
    y_true = [0 if label == "no" else 1 for label in label_list]
    y_pred = [0 if pred == "no" else 1 for pred in pred_list]

    cm = confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = tp + fp + tn + fn

    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "yes_ratio": y_pred.count(1) / len(y_pred) if y_pred else 0.0,
        "object_hallucination_rate": fp / total if total else 0.0,
        "n_object_hallucinated": fp,
        "total_samples": total,
        "confusion_matrix": cm,
    }


def hallucination_rate(gpt_flags: list[int | None]) -> float:
    """
    Hallucination Rate — defined in MMHal-Bench (Sun et al., 2024).

    Hallucination Rate = N_{score<3} / N_total

    GPT-4 rates each response on a 0-6 Likert scale. Scores < 3 (i.e.,
    0/1/2) indicate "with hallucination", scores >= 3 "no hallucination".
    The rate is the proportion of hallucinated responses.

    This metric appears as "Hallucination Rate" in Table 6 of the paper;
    the exact thresholding logic (score < 3 = hallucination) is from the
    official eval script at https://github.com/Shengcao1006/MMHal-Bench.

    Reference:
        Sun et al., "Aligning Large Multimodal Models with Factually
        Augmented RLHF," ACL Findings, 2024. arXiv:2309.14525.

    Args:
        gpt_flags: List where 1 = hallucination, 0 = no hallucination,
                   None = evaluation error.

    Returns:
        Float in [0, 1]. Lower is better.
    """
    valid = [f for f in gpt_flags if f is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


# ==================== Human Alignment Metrics ====================
# These are used specifically in human-as-judge experiments to compare
# automated detection results against human-annotated gold labels.

def cohens_kappa(y_true: list[int], y_pred: list[int]) -> float:
    """
    Cohen's Kappa — inter-rater agreement corrected for chance.

    kappa = (p_o - p_e) / (1 - p_e)

    where p_o = observed agreement proportion,
          p_e = expected agreement by chance.

    Interpretation:
        < 0.00  — worse than chance
        0.00–0.20 — slight
        0.21–0.40 — fair
        0.41–0.60 — moderate
        0.61–0.80 — substantial
        0.81–1.00 — almost perfect
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    # Confusion matrix as counts
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    # Observed agreement
    p_o = (tp + tn) / n

    # Expected agreement
    # P(judge says positive) * P(true is positive) + P(judge says negative) * P(true is negative)
    p_positive_judge = (tp + fp) / n
    p_positive_true = (tp + fn) / n
    p_negative_judge = (tn + fn) / n
    p_negative_true = (tn + fp) / n
    p_e = (p_positive_judge * p_positive_true) + (p_negative_judge * p_negative_true)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """
    Pearson correlation coefficient — measures linear correlation
    between two sets of scores (e.g., GPT scores vs. human scores).

    r = Σ((x_i - x̄)(y_i - ȳ)) / sqrt(Σ(x_i - x̄)² * Σ(y_i - ȳ)²)

    Used in human-alignment experiments to assess whether automated
    scores track human judgments.
    """
    n = len(x)
    if n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def human_alignment_report(
    human_labels: list[int],
    detector_labels: list[int],
    human_scores: list[float] | None = None,
    detector_scores: list[float] | None = None,
) -> dict:
    """
    Compute human-alignment metrics for automated detection vs. human gold labels.

    Args:
        human_labels: Binary hallucination labels from human annotators.
        detector_labels: Binary hallucination labels from automated detector.
        human_scores: Optional continuous scores from human annotators.
        detector_scores: Optional continuous scores from automated detector.

    Returns:
        Dict with accuracy, f1, kappa, and optionally pearson_r.
    """
    report = {
        "accuracy": accuracy(human_labels, detector_labels),
        "precision": precision(human_labels, detector_labels),
        "recall": recall(human_labels, detector_labels),
        "f1": f1_score(human_labels, detector_labels),
        "cohens_kappa": cohens_kappa(human_labels, detector_labels),
    }

    if human_scores is not None and detector_scores is not None:
        report["pearson_r"] = pearson_correlation(human_scores, detector_scores)

    return report
