"""
Evaluation metrics for hallucination detection.

Metrics:
- Accuracy  — (TP + TN) / Total
- Precision — TP / (TP + FP)
- Recall    — TP / (TP + FN)
- F1        — 2 * P * R / (P + R)
- Hallucination Rate  — proportion flagged as hallucinatory by MLLM Judge

Hallucination detection is treated as a binary classification:
- Positive class = hallucination detected
- Ground truth label comes from:
  - Rule-based: comparison with dataset ground truth
  - Human annotation: gold labels
- For MLLM Judge vs. ground truth comparison, we treat MLLM Judge
  predictions as the detection result and ground truth as the label.
"""

import math

import numpy as np
from scipy.stats import bootstrap as _scipy_bootstrap
from scipy.stats import norm as _scipy_norm
from scipy.stats import pearsonr as _scipy_pearsonr
from sklearn.metrics import cohen_kappa_score as _sk_cohen_kappa
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
from statsmodels.stats.proportion import proportion_confint as _sm_prop_confint

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

    denom_p = tp + fp
    denom_r = tp + fn
    p = tp / denom_p if denom_p else 0.0
    r = tp / denom_r if denom_r else 0.0

    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "yes_ratio": y_pred.count(1) / len(y_pred) if y_pred else 0.0,
        "object_hallucination_rate": fp / total if total else 0.0,
        "n_object_hallucinated": fp,
        "total_samples": total,
        "confusion_matrix": cm,
    }


def hallucination_rate(gpt_flags: list[int | None]) -> float:
    """Proportion of hallucinated responses (score < 3 per MMHal-Bench)."""
    valid = [f for f in gpt_flags if f is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


_MLLM_JUDGE_TYPE_COUNTS_INIT = {"faithfulness": 0, "factuality": 0, "logical": 0, "none": 0, "error": 0}


def compute_mllm_judge_summary(details: list[dict], total: int) -> tuple[dict, list[int]]:
    """Aggregate MLLM Judge results into metrics. Used by all GPT-judge runners."""
    scores: list[float] = []
    hallucination_flags: list[int] = []
    type_counts = dict(_MLLM_JUDGE_TYPE_COUNTS_INIT)

    for detail in details:
        score = detail.get("score")
        has_h = detail.get("has_hallucination")
        htype = detail.get("hallucination_type", "error")
        if htype not in type_counts:
            htype = "error"
        if score is not None:
            scores.append(score)
        if has_h is not None:
            hallucination_flags.append(1 if has_h else 0)
        type_counts[htype] += 1

    metrics = {
        "total_samples": total,
        "completed_samples": len(details),
        "hallucination_rate": hallucination_rate(hallucination_flags) if hallucination_flags else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "n_hallucinated": sum(hallucination_flags),
        "n_valid": len(hallucination_flags),
        "type_counts": type_counts,
    }
    return metrics, hallucination_flags


# ==================== Human Alignment Metrics ====================

def cohens_kappa(y_true: list[int], y_pred: list[int]) -> float:
    """
    Cohen's Kappa — inter-rater agreement corrected for chance.
    Wraps sklearn.metrics.cohen_kappa_score; returns 1.0 when both raters
    assign every sample to the same single class (sklearn would return NaN
    in that degenerate case).

    Interpretation (Landis & Koch 1977):
        < 0.00  — worse than chance
        0.00–0.20 — slight
        0.21–0.40 — fair
        0.41–0.60 — moderate
        0.61–0.80 — substantial
        0.81–1.00 — almost perfect
    """
    if len(y_true) == 0:
        return 0.0
    if y_true == y_pred and len(set(y_true)) == 1:
        return 1.0
    return float(_sk_cohen_kappa(y_true, y_pred))


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """
    Pearson correlation coefficient. Wraps scipy.stats.pearsonr;
    returns 0.0 when either input has zero variance.
    """
    if len(x) == 0:
        return 0.0
    arr_x, arr_y = np.asarray(x), np.asarray(y)
    if arr_x.std() == 0 or arr_y.std() == 0:
        return 0.0
    return float(_scipy_pearsonr(arr_x, arr_y).statistic)


def human_alignment_report(
    human_labels: list[int],
    detector_labels: list[int],
    human_scores: list[float] | None = None,
    detector_scores: list[float] | None = None,
) -> dict:
    """Compute human-alignment metrics for automated detection vs. human gold labels."""
    cm = confusion_matrix(human_labels, detector_labels)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    total = tp + tn + fp + fn

    denom_p = tp + fp
    denom_r = tp + fn
    p = tp / denom_p if denom_p else 0.0
    r = tp / denom_r if denom_r else 0.0

    report = {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "cohens_kappa": cohens_kappa(human_labels, detector_labels),
    }
    if human_scores is not None and detector_scores is not None:
        report["pearson_r"] = pearson_correlation(human_scores, detector_scores)
    return report


# ==================== Confidence Intervals & Significance Tests ====================


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% CI for k/n. Wraps statsmodels.proportion_confint."""
    if n == 0:
        return (0.0, 0.0)
    lo, hi = _sm_prop_confint(k, n, alpha=alpha, method="wilson")
    return (float(lo), float(hi))


def paired_diff_ci(b: int, c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wald CI for the paired proportion difference Δ = (b - c) / n.

    b, c are the off-diagonal counts of the 2x2 paired contingency table
    (Fleiss, Levin & Paik 2003, §13.1):

        Var(Δ) = [(b + c) - (b - c)^2 / n] / n^2
    """
    if n == 0:
        return (0.0, 0.0)
    z = float(_scipy_norm.ppf(1 - alpha / 2))
    diff = (b - c) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(max(0.0, var))
    return (diff - z * se, diff + z * se)


def mcnemar_test(b: int, c: int, exact: bool | None = None) -> dict:
    """
    McNemar test for paired binary data. Wraps statsmodels.

    Uses exact binomial when b + c < 25; continuity-corrected chi² otherwise.
    Returns {chi2, p_value, method, b, c, n_discordant} with plain Python floats.
    """
    n_disc = b + c
    if exact is None:
        exact = n_disc < 25

    table = [[0, b], [c, 0]]
    result = _sm_mcnemar(table, exact=exact, correction=not exact)

    chi2 = float(result.statistic) if result.statistic is not None else 0.0
    p = float(result.pvalue)
    method = "exact_binomial" if exact else "chi2_cc"

    if exact and n_disc > 0:
        chi2 = ((b - c) ** 2) / n_disc
    return {"chi2": chi2, "p_value": p, "method": method,
            "b": b, "c": c, "n_discordant": n_disc}


def bootstrap_kappa_ci(
    y_true: list[int],
    y_pred: list[int],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for Cohen's kappa via scipy.stats.bootstrap.

    Note: under perfect agreement on a small sample, the percentile bootstrap
    is degenerate ([1, 1]); this reflects a limitation of the method, not the
    true precision of kappa at that sample size.
    """
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if (yt == yp).all():
        return (1.0, 1.0)

    def _stat(a: np.ndarray, b: np.ndarray) -> float:
        return cohens_kappa(a.tolist(), b.tolist())

    result = _scipy_bootstrap(
        (yt, yp),
        _stat,
        paired=True,
        n_resamples=n_resamples,
        random_state=seed,
        confidence_level=1 - alpha,
        method="percentile",
        vectorized=False,
    )
    return (float(result.confidence_interval.low),
            float(result.confidence_interval.high))


def paired_flip_counts(
    flags_a: dict[str, int],
    flags_b: dict[str, int],
) -> tuple[int, int, int, int, list[str]]:
    """
    Given two dicts pid → hallucination_flag (0/1), return (a, b, c, d, pids)
    contingency table over the intersection of pids:

        a = #pids where A=0 and B=0
        b = #pids where A=0 and B=1   (A→B introduced hallucination)
        c = #pids where A=1 and B=0   (A→B fixed hallucination)
        d = #pids where A=1 and B=1
        pids = sorted list of shared pids actually used
    """
    shared = sorted(set(flags_a) & set(flags_b))
    a = b = c = d = 0
    for pid in shared:
        fa, fb = flags_a[pid], flags_b[pid]
        if fa == 0 and fb == 0:
            a += 1
        elif fa == 0 and fb == 1:
            b += 1
        elif fa == 1 and fb == 0:
            c += 1
        else:
            d += 1
    return a, b, c, d, shared
