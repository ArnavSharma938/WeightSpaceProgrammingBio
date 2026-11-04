from __future__ import annotations

import math

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (np.asarray(scores) >= 0.5).astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    safe = lambda numerator, denominator: float(numerator / denominator) if denominator else float("nan")
    sensitivity = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe(tp + tn, len(labels)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": safe(tp, tp + fp),
        "recall": sensitivity,
        "specificity": specificity,
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else float("nan"),
        "roc_auc": roc_auc(labels, scores),
        "mean_positive_score": float(np.mean(scores[labels == 1])) if (labels == 1).any() else float("nan"),
        "mean_negative_score": float(np.mean(scores[labels == 0])) if (labels == 0).any() else float("nan"),
    }
