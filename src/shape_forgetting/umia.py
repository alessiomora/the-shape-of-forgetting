from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0))


def best_threshold_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = np.r_[-np.inf, (sorted_scores[:-1] + sorted_scores[1:]) / 2.0, np.inf]
    best_acc = -1.0
    best_threshold = 0.0
    for threshold in candidates:
        pred = scores >= threshold
        acc = accuracy_score(labels, pred)
        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)
    return best_acc, best_threshold


def compute_umia(
    forget_probs: np.ndarray,
    forget_labels: np.ndarray,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
) -> dict[str, float | str]:
    """DeepUnlearn-style membership diagnostic.

    Forget samples are treated as the positive/member class and validation
    samples as the non-member reference class.  We report the strongest of
    three simple output-only scores for auditability.
    """
    y_true = np.concatenate([np.ones(len(forget_labels), dtype=bool), np.zeros(len(val_labels), dtype=bool)])
    score_bank = {
        "confidence": np.concatenate([
            forget_probs[np.arange(len(forget_labels)), forget_labels],
            val_probs[np.arange(len(val_labels)), val_labels],
        ]),
        "negative_loss": -np.concatenate([cross_entropy(forget_probs, forget_labels), cross_entropy(val_probs, val_labels)]),
        "negative_entropy": -np.concatenate([entropy(forget_probs), entropy(val_probs)]),
    }
    rows = []
    for name, scores in score_bank.items():
        acc, threshold = best_threshold_accuracy(scores, y_true)
        try:
            auroc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auroc = 0.5
        rows.append((acc, auroc, name, threshold))
    acc, auroc, name, threshold = max(rows, key=lambda item: (item[0], item[1]))
    return {
        "u_mia_accuracy": float(acc),
        "u_mia_auroc": float(auroc),
        "u_mia_indiscernibility": float(1.0 - abs(acc - 0.5) / 0.5),
        "attack_score": name,
        "attack_threshold": float(threshold),
        "forget_samples": int(len(forget_labels)),
        "validation_samples": int(len(val_labels)),
    }
