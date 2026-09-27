from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rsf import softmax


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum(axis=1).mean())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum(axis=1).mean())


def distribution_metrics(retrained_logits: np.ndarray, candidate_logits: np.ndarray) -> dict[str, float]:
    p_r = softmax(retrained_logits)
    p_u = softmax(candidate_logits)
    return {
        "forget_kl_retrained_to_role": kl_divergence(p_r, p_u),
        "forget_total_variation": total_variation(p_r, p_u),
        "forget_retrained_argmax_agreement": float((p_r.argmax(axis=1) == p_u.argmax(axis=1)).mean()),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
