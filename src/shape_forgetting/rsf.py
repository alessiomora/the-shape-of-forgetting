from __future__ import annotations

import numpy as np

EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), EPS, None)


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, EPS, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def hard_entropy_gate(probs: np.ndarray, rho: float) -> np.ndarray:
    if rho <= 0:
        return np.zeros(probs.shape[0], dtype=bool)
    if rho >= 1:
        return np.ones(probs.shape[0], dtype=bool)
    count = int(round(float(rho) * probs.shape[0]))
    gate = np.zeros(probs.shape[0], dtype=bool)
    if count > 0:
        gate[np.argsort(-entropy(probs), kind="mergesort")[:count]] = True
    return gate


def top1_tail_calibration(probs: np.ndarray, tau: float = 1.5) -> np.ndarray:
    out = np.array(probs, dtype=np.float64, copy=True)
    rows = np.arange(out.shape[0])
    source = out.argmax(axis=1)
    top_mass = out[rows, source]
    tail_mass = np.clip(1.0 - top_mass, 0.0, 1.0)
    top_s = np.clip(top_mass, EPS, 1.0) ** (1.0 / tau)
    tail_s = np.clip(tail_mass, EPS, 1.0) ** (1.0 / tau)
    new_top = top_s / np.clip(top_s + tail_s, EPS, None)
    new_tail = 1.0 - new_top
    out *= (new_tail / np.clip(tail_mass, EPS, None))[:, None]
    out[rows, source] = new_top
    out[tail_mass <= EPS] = probs[tail_mass <= EPS]
    return normalize(out)


def rsf_targets(original_logits: np.ndarray, rho: float, tau_s: float = 1.5) -> tuple[np.ndarray, dict[str, float]]:
    probs = softmax(original_logits)
    base = top1_tail_calibration(probs, tau=tau_s)
    rows = np.arange(base.shape[0])
    source = probs.argmax(axis=1)
    gate = hard_entropy_gate(probs, rho)

    receiver = np.array(base, copy=True)
    receiver[rows, source] = 0.0
    receiver = normalize(receiver)
    empty_tail = receiver.sum(axis=1) <= EPS
    if np.any(empty_tail):
        receiver[empty_tail] = 1.0 / max(1, receiver.shape[1] - 1)
        receiver[empty_tail, source[empty_tail]] = 0.0
        receiver[empty_tail] = normalize(receiver[empty_tail])

    masked = np.array(base, copy=True)
    masked[rows, source] = -np.inf
    runner_up = masked.argmax(axis=1)
    denom = 1.0 + receiver[rows, runner_up]
    moved = (base[rows, source] - base[rows, runner_up]) / np.clip(denom, EPS, None)
    moved = np.clip(moved + 1e-6, 0.0, base[rows, source])
    moved = np.where(gate, moved, 0.0)

    out = np.array(base, copy=True)
    out[rows, source] -= moved
    out += moved[:, None] * receiver
    out = normalize(out).astype(np.float32)
    hard_change = source != out.argmax(axis=1)
    return out, {
        "rho": float(rho),
        "tau_s": float(tau_s),
        "requested_gate_rate": float(rho),
        "realized_gate_rate": float(gate.mean()),
        "realized_change_rate": float(hard_change.mean()),
        "mean_moved_mass": float(moved.mean()),
    }
