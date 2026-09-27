from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import ExperimentConfig, with_training_overrides
from .data import build_eval_loader, build_train_loader, make_or_load_split
from .metrics import accuracy_from_logits, distribution_metrics, write_json
from .models import create_model
from .rsf import rsf_targets, softmax
from .seed import seed_everything
from .umia import compute_umia


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimizer_for(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def set_lr(config: ExperimentConfig, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    if config.scheduler != "cosine":
        return
    if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
        mult = float(epoch + 1) / float(config.warmup_epochs)
    else:
        denom = max(1, config.epochs - config.warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - config.warmup_epochs) / denom))
        mult = 0.5 * (1.0 + np.cos(np.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * mult


def checkpoint_payload(config: ExperimentConfig, model: nn.Module, role: str, seed: int, meta: dict | None = None) -> dict:
    return {
        "model": model.state_dict(),
        "config_name": config.name,
        "role": role,
        "seed": seed,
        "meta": meta or {},
    }


def load_model(config: ExperimentConfig, seed: int, role: str, device: torch.device | None = None) -> nn.Module:
    device = default_device() if device is None else device
    model = create_model(config).to(device)
    path = config.checkpoint_path(seed, role)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for role {role}: {path}")
    payload = torch.load(path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def collect_logits(
    config: ExperimentConfig,
    seed: int,
    role: str,
    split: str,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = default_device() if device is None else device
    split_indices = make_or_load_split(config, seed)
    if split == "test":
        loader = build_eval_loader(config, train=False, indices=split_indices.test)
    else:
        indices = getattr(split_indices, split)
        loader = build_eval_loader(config, train=True, indices=indices)
    model = load_model(config, seed, role, device=device)
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    for images, labels, source_indices in tqdm(loader, desc=f"collect {role}/{split}", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images).detach().cpu().numpy()
        logits_all.append(logits)
        labels_all.append(labels.numpy())
        indices_all.append(source_indices.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all), np.concatenate(indices_all)


def evaluate_role(
    config: ExperimentConfig,
    seed: int,
    role: str,
    reference_role: str = "retrained",
) -> dict:
    device = default_device()
    metrics: dict[str, float | int | str] = {"role": role, "seed": seed, "config": config.name}
    for split in ("forget", "retain", "val", "test"):
        logits, labels, _indices = collect_logits(config, seed, role, split, device=device)
        metrics[f"{split}_acc"] = accuracy_from_logits(logits, labels)
    ref_path = config.checkpoint_path(seed, reference_role)
    if role != reference_role and ref_path.exists():
        ref_logits, _labels, _indices = collect_logits(config, seed, reference_role, "forget", device=device)
        cand_logits, _labels2, _indices2 = collect_logits(config, seed, role, "forget", device=device)
        metrics.update(distribution_metrics(ref_logits, cand_logits))
        original_path = config.checkpoint_path(seed, "original")
        if original_path.exists():
            original_logits, _labels3, _indices3 = collect_logits(config, seed, "original", "forget", device=device)
            original_argmax = original_logits.argmax(axis=1)
            ref_change = ref_logits.argmax(axis=1) != original_argmax
            cand_change = cand_logits.argmax(axis=1) != original_argmax
            metrics["forget_change_rate"] = float(cand_change.mean())
            denom = max(1, int(cand_change.sum() + ref_change.sum()))
            tp = float(np.logical_and(cand_change, ref_change).sum())
            precision = tp / max(1.0, float(cand_change.sum()))
            recall = tp / max(1.0, float(ref_change.sum()))
            metrics["forget_change_precision"] = precision
            metrics["forget_change_recall"] = recall
            metrics["forget_change_f1"] = float(2.0 * precision * recall / max(1e-12, precision + recall))
            metrics["reference_change_rate"] = float(ref_change.mean())
            metrics["change_symmetric_count"] = int(denom)
    else:
        metrics["forget_kl_retrained_to_role"] = 0.0 if role == reference_role else None
        metrics["forget_total_variation"] = 0.0 if role == reference_role else None
        metrics["forget_retrained_argmax_agreement"] = 1.0 if role == reference_role else None
    write_json(config.metrics_path(seed, role), metrics)
    return metrics


def train_reference(config: ExperimentConfig, seed: int, role: str, training_seed: int | None = None) -> dict:
    if role not in {"original", "retrained"}:
        raise ValueError("Reference training role must be original or retrained.")
    training_seed = seed if training_seed is None else training_seed
    seed_everything(training_seed, deterministic=config.deterministic)
    device = default_device()
    split = make_or_load_split(config, seed)
    indices = split.train_all if role == "original" else split.retain
    loader = build_train_loader(config, indices, seed=training_seed)
    model = create_model(config).to(device)
    optimizer = optimizer_for(config, model)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")
    start = time.perf_counter()
    for epoch in range(config.epochs):
        set_lr(config, optimizer, epoch)
        model.train()
        running = 0.0
        for images, labels, _indices in tqdm(loader, desc=f"{role} epoch {epoch + 1}/{config.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.amp and device.type == "cuda"):
                loss = F.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu()) * images.size(0)
    runtime = time.perf_counter() - start
    path = config.checkpoint_path(seed, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(config, model, role, seed, {"runtime_seconds": runtime}), path)
    metrics = evaluate_role(config, seed, role)
    metrics["runtime_seconds"] = runtime
    write_json(config.metrics_path(seed, role), metrics)
    return metrics


class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, base_loader_dataset, targets_by_index: dict[int, np.ndarray]) -> None:
        self.base = base_loader_dataset
        self.targets_by_index = targets_by_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, _label, source_index = self.base[index]
        return image, torch.from_numpy(self.targets_by_index[int(source_index)]).float(), source_index


def train_rsf(
    config: ExperimentConfig,
    seed: int,
    rho: float,
    epochs: int,
    lr: float,
    wd: float,
    optimizer_name: str = "adamw",
    batch_size: int | None = None,
    training_seed: int = 12345,
    output_role: str | None = None,
    save_checkpoint: bool = True,
) -> dict:
    output_role = output_role or f"rsf_rho{rho:g}_ep{epochs}_lr{lr:g}_wd{wd:g}"
    seed_everything(training_seed, deterministic=config.deterministic)
    run_cfg = with_training_overrides(
        config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=wd,
        optimizer=optimizer_name,
    )
    device = default_device()
    original_logits, _labels, source_indices = collect_logits(config, seed, "original", "forget", device=device)
    targets, target_meta = rsf_targets(original_logits, rho=rho, tau_s=1.5)
    targets_by_index = {int(idx): targets[pos] for pos, idx in enumerate(source_indices)}
    split = make_or_load_split(config, seed)
    base_loader = build_train_loader(run_cfg, split.forget, seed=training_seed, batch_size=batch_size)
    target_dataset = TargetDataset(base_loader.dataset, targets_by_index)
    loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=run_cfg.batch_size if batch_size is None else batch_size,
        shuffle=True,
        num_workers=run_cfg.num_workers,
        pin_memory=True,
    )
    model = load_model(config, seed, "original", device=device)
    optimizer = optimizer_for(run_cfg, model)
    scaler = GradScaler(enabled=run_cfg.amp and device.type == "cuda")
    start = time.perf_counter()
    for epoch in range(run_cfg.epochs):
        set_lr(run_cfg, optimizer, epoch)
        model.train()
        for images, target_probs, _indices in tqdm(loader, desc=f"{output_role} epoch {epoch + 1}/{run_cfg.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            target_probs = target_probs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=run_cfg.amp and device.type == "cuda"):
                log_probs = F.log_softmax(model(images), dim=1)
                loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    runtime = time.perf_counter() - start
    if save_checkpoint:
        path = config.checkpoint_path(seed, output_role)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"runtime_seconds": runtime, "rho": rho, "epochs": epochs, "lr": lr, "wd": wd, **target_meta}
        torch.save(checkpoint_payload(config, model, output_role, seed, meta), path)
    else:
        temp_role = output_role
        path = config.checkpoint_path(seed, temp_role)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(config, model, temp_role, seed, {"temporary": True}), path)
    metrics = evaluate_role(config, seed, output_role)
    metrics.update(target_meta)
    metrics["rho"] = float(rho)
    metrics["rsf_epochs"] = int(epochs)
    metrics["rsf_lr"] = float(lr)
    metrics["rsf_wd"] = float(wd)
    metrics["runtime_seconds"] = runtime
    write_json(config.metrics_path(seed, output_role), metrics)
    if not save_checkpoint:
        try:
            config.checkpoint_path(seed, output_role).unlink()
        except FileNotFoundError:
            pass
    return metrics


def run_rho_sweep(
    config: ExperimentConfig,
    seed: int,
    rho_values: Iterable[float],
    epochs: int,
    lr: float,
    wd: float,
    optimizer_name: str = "adamw",
    no_save_checkpoints: bool = True,
    output_name: str = "rho_sweep.csv",
) -> Path:
    rows: list[dict] = []
    for rho in rho_values:
        role = f"rsf_sweep_rho{rho:.2f}".replace(".", "p")
        rows.append(
            train_rsf(
                config,
                seed,
                rho=float(rho),
                epochs=epochs,
                lr=lr,
                wd=wd,
                optimizer_name=optimizer_name,
                output_role=role,
                save_checkpoint=not no_save_checkpoints,
            )
        )
    path = config.seed_dir(seed) / output_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def run_umia(config: ExperimentConfig, seed: int, role: str) -> dict:
    device = default_device()
    forget_logits, forget_labels, _ = collect_logits(config, seed, role, "forget", device=device)
    val_logits, val_labels, _ = collect_logits(config, seed, role, "val", device=device)
    result = compute_umia(softmax(forget_logits), forget_labels, softmax(val_logits), val_labels)
    result["role"] = role
    result["seed"] = seed
    result["config"] = config.name
    write_json(config.umia_path(seed, role), result)
    return result
