from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str
    model: str
    num_classes: int
    forget_fraction: float
    forget_strategy: str
    val_fraction: float
    data_dir: Path
    run_dir: Path
    epochs: int
    batch_size: int
    num_workers: int
    amp: bool
    deterministic: bool
    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float
    warmup_epochs: int
    image_size: int
    normalization: str
    scheduler: str
    extraction_batch_size: int

    @property
    def experiment_dir(self) -> Path:
        return self.run_dir / self.name

    def seed_dir(self, seed: int) -> Path:
        return self.experiment_dir / f"seed_{seed}"

    def checkpoint_path(self, seed: int, role: str) -> Path:
        return self.seed_dir(seed) / "checkpoints" / f"{role}.pt"

    def metrics_path(self, seed: int, role: str) -> Path:
        return self.seed_dir(seed) / "metrics" / f"{role}.json"

    def umia_path(self, seed: int, role: str) -> Path:
        return self.seed_dir(seed) / "metrics" / f"{role}_umia.json"

    def split_path(self, seed: int) -> Path:
        return self.seed_dir(seed) / "splits.npz"


def load_config(path: str | Path, data_dir: str | None = None, run_dir: str | None = None) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    exp = raw["experiment"]
    paths = raw.get("paths", {})
    train = raw["training"]
    scheduler = raw.get("scheduler", {})
    extraction = raw.get("extraction", {})
    cfg = ExperimentConfig(
        name=str(exp["name"]),
        dataset=str(exp["dataset"]),
        model=str(exp["model"]),
        num_classes=int(exp["num_classes"]),
        forget_fraction=float(exp["forget_fraction"]),
        forget_strategy=str(exp.get("forget_strategy", "deepunlearn_random")),
        val_fraction=float(exp.get("val_fraction", 0.15)),
        data_dir=Path(data_dir or paths.get("data_dir", "./data")),
        run_dir=Path(run_dir or paths.get("run_dir", "./runs")),
        epochs=int(train["epochs"]),
        batch_size=int(train["batch_size"]),
        num_workers=int(train.get("num_workers", 8)),
        amp=bool(train.get("amp", True)),
        deterministic=bool(train.get("deterministic", True)),
        optimizer=str(train["optimizer"]),
        learning_rate=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
        momentum=float(train.get("momentum", 0.9)),
        warmup_epochs=int(train.get("warmup_epochs", 0)),
        image_size=int(train["image_size"]),
        normalization=str(train["normalization"]),
        scheduler=str(scheduler.get("name", "cosine")),
        extraction_batch_size=int(extraction.get("batch_size", train["batch_size"])),
    )
    return cfg


def with_training_overrides(
    cfg: ExperimentConfig,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    optimizer: str | None = None,
) -> ExperimentConfig:
    return replace(
        cfg,
        epochs=cfg.epochs if epochs is None else epochs,
        batch_size=cfg.batch_size if batch_size is None else batch_size,
        learning_rate=cfg.learning_rate if learning_rate is None else learning_rate,
        weight_decay=cfg.weight_decay if weight_decay is None else weight_decay,
        optimizer=cfg.optimizer if optimizer is None else optimizer,
    )
