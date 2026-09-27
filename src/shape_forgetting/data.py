from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from .config import ExperimentConfig
from .seed import dataloader_generator


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitIndices:
    forget: np.ndarray
    retain: np.ndarray
    train_all: np.ndarray
    val: np.ndarray
    test: np.ndarray


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray | None = None) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, index: int):
        source_index = int(self.indices[index]) if self.indices is not None else int(index)
        image, label = self.dataset[source_index]
        return image, int(label), source_index


class TinyImageNetVal(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], transform=None) -> None:
        annotations = root / "val" / "val_annotations.txt"
        image_dir = root / "val" / "images"
        if not annotations.exists():
            raise FileNotFoundError(f"Missing Tiny-ImageNet annotations: {annotations}")
        samples: list[tuple[Path, int]] = []
        with annotations.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip().split("\t")
                if len(parts) >= 2:
                    samples.append((image_dir / parts[0], class_to_idx[parts[1]]))
        self.samples = samples
        self.targets = [label for _path, label in samples]
        self.classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = default_loader(str(path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class Food101WithTargets(Dataset):
    def __init__(self, root: Path, train: bool, transform=None) -> None:
        self.dataset = datasets.Food101(root=str(root), split="train" if train else "test", download=True)
        self.transform = transform
        self.targets = [int(label) for label in getattr(self.dataset, "_labels")]
        self.classes = list(getattr(self.dataset, "classes"))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class OxfordIIITPetWithTargets(Dataset):
    def __init__(self, root: Path, train: bool, transform=None) -> None:
        self.dataset = datasets.OxfordIIITPet(
            root=str(root),
            split="trainval" if train else "test",
            target_types="category",
            download=True,
        )
        self.transform = transform
        self.targets = [int(label) for label in getattr(self.dataset, "_labels")]
        self.classes = list(getattr(self.dataset, "classes", getattr(self.dataset, "_classes", [])))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class StanfordCarsWithTargets(Dataset):
    def __init__(self, root: Path, train: bool, transform=None) -> None:
        self.dataset = datasets.StanfordCars(
            root=str(root),
            split="train" if train else "test",
            transform=None,
            download=False,
        )
        self.transform = transform
        samples = getattr(self.dataset, "_samples", getattr(self.dataset, "samples", None))
        if samples is None:
            raise ValueError("Expected StanfordCars samples metadata.")
        self.targets = [int(sample[-1]) for sample in samples]
        self.classes = list(getattr(self.dataset, "classes", getattr(self.dataset, "_classes", [])))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def normalization_stats(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if name == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    if name == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    raise ValueError(f"Unknown normalization: {name}")


def build_transforms(config: ExperimentConfig, train: bool):
    mean, std = normalization_stats(config.normalization)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(config.image_size + 32),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_dataset(config: ExperimentConfig, train: bool, eval_transform: bool = False) -> Dataset:
    transform = build_transforms(config, train=train and not eval_transform)
    if config.dataset == "cifar100":
        return datasets.CIFAR100(root=str(config.data_dir), train=train, transform=transform, download=True)
    if config.dataset == "tinyimagenet":
        root = config.data_dir / "tiny-imagenet-200"
        train_root = root / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing Tiny-ImageNet train directory: {train_root}")
        train_dataset = datasets.ImageFolder(root=str(train_root), transform=transform if train else None)
        return train_dataset if train else TinyImageNetVal(root, train_dataset.class_to_idx, transform=transform)
    if config.dataset == "food101":
        return Food101WithTargets(config.data_dir, train=train, transform=transform)
    if config.dataset == "oxford_pets":
        return OxfordIIITPetWithTargets(config.data_dir, train=train, transform=transform)
    if config.dataset == "stanford_cars":
        return StanfordCarsWithTargets(config.data_dir, train=train, transform=transform)
    raise ValueError(f"Unknown dataset: {config.dataset}")


def labels_for(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose targets.")
    return np.asarray(targets, dtype=np.int64)


def make_or_load_split(config: ExperimentConfig, seed: int) -> SplitIndices:
    path = config.split_path(seed)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return SplitIndices(
                forget=data["forget"],
                retain=data["retain"],
                train_all=data["train_all"],
                val=data["val"],
                test=data["test"],
            )

    if config.forget_strategy != "deepunlearn_random":
        raise ValueError("Minimal repo supports only deepunlearn_random splits.")
    train = build_dataset(config, train=True, eval_transform=True)
    test = build_dataset(config, train=False, eval_transform=True)
    total = len(train)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total)
    val_count = int(round(total * config.val_fraction))
    val = np.sort(permutation[:val_count].astype(np.int64))
    train_pool = np.sort(permutation[val_count:].astype(np.int64))
    forget_count = int(round(train_pool.size * config.forget_fraction))
    forget_positions = rng.permutation(train_pool.size)[:forget_count]
    forget = np.sort(train_pool[forget_positions].astype(np.int64))
    retain = np.setdiff1d(train_pool, forget, assume_unique=True)
    split = SplitIndices(
        forget=forget,
        retain=retain.astype(np.int64),
        train_all=train_pool.astype(np.int64),
        val=val,
        test=np.arange(len(test), dtype=np.int64),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        forget=split.forget,
        retain=split.retain,
        train_all=split.train_all,
        val=split.val,
        test=split.test,
        forget_strategy=np.asarray([config.forget_strategy]),
        val_fraction=np.asarray([config.val_fraction], dtype=np.float32),
    )
    return split


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def build_train_loader(config: ExperimentConfig, indices: np.ndarray, seed: int, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        IndexedDataset(build_dataset(config, train=True), indices),
        batch_size=config.batch_size if batch_size is None else batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=dataloader_generator(seed),
        worker_init_fn=seed_worker,
    )


def build_eval_loader(config: ExperimentConfig, train: bool, indices: np.ndarray | None) -> DataLoader:
    return DataLoader(
        IndexedDataset(build_dataset(config, train=train, eval_transform=True), indices),
        batch_size=config.extraction_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
