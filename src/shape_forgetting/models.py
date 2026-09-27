from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig


class TimmClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(config: ExperimentConfig, pretrained: bool = True) -> nn.Module:
    if config.model in {"deit_tiny_patch16_224", "vit_small_patch16_224"}:
        return TimmClassifier(config.model, config.num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model in minimal repo: {config.model}")


def uses_pretrained_initialization(config: ExperimentConfig) -> bool:
    return config.model in {"deit_tiny_patch16_224", "vit_small_patch16_224"}
