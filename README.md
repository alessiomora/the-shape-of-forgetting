# The Shape of Forgetting: Minimal Reproducibility Code

This repository contains a minimal implementation for reproducing the Section 6
forget-only experiments of the paper. It intentionally includes only:

- deterministic DeepUnlearn-style random deletion splits;
- original and retrained reference training;
- Rank-Spread Flip (RSF) forget-only distillation;
- RSF change-budget (`rho`) sensitivity sweeps;
- U-MIA logging on forget vs validation samples.

It excludes the exploratory diagnostics, fixed-teacher tables, and other
baseline implementations from the research codebase.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package exposes one command:

```bash
shape-forgetting --help
```

## Data Layout

Use `--data-dir` to point to a dataset root. If omitted, the default is
`./data`.

- CIFAR-100, Food-101, and Oxford-IIIT Pets are downloaded automatically by
  `torchvision`.
- Tiny-ImageNet is expected at:
  `<data_dir>/tiny-imagenet-200`, with the official layout
  `train/<wnid>/images/*.JPEG`, `val/images/*.JPEG`, and
  `val/val_annotations.txt`.
- Stanford Cars uses the `torchvision.datasets.StanfordCars` layout if already
  present under `<data_dir>`. Download it manually if torchvision cannot fetch
  it in your environment.

Outputs are written under `--run-dir` (default `./runs`):

```text
runs/<config>/seed_0/splits.npz
runs/<config>/seed_0/checkpoints/original.pt
runs/<config>/seed_0/checkpoints/retrained.pt
runs/<config>/seed_0/checkpoints/rsf_*.pt
runs/<config>/seed_0/metrics/*.json
runs/<config>/seed_0/metrics/*_umia.json
runs/<config>/seed_0/rho_sweep.csv
```

## Supported Section 6 Settings

Configs live in `configs/`:

- `cifar100_deit_tiny_deepunlearn_random10.toml`
- `cifar100_deit_tiny_deepunlearn_random20.toml`
- `tinyimagenet_deit_tiny_deepunlearn_random10.toml`
- `tinyimagenet_deit_tiny_deepunlearn_random20.toml`
- `tinyimagenet_vit_small_deepunlearn_random10.toml`
- `food101_vit_small_deepunlearn_random10.toml`
- `food101_vit_small_deepunlearn_random20.toml`
- `oxford_pets_vit_small_deepunlearn_random10.toml`
- `oxford_pets_vit_small_deepunlearn_random20.toml`
- `stanford_cars_vit_small_deepunlearn_random10.toml`
- `stanford_cars_vit_small_deepunlearn_random20.toml`

The split protocol reserves 15% of the original training split as validation
data for U-MIA, then removes either 10% or 20% of the remaining training pool as
forget data. The retrained model is trained on the retain subset only.

## Minimal Example

```bash
CONFIG=configs/oxford_pets_vit_small_deepunlearn_random10.toml
DATA_DIR=/path/to/data
RUN_DIR=./runs

shape-forgetting make-split --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR"
shape-forgetting train --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role original
shape-forgetting train --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role retrained

shape-forgetting rsf \
  --config "$CONFIG" \
  --seed 0 \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --rho 0.10 \
  --epochs 5 \
  --lr 1e-4 \
  --wd 0

shape-forgetting eval --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role original
shape-forgetting eval --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role retrained
shape-forgetting eval --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role rsf_rho0.1_ep5_lr0.0001_wd0

shape-forgetting u-mia --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role original
shape-forgetting u-mia --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role retrained
shape-forgetting u-mia --config "$CONFIG" --seed 0 --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role rsf_rho0.1_ep5_lr0.0001_wd0
```

## RSF Rho Sweep

```bash
shape-forgetting rho-sweep \
  --config "$CONFIG" \
  --seed 0 \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --rho-grid 0:1:0.05 \
  --epochs 5 \
  --lr 1e-4 \
  --wd 0 \
  --no-save-checkpoints
```

The sweep writes a CSV with forget, retain, validation, and test accuracy, plus
forget-set KL/TV to the retrained model when the retrained checkpoint exists.

## RSF Presets Used in the Paper

These are the selected Section 6 RSF presets used as starting points for the
reported experiments and sensitivity sweeps:

| Setting | rho | Epochs | LR | WD |
|---|---:|---:|---:|---:|
| CIFAR-100 / DeiT-T / random-10 | 0.30 | 10 | 1e-4 | 0 |
| CIFAR-100 / DeiT-T / random-20 | 0.45 | 10 | 1e-4 | 0 |
| Tiny-ImageNet / DeiT-T / random-10 | 0.50 | 10 | 1e-4 | 0 |
| Tiny-ImageNet / DeiT-T / random-20 | 0.60 | 10 | 1e-4 | 0 |
| Tiny-ImageNet / ViT-S / random-10 | 0.35 | 3 | 1e-4 | 0 |
| Food-101 / ViT-S / random-10 | 0.30 | 3 | 1e-4 | 0 |
| Food-101 / ViT-S / random-20 | 0.45 | 3 | 1e-4 | 0 |
| Oxford Pets / ViT-S / random-10 | 0.10 | 5 | 1e-4 | 0 |
| Oxford Pets / ViT-S / random-20 | 0.10 | 10 | 3e-5 | 0 |
| Stanford Cars / ViT-S / random-10 | 0.35 | 10 | 1e-4 | 0 |
| Stanford Cars / ViT-S / random-20 | 0.40 | 10 | 1e-4 | 0 |

## U-MIA

The `u-mia` command implements the output-only U-MIA diagnostic used for the
paper tables. Forget samples are treated as the positive side of the attack and
validation samples as the non-member reference side. The implementation reports:

- `u_mia_accuracy`
- `u_mia_auroc`
- `u_mia_indiscernibility`
- attack score type and threshold metadata

The validation split is created by `make-split` and is never used for RSF target
construction.

## Reproducibility Notes

All commands accept `--seed`, `--data-dir`, and `--run-dir`. The code seeds
Python, NumPy, PyTorch, and dataloader workers. Exact values may still vary
slightly across CUDA/cuDNN versions and hardware.
