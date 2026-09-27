from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def parse_rho_grid(spec: str) -> list[float]:
    if ":" in spec:
        start_s, stop_s, step_s = spec.split(":")
        start, stop, step = float(start_s), float(stop_s), float(step_s)
        values: list[float] = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(current, 10))
            current += step
        return values
    return [float(part) for part in spec.split(",") if part.strip()]


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a TOML config.")
    parser.add_argument("--seed", type=int, default=0, help="Split seed.")
    parser.add_argument("--data-dir", default=None, help="Override dataset root.")
    parser.add_argument("--run-dir", default=None, help="Override output root.")


def print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shape-forgetting")
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("make-split", help="Create or load a deterministic split.")
    add_common(p_split)

    p_train = sub.add_parser("train", help="Train original or retrained reference model.")
    add_common(p_train)
    p_train.add_argument("--role", choices=["original", "retrained"], required=True)
    p_train.add_argument("--training-seed", type=int, default=None)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint role.")
    add_common(p_eval)
    p_eval.add_argument("--role", required=True)
    p_eval.add_argument("--reference-role", default="retrained")

    p_rsf = sub.add_parser("rsf", help="Run RSF forget-only KD unlearning.")
    add_common(p_rsf)
    p_rsf.add_argument("--rho", type=float, required=True)
    p_rsf.add_argument("--epochs", type=int, required=True)
    p_rsf.add_argument("--lr", type=float, required=True)
    p_rsf.add_argument("--wd", type=float, default=0.0)
    p_rsf.add_argument("--optimizer", default="adamw")
    p_rsf.add_argument("--training-seed", type=int, default=12345)
    p_rsf.add_argument("--batch-size", type=int, default=None)
    p_rsf.add_argument("--output-role", default=None)
    p_rsf.add_argument("--no-save-checkpoint", action="store_true")

    p_sweep = sub.add_parser("rho-sweep", help="Run an RSF rho sensitivity sweep.")
    add_common(p_sweep)
    p_sweep.add_argument("--rho-grid", default="0:1:0.05")
    p_sweep.add_argument("--epochs", type=int, required=True)
    p_sweep.add_argument("--lr", type=float, required=True)
    p_sweep.add_argument("--wd", type=float, default=0.0)
    p_sweep.add_argument("--optimizer", default="adamw")
    p_sweep.add_argument("--no-save-checkpoints", action="store_true")
    p_sweep.add_argument("--output-name", default="rho_sweep.csv")

    p_umia = sub.add_parser("u-mia", help="Run output-only U-MIA on forget vs validation data.")
    add_common(p_umia)
    p_umia.add_argument("--role", required=True)
    p_umia.add_argument("--reference-role", default="retrained", help="Accepted for script compatibility.")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, data_dir=args.data_dir, run_dir=args.run_dir)

    if args.command == "make-split":
        from .data import make_or_load_split

        split = make_or_load_split(cfg, args.seed)
        print_json(
            {
                "config": cfg.name,
                "seed": args.seed,
                "split_path": str(cfg.split_path(args.seed)),
                "forget": int(len(split.forget)),
                "retain": int(len(split.retain)),
                "val": int(len(split.val)),
                "test": int(len(split.test)),
            }
        )
    elif args.command == "train":
        from .train import train_reference

        print_json(train_reference(cfg, args.seed, args.role, training_seed=args.training_seed))
    elif args.command == "eval":
        from .train import evaluate_role

        print_json(evaluate_role(cfg, args.seed, args.role, reference_role=args.reference_role))
    elif args.command == "rsf":
        from .train import train_rsf

        print_json(
            train_rsf(
                cfg,
                args.seed,
                rho=args.rho,
                epochs=args.epochs,
                lr=args.lr,
                wd=args.wd,
                optimizer_name=args.optimizer,
                batch_size=args.batch_size,
                training_seed=args.training_seed,
                output_role=args.output_role,
                save_checkpoint=not args.no_save_checkpoint,
            )
        )
    elif args.command == "rho-sweep":
        from .train import run_rho_sweep

        path = run_rho_sweep(
            cfg,
            args.seed,
            parse_rho_grid(args.rho_grid),
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            optimizer_name=args.optimizer,
            no_save_checkpoints=args.no_save_checkpoints,
            output_name=args.output_name,
        )
        print_json({"rho_sweep": str(path)})
    elif args.command == "u-mia":
        from .train import run_umia

        print_json(run_umia(cfg, args.seed, args.role))
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
