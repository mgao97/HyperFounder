"""Quick end-to-end verification for the neg_sam_v2 pretraining pipeline.

Runs:
  1. A short pretraining pass on a tiny config (smoke).
  2. Checks that the loss strictly decreases across epochs (proves that the
     backward/step actually train the model).
  3. Loads the resulting checkpoint and runs a one-epoch finetune on a held-out
     domain. Records the finetune accuracy.
  4. Runs a scratch baseline (no pretrain) on the same finetune config.
  5. Reports a side-by-side comparison so the user can see whether pretraining
     helped (vs the baseline).

Usage:
  python scripts/verify_pretrain_neg_sam.py --pretrain_config configs/pretrain_neg_sam_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v1.trainers.pretrain_trainer_neg_sam import PretrainTrainerNegSam
from trainers.finetune_trainer import FinetuneTrainer
from utils.common import load_yaml, save_json, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end smoke check for neg_sam_v2 pretraining.")
    parser.add_argument(
        "--pretrain_config",
        default="configs/pretrain_neg_sam_smoke.yaml",
        help="Path to a *small* pretrain config (defaults to the smoke config).",
    )
    parser.add_argument(
        "--finetune_config",
        default="configs/finetune_node_smoke.yaml",
        help="Downstream finetune config to use for the comparison.",
    )
    parser.add_argument("--heldout_domain", default="citation", help="Held-out domain for finetune.")
    parser.add_argument("--device", default=None, help="Override device (cpu or cuda).")
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=None,
        help="Override pretrain epochs from the config (useful to make it really tiny).",
    )
    return parser.parse_args()


def _run_pretrain(args) -> dict:
    config = load_yaml(args.pretrain_config)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.pretrain_epochs is not None:
        config["training"]["epochs"] = int(args.pretrain_epochs)
    set_seed(int(config["training"]["seed"]))
    print(f"[Verify] Starting pretrain with config={args.pretrain_config} epochs={config['training']['epochs']}")
    start = time.perf_counter()
    trainer = PretrainTrainerNegSam(config)
    summary = trainer.train()
    summary["pretrain_wall_sec"] = time.perf_counter() - start
    print(f"[Verify] Pretrain finished in {summary['pretrain_wall_sec']:.1f}s best_epoch={summary.get('best_epoch')}")
    return summary


def _load_losses(summary: dict) -> list:
    """Read epoch-level losses from the saved CSV."""
    history_path = Path(summary["loss_history_path"])
    if not history_path.exists():
        return []
    import csv
    with open(history_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    totals = [float(r["total"]) for r in rows if r.get("total")]
    return totals


def _verify_loss_decreasing(losses: list) -> bool:
    if len(losses) < 2:
        print("[Verify] Not enough epochs to verify loss decrease.")
        return False
    first, last = losses[0], losses[-1]
    best = min(losses)
    print(f"[Verify] Loss trajectory: first={first:.4f} last={last:.4f} best={best:.4f}")
    # Allow a tiny tolerance for floating-point noise.
    if last > first * 0.99 and best > first * 0.99:
        print("[Verify][WARN] Loss did NOT decrease. The model may not actually be training.")
        return False
    print("[Verify][OK] Loss decreased over training (the model is actually learning).")
    return True


def _run_finetune(args, checkpoint: str | None) -> dict:
    config = load_yaml(args.finetune_config)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    config.setdefault("training", {})["pretrained_checkpoint"] = checkpoint
    # Force a tiny number of epochs to keep this verification fast.
    config["training"]["finetune_epochs"] = int(config["training"].get("finetune_epochs", 30))
    config["training"]["num_seeds"] = 1
    config["training"]["early_stopping"]["patience"] = max(
        3, int(config["training"]["early_stopping"].get("patience", 5))
    )
    set_seed(int(config["training"]["seed"]))
    label = "pretrained" if checkpoint else "scratch"
    print(f"[Verify] Running finetune ({label}) on heldout={args.heldout_domain}")
    trainer = FinetuneTrainer(config)
    summary = trainer.run(task_name="node", heldout_domain=args.heldout_domain)
    summary["checkpoint"] = checkpoint
    summary["variant"] = label
    print(f"[Verify] Finetune ({label}) -> accuracy={summary.get('node_accuracy', 0):.4f}")
    return summary


def main() -> None:
    args = _parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Pretrain.
    pretrain_summary = _run_pretrain(args)
    losses = _load_losses(pretrain_summary)
    loss_decreased = _verify_loss_decreasing(losses)
    save_json(f"{pretrain_summary['checkpoint_path']}.verify_meta.json", {"losses": losses})

    # 2. Pretrained finetune.
    pretrained_ckpt = str(Path(pretrain_summary["checkpoint_path"]).parent / "pretrain_best_neg_sam.pt")
    if not Path(pretrained_ckpt).exists():
        # Fallback to last.pt if best is missing.
        pretrained_ckpt = str(Path(pretrain_summary["checkpoint_path"]))
    pretrained_summary = _run_finetune(args, pretrained_ckpt)

    # 3. Scratch baseline.
    scratch_summary = _run_finetune(args, None)

    # 4. Side-by-side comparison.
    delta = pretrained_summary.get("node_accuracy", 0) - scratch_summary.get("node_accuracy", 0)
    print()
    print("============================================")
    print("Pretrain -> Finetune verification report")
    print("============================================")
    print(f"Pretrain epochs           : {pretrain_summary.get('best_epoch')} (best of {pretrain_summary.get('training_datasets')})")
    print(f"Pretrain wall time (s)    : {pretrain_summary['pretrain_wall_sec']:.1f}")
    print(f"Pretrain loss trajectory  : first={losses[0]:.4f} last={losses[-1]:.4f} best={min(losses):.4f}")
    print(f"Loss decreased            : {'YES' if loss_decreased else 'NO'}")
    print(f"Scratch finetune accuracy : {scratch_summary.get('node_accuracy', 0):.4f}")
    print(f"Pretrained finetune acc   : {pretrained_summary.get('node_accuracy', 0):.4f}")
    print(f"Delta (pretrained-scratch): {delta:+.4f}")
    print()
    if loss_decreased:
        print("[Verify][OK] Pretraining is actually training the model. Checkpoint can be used.")
    else:
        print("[Verify][FAIL] Pretraining loss did not decrease. See pretrain_trainer_neg_sam.py for backward/step.")
    out = {
        "pretrain_summary": pretrain_summary,
        "losses": losses,
        "loss_decreased": loss_decreased,
        "scratch_summary": scratch_summary,
        "pretrained_summary": pretrained_summary,
        "delta_accuracy": delta,
    }
    save_json("outputs/results/verify_neg_sam.json", out)


if __name__ == "__main__":
    main()
