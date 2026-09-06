"""v2 pretraining entry point (domain-agnostic).

Invoke:
  cd /home/user/GSK/mgao/HyperFounder
  PYTHONNOUSERSITE=1 /home/user/.conda/envs/grag/bin/python \
      v2/scripts/run_pretrain_v2.py --config v2/configs/pretrain_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.trainers.pretrain_trainer_v2 import V2PretrainTrainer
from v2.utils.common import load_yaml, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str,
                        help="Path to pretrain yaml (e.g. v2/configs/pretrain_v2.yaml)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config training.seed (for strict ablation repeatability)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override config training.output_dir (per-ablation out slot)")
    parser.add_argument("--use_hor", type=str, default=None,
                        choices=["true", "false"],
                        help="Override model.use_hor for W5 ablations")
    parser.add_argument("--ablate_cca_card", action="store_true")
    parser.add_argument("--ablate_cca_film", action="store_true")
    parser.add_argument("--ablate_cca_tau", action="store_true")
    parser.add_argument("--ablate_hca_bias", action="store_true")
    parser.add_argument("--ablate_hca_full", action="store_true")
    parser.add_argument("--override_model_hidden", type=int, default=None,
                        help="Override model.hidden_dim (useful for quick ablations)")
    parser.add_argument("--override_steps_per_epoch", type=int, default=None)
    parser.add_argument("--override_epochs", type=int, default=None)
    parser.add_argument("--override_max_nodes", type=int, default=None)
    parser.add_argument("--override_max_edges", type=int, default=None)
    parser.add_argument("--use_ib", action="store_true",
                        help="Enable T1 variational information bottleneck.")
    parser.add_argument("--ib_beta", type=float, default=None,
                        help="T1 KL coefficient beta.")
    parser.add_argument("--ib_latent_dim", type=int, default=None,
                        help="Optional latent dim for VIB bottlenecks; default = hidden_dim.")
    parser.add_argument("--uncertainty_mode", type=str, default=None,
                        choices=["kendall", "residual", "fixed"],
                        help="Task weighting mode: baseline kendall / T4 residual / fixed weights.")
    parser.add_argument("--checkpoint_fractions", type=str, default=None,
                        help="Comma-separated fractions like 0.25,0.5,0.75 for T5 checkpoint saves.")
    parser.add_argument("--no_save_csv", action="store_true",
                        help="Disable step/epoch CSV (for debug dry runs).")
    args = parser.parse_args()

    config = load_yaml(args.config)

    # Seed override (CLI > config) — strict ablation reproducibility
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.use_hor is not None:
        config["model"]["use_hor"] = (args.use_hor == "true")
    for key, attr in (("ablate_cca_card", "ablate_cca_card"),
                     ("ablate_cca_film", "ablate_cca_film"),
                     ("ablate_cca_tau",  "ablate_cca_tau"),
                     ("ablate_hca_bias", "ablate_hca_bias"),
                     ("ablate_hca_full", "ablate_hca_full")):
        if getattr(args, key, False):
            config["model"][attr] = True

    # CLI overrides (cheap ablation helper)
    if args.override_model_hidden is not None:
        config["model"]["hidden_dim"] = args.override_model_hidden
    if args.override_epochs is not None:
        config["training"]["epochs"] = args.override_epochs
    if args.override_steps_per_epoch is not None:
        config["training"]["steps_per_epoch"] = args.override_steps_per_epoch
    if args.override_max_nodes is not None:
        config["training"].setdefault("minibatch", {})["max_nodes"] = args.override_max_nodes
    if args.override_max_edges is not None:
        config["training"].setdefault("minibatch", {})["max_edges"] = args.override_max_edges
    if args.use_ib:
        config["training"]["use_ib"] = True
    if args.ib_beta is not None:
        config["training"]["ib_beta"] = float(args.ib_beta)
    if args.ib_latent_dim is not None:
        config["training"]["ib_latent_dim"] = int(args.ib_latent_dim)
    if args.uncertainty_mode is not None:
        config["training"]["uncertainty_mode"] = args.uncertainty_mode
        if args.uncertainty_mode == "fixed":
            config["training"]["use_kendall_uw"] = False
        elif args.uncertainty_mode == "kendall":
            config["training"]["use_kendall_uw"] = True
    if args.checkpoint_fractions is not None:
        vals = [float(v.strip()) for v in args.checkpoint_fractions.split(",") if v.strip()]
        config["training"]["checkpoint_fractions"] = vals

    seed = int(config["training"].get("seed", 7))
    set_seed(seed)

    # Echo effective config to logs dir
    log_dir = Path(config["training"]["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "pretrain_config_effective_v2.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Guard: force deterministic CUDA backend if CUDA
    if config["training"].get("device", "cuda") == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    trainer = V2PretrainTrainer(config)
    if args.no_save_csv:
        # Monkey-patch to skip CSV writes for dry-runs (keeps JSON summary)
        import types
        orig_train = trainer.train
        def patched_train(self=trainer):
            try:
                orig_train()
            finally:
                self.step_losses = []
                self.epoch_losses = []
        trainer.train = types.MethodType(lambda s: patched_train(), trainer)
    trainer.train()


if __name__ == "__main__":
    main()
