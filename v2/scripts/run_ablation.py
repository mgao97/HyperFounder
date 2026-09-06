"""W3 (CCA × 4 rows ablation runner) + W4 (HCA × 3 rows ablation runner).

One ablation per CLI `--variant`.  Reuses pretrain_trainer_v2 as backend: we
just override the relevant YAML/CLI flags the trainer already respects.

W3 — Challenge 1 (Cardinality robustness) ablation table × 4 rows:
  w3_full      : 完整 CCA (card_mlp + FiLM + τ)
  w3_no_card   : 去掉基数编码 (card_mlp → constant 0 vector, FiLM γ=1 β=0 via scale=1)
  w3_no_film   : 保留基数编码 c_e，但 FiLM 恒等 (γ=1, β=0)
  w3_no_tau    : 保留 c_e + FiLM，但 τ=ε constant (remove tau_head)

W4 — Challenge 2 (Overlap context) ablation table × 3 rows:
  w4_full      : 完整 HCA (q/k/v attn + bias_mlp(ô))
  w4_no_bias   : HCA 去掉 overlap-conditioned bias_mlp (bias = 0)
  w4_no_hca    : 完全跳过 HCA (edge→node 只做 mean 回传)

Plus W5 = HOR ablation: use_hor=true/false 对比 (W5 已经 trainer 支持)。

CLI:
  python v2/scripts/run_ablation.py --variant w3_full --device cuda:0
  python v2/scripts/run_ablation.py --variant w4_no_hca --device cuda:1

Output per run: outputs_v2/ablations/<variant>/epochs.csv + checkpoints/*.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

W3_VARS = {"w3_full", "w3_no_card", "w3_no_film", "w3_no_tau"}
W4_VARS = {"w4_full", "w4_no_bias", "w4_no_hca"}
W5_VARS = {"w5_with_hor", "w5_without_hor"}
ALL_VARS = W3_VARS | W4_VARS | W5_VARS


def _load_yaml(p: Path) -> dict:
    import yaml
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _override_for_variant(variant: str, cfg: dict) -> dict:
    """Apply variant-level overrides to a COPY of the config dict.

    The overrides are intentionally additive / flag-style: the downstream
    encoder/trainer code must read `training.ablation_*` toggles.  We patch
    `model`/`training` keys so the existing trainer's encoder constructor
    receives them.
    """
    c = deepcopy(cfg)
    c["training"]["ablation_variant"] = variant
    if variant in W3_VARS:
        c["training"]["ablate_cca_card"] = variant == "w3_no_card"
        c["training"]["ablate_cca_film"] = variant == "w3_no_film"
        c["training"]["ablate_cca_tau"] = variant == "w3_no_tau"
    if variant in W4_VARS:
        c["training"]["ablate_hca_bias"] = variant == "w4_no_bias"
        c["training"]["ablate_hca_full"] = variant == "w4_no_hca"
    if variant in W5_VARS:
        # W5 行：切换 HOR §5 模块 on/off
        c["model"]["use_hor"] = (variant == "w5_with_hor")
    c["training"]["output_dir"] = f"outputs_v2/ablations/{variant}"
    # shorter for ablation prototyping — default 60 epochs / 32 steps can be
    # overridden by CLI below; we keep defaults identical to pretrain_v2.yaml.
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(ALL_VARS))
    ap.add_argument("--config", type=str, default="v2/configs/pretrain_v2.yaml")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--override_epochs", type=int, default=None)
    ap.add_argument("--override_steps_per_epoch", type=int, default=None)
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = _load_yaml(cfg_path)
    cfg = _override_for_variant(args.variant, cfg)
    if args.override_epochs is not None:
        cfg["training"]["epochs"] = args.override_epochs
    if args.override_steps_per_epoch is not None:
        cfg["training"]["steps_per_epoch"] = args.override_steps_per_epoch
    # Force a single device
    cfg["training"]["device"] = args.device

    out_root = ROOT / cfg["training"]["output_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ablation_config_effective.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True, default=str) + "\n"
    )

    # Delegate to pretrain_trainer_v2 (it will pickle effective config itself too)
    from v2.trainers.pretrain_trainer_v2 import V2PretrainTrainer

    print(f"[ablation {args.variant}] device={args.device}  epochs={cfg['training']['epochs']}  "
          f"steps/ep={cfg['training']['steps_per_epoch']}  out_dir={cfg['training']['output_dir']}")
    trainer = V2PretrainTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
