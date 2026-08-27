from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.finetune_trainer_v1 import FinetuneTrainerV1
from models.encoder_v1 import UnifiedHypergraphEncoderV1
from utils.common import load_yaml, save_json, set_seed


class FinetuneTrainerV1Entry(FinetuneTrainerV1):
    """v1 entry: build the v1 encoder (OOM-safe attention) instead of the legacy one."""

    def build_encoder(self) -> UnifiedHypergraphEncoderV1:
        domain_names = sorted(set(self.config.get("data", {}).get("domain_map", {}).values()))
        encoder = UnifiedHypergraphEncoderV1(
            in_dim=int(self.config["model"]["input_dim"]),
            hidden_dim=int(self.config["model"]["hidden_dim"]),
            dropout=float(self.config["model"]["dropout"]),
            num_layers=int(self.config["model"]["num_layers"]),
            num_heads=int(self.config["model"]["num_heads"]),
            structure_pe_dim=int(self.config["model"].get("structure_pe_dim", self.config["model"].get("spectral_dim", 0))),
            num_domains=len(domain_names) if domain_names else 1,
            domain_names=domain_names,
            max_k=int(self.config["model"].get("max_k", 512)),
            use_domain_adapter=bool(self.config["model"].get("use_domain_adapter", True)),
            adapter_type=str(self.config["model"].get("adapter_type", "adapter")),
            adapter_dim=int(self.config["model"].get("adapter_dim", 32)),
            num_experts=int(self.config["model"].get("num_experts", 4)),
        ).to(self.device)
        checkpoint_path = self.config["training"].get("pretrained_checkpoint")
        if checkpoint_path and Path(checkpoint_path).exists():
            import torch
            state = torch.load(checkpoint_path, map_location=self.device)
            self.pretrain_config = state.get("config")
            self._register_domain_projectors(encoder, domain_names)
            current_state = encoder.state_dict()
            compatible_state = {
                key: value
                for key, value in state["encoder"].items()
                if key in current_state and current_state[key].shape == value.shape
            }
            encoder.load_state_dict(compatible_state, strict=False)
        return encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-domain transfer with v1 (OOM-safe + shared-branch) model.")
    parser.add_argument("--config", required=True, help="Path to the finetune config.")
    parser.add_argument("--heldout_domain", required=True, help="Held-out domain alias or full name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["training"]["seed"]))
    task_name = config["training"]["task_name"]
    print(
        "[HyperFounder][Transfer][v1] Start:"
        f" task={task_name}"
        f" heldout_domain={args.heldout_domain}"
        f" config={args.config}"
        f" seed={config['training']['seed']}"
    )
    if task_name != "node":
        raise ValueError(f"v1 entry currently supports task_name='node' only, got '{task_name}'.")
    trainer = FinetuneTrainerV1Entry(config)
    summary = trainer.run(task_name=task_name, heldout_domain=args.heldout_domain)
    summary["pretrained_checkpoint"] = config["training"].get("pretrained_checkpoint")
    out_dir = Path(config.get("outputs", {}).get("root", "outputs_neg_sam_v1")) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / f"transfer_{task_name}_{args.heldout_domain}.json", summary)
    print(
        "[HyperFounder][Transfer][v1] Finished:"
        f" task={task_name} heldout_domain={args.heldout_domain}"
        f" result_json={out_dir}/transfer_{task_name}_{args.heldout_domain}.json"
    )
    print(summary)


if __name__ == "__main__":
    main()
