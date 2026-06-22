from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from models.encoder import UnifiedHypergraphEncoder
from utils.common import ensure_dir
from utils.dataset_registry import get_dataset_spec
from utils.dhg_datasets import load_domain_graphs
from utils.hypergraph import build_domain_aliases, iter_graphs


class DownstreamTrainerBase:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config["training"].get("device", "cpu"))
        self.output_dir = Path(config["training"]["output_dir"])
        ensure_dir(self.output_dir / "results")
        self.pretrain_config: Dict | None = None

    def build_encoder(self) -> UnifiedHypergraphEncoder:
        encoder = UnifiedHypergraphEncoder(
            in_dim=int(self.config["model"]["input_dim"]),
            hidden_dim=int(self.config["model"]["hidden_dim"]),
            dropout=float(self.config["model"]["dropout"]),
            num_layers=int(self.config["model"]["num_layers"]),
            num_heads=int(self.config["model"]["num_heads"]),
            spectral_dim=int(self.config["model"]["spectral_dim"]),
        ).to(self.device)
        checkpoint_path = self.config["training"].get("pretrained_checkpoint")
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            self.pretrain_config = state.get("config")
            current_state = encoder.state_dict()
            compatible_state = {
                key: value
                for key, value in state["encoder"].items()
                if key in current_state and current_state[key].shape == value.shape
            }
            encoder.load_state_dict(compatible_state, strict=False)
        return encoder

    def resolve_heldout(self, heldout_domain: str) -> str:
        aliases = build_domain_aliases()
        return aliases.get(heldout_domain, heldout_domain)

    def select_dataset_names(self, heldout_domain: str) -> List[str]:
        dataset_names = list(self.config["data"]["datasets"])
        explicit_domain_map = self.config["data"].get("domain_map", {})
        if heldout_domain in dataset_names:
            return [heldout_domain]
        selected = []
        for dataset_name in dataset_names:
            dataset_domain = explicit_domain_map.get(dataset_name, get_dataset_spec(dataset_name).domain)
            if dataset_domain == heldout_domain:
                selected.append(dataset_name)
        if not selected:
            available = sorted(set(explicit_domain_map.get(name, get_dataset_spec(name).domain) for name in dataset_names))
            raise ValueError(f"Unknown held-out target '{heldout_domain}'. Available domains: {', '.join(available)}")
        return selected

    def load_target_graphs(self, dataset_names: List[str], require_node_splits: bool = False) -> List:
        local_config = {
            **self.config,
            "data": {
                **self.config["data"],
                "datasets": dataset_names,
            },
        }
        return iter_graphs(
            load_domain_graphs(
                local_config,
                seed=int(self.config["training"]["seed"]),
                require_node_splits=require_node_splits,
            )
        )

    def attach_pretrain_domains(self, summary: Dict) -> Dict:
        if self.pretrain_config is not None:
            pretrain_domain_map = self.pretrain_config.get("data", {}).get("domain_map", {})
            summary["pretrain_domains"] = sorted(set(pretrain_domain_map.values()))
        return summary

