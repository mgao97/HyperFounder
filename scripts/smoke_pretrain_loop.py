from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoder import UnifiedHypergraphEncoder
from models.heads import TaskHeads
from models.pretext_tasks import compute_pretraining_losses
from utils.common import set_seed
from utils.hypergraph import SimpleHypergraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal smoke test for pretraining forward/loss loop.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    num_nodes = 12
    hyperedges = [[0, 1, 2], [2, 3, 4, 5], [6, 7], [7, 8, 9, 10, 11]]
    input_dim = 16
    hidden_dim = 32

    x = torch.randn(num_nodes, input_dim, dtype=torch.float32)
    hg = SimpleHypergraph(
        num_nodes=num_nodes,
        hyperedges=hyperedges,
        x=x,
        name="toy_hg",
        domain="toy",
        dataset_name="toy",
        node_labels=torch.zeros(num_nodes, dtype=torch.long),
        edge_labels=None,
        graph_label=None,
        node_train_mask=None,
        node_val_mask=None,
        node_test_mask=None,
        metadata={"domain_id": 0},
    )

    encoder = UnifiedHypergraphEncoder(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=0.1,
        num_layers=2,
        num_heads=4,
        structure_pe_dim=8,
        num_domains=1,
        domain_names=["toy"],
        topk=4,
        pooled_nodes=8,
        pooled_edges=4,
    ).to(device)

    heads = TaskHeads(
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        num_domains=1,
    ).to(device)

    config = {
        "model": {"input_dim": input_dim, "hidden_dim": hidden_dim},
        "training": {
            "motif_budget": 0,
            "feature_mask_rate": 0.15,
            "edge_dropout_rate": 0.2,
            "contrastive_strategy": "node_dropping",
            "loss_weights": {
                "masked_node": 1.0,
                "hyperedge_recon": 1.0,
                "contrastive": 1.0,
                "size_pred": 1.0,
                "domain_align": 0.0,
            },
        },
    }

    losses = compute_pretraining_losses(
        encoder=encoder,
        heads=heads,
        hg=hg,
        task_cache={},
        config=config,
        device=device,
        epoch=1,
        drop_tasks=set(),
    )

    printable = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
    print(printable)


if __name__ == "__main__":
    main()

