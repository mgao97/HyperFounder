"""Linear-probe verification for neg_sam_v2 pretraining.

The cleanest "did pretraining actually help?" test:
1. Take the encoder from the pretrained checkpoint.
2. Freeze it. Train ONLY a linear classifier on each downstream dataset.
3. Compare against the same encoder with random init (still frozen) + linear classifier.

This isolates the encoder's representation quality from optimizer / fine-tuning noise.
If pretrained_acc >> random_acc, the encoder really learned useful features.

Usage:
    python scripts/linear_probe_neg_sam.py --device cuda
    python scripts/linear_probe_neg_sam.py --device cuda \
        --pretrained outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt \
        --datasets cora_cc cooking_200 \
        --seeds 7 13 42 \
        --epochs 100 --patience 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoder import UnifiedHypergraphEncoder
from trainers.downstream_base import DownstreamTrainerBase
from utils.common import load_yaml, save_json, set_seed
from utils.metrics import multiclass_accuracy, multiclass_macro_f1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear-probe test for pretrained encoder.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pretrained",
        default="outputs_neg_sam_smoke/checkpoints/pretrain_best_neg_sam.pt",
        help="Path to the pretrained encoder checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cora_cc", "cooking_200"],
        help="Datasets to linear-probe on.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--epochs", type=int, default=50, help="Max linear-probe epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Linear classifier LR.")
    parser.add_argument(
        "--config",
        default="configs/pretrain_neg_sam_smoke.yaml",
        help="Downstream config for hidden_dim/num_layers/etc.",
    )
    parser.add_argument("--output", default="outputs/results/linear_probe_neg_sam.json")
    return parser.parse_args()


def _build_encoder(config: dict, device: torch.device) -> UnifiedHypergraphEncoder:
    """Build an encoder with the right shape (no checkpoint loaded yet)."""
    enc = UnifiedHypergraphEncoder(
        in_dim=int(config["model"]["input_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=0.0,  # we want a clean linear probe; disable dropout inside encoder
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        structure_pe_dim=int(config["model"].get("structure_pe_dim", 0)),
        num_domains=1,
        domain_names=["probe"],
        max_k=int(config["model"].get("max_k", 512)),
    ).to(device)
    return enc


def _load_encoder_weights(encoder: UnifiedHypergraphEncoder, ckpt_path: str, device: torch.device) -> int:
    state = torch.load(ckpt_path, map_location=device)
    current = encoder.state_dict()
    compatible = {k: v for k, v in state["encoder"].items() if k in current and current[k].shape == v.shape}
    missing, unexpected = encoder.load_state_dict(compatible, strict=False)
    n_loaded = len(compatible)
    print(f"  loaded {n_loaded} encoder tensors from {ckpt_path}")
    return n_loaded


def _linear_probe_one(
    encoder: UnifiedHypergraphEncoder,
    graph,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
) -> dict:
    set_seed(seed)
    encoder.eval()  # ALWAYS eval — encoder is frozen
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Encode once (encoder is frozen, no need to redo per epoch).
    with torch.no_grad():
        node_emb, _, _, _ = encoder(
            graph,
            graph.x.to(device),
            motif_budget=0,
            motifs=[],
            motif_seed=0,
        )
    num_classes = int(graph.metadata["num_node_classes"])
    classifier = torch.nn.Linear(node_emb.size(-1), num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=5e-4)

    train_mask = graph.node_train_mask.to(device)
    val_mask = graph.node_val_mask.to(device)
    test_mask = graph.node_test_mask.to(device)
    labels = graph.node_labels.to(device)

    best_val, best_epoch, bad = -1.0, -1, 0
    best_state = None
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(node_emb)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(node_emb)[val_mask]
            val_acc = multiclass_accuracy(val_logits, labels[val_mask])
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(node_emb)[test_mask]
        test_labels = labels[test_mask]
        test_acc = multiclass_accuracy(test_logits, test_labels)
        test_f1 = multiclass_macro_f1(test_logits, test_labels, num_classes=num_classes)

    return {
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "best_val_accuracy": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
    }


def _load_graphs(args) -> list:
    """Load target graphs via the downstream base helper."""
    cfg = {
        "model": {
            "input_dim": 128,
            "hidden_dim": 128,
            "dropout": 0.0,
            "num_layers": 4,
            "num_heads": 8,
            "structure_pe_dim": 8,
        },
        "training": {"seed": 7, "device": args.device},
        "data": {
            "datasets": args.datasets,
            "cache_dir": "data/cache",
            "splits": {"train_ratio": 0.6, "val_ratio": 0.2},
        },
    }
    base = DownstreamTrainerBase.__new__(DownstreamTrainerBase)
    base.config = cfg
    base.device = torch.device(args.device)
    base.pretrain_config = None
    graphs = base.load_target_graphs(args.datasets, require_node_splits=True)
    return graphs


def main() -> None:
    args = _parse_args()
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[LinearProbe] WARN: requested device={requested_device} but CUDA is unavailable; falling back to cpu"
        )
        requested_device = "cpu"
    device = torch.device(requested_device)
    cfg = load_yaml(args.config)

    print(f"[LinearProbe] device={device}, datasets={args.datasets}, seeds={args.seeds}")
    print(f"[LinearProbe] pretrained_checkpoint={args.pretrained}")

    graphs = _load_graphs(args)

    summary = {
        "config": args.config,
        "pretrained_checkpoint": str(args.pretrained),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "results": {},
    }

    print()
    print("=" * 72)
    print(f"{'dataset':<15} {'variant':<12} {'test_acc':>10} {'test_f1':>10} {'best_val':>10} {'epochs':>8}")
    print("=" * 72)

    for graph in graphs:
        ds_name = graph.dataset_name
        ds_results = summary["results"][ds_name] = {"pretrained": [], "random": []}

        for variant, ckpt_path in [("pretrained", args.pretrained), ("random", None)]:
            enc = _build_encoder(cfg, device)
            if ckpt_path is not None and Path(ckpt_path).exists():
                n_loaded = _load_encoder_weights(enc, ckpt_path, device)
                n_total = sum(1 for _ in enc.state_dict().keys())
                if n_loaded < n_total * 0.5:
                    print(f"  [WARN] only {n_loaded}/{n_total} encoder tensors matched (likely hidden_dim / num_layers mismatch). Pass --config <matching_pretrain_yaml> to load full weights.")
            else:
                print(f"  [warn] {variant} checkpoint missing, using random init")

            for seed in args.seeds:
                t0 = time.perf_counter()
                res = _linear_probe_one(
                    encoder=enc,
                    graph=graph,
                    device=device,
                    seed=seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    lr=args.lr,
                )
                res["wall_sec"] = time.perf_counter() - t0
                ds_results[variant].append(res)
                print(
                    f"{ds_name:<15} {variant:<12} {res['test_accuracy']:>10.4f}"
                    f" {res['test_macro_f1']:>10.4f} {res['best_val_accuracy']:>10.4f}"
                    f" {res['epochs_run']:>8d}"
                )

    # Aggregate.
    print()
    print("=" * 72)
    print("Summary (mean over seeds):")
    print("=" * 72)
    overall = {"pretrained": [], "random": []}
    for ds_name, res in summary["results"].items():
        for variant in ("pretrained", "random"):
            accs = [r["test_accuracy"] for r in res[variant]]
            f1s = [r["test_macro_f1"] for r in res[variant]]
            mean_acc = sum(accs) / max(len(accs), 1)
            mean_f1 = sum(f1s) / max(len(f1s), 1)
            overall[variant].extend(accs)
            print(f"  {ds_name:<15} {variant:<12} acc={mean_acc:.4f}  f1={mean_f1:.4f}")
        delta = sum(r["test_accuracy"] for r in res["pretrained"]) / len(res["pretrained"]) - sum(
            r["test_accuracy"] for r in res["random"]
        ) / max(len(res["random"]), 1)
        print(f"  {ds_name:<15} {'Δ(pre-rand)':<12} {delta:+.4f}")
        if delta > 0.05:
            print(f"  {ds_name:<15} {'verdict':<12} ✓ STRONG signal (Δ={delta:+.4f})")
        elif delta > 0.01:
            print(f"  {ds_name:<15} {'verdict':<12} ~ modest signal (Δ={delta:+.4f})")
        else:
            print(f"  {ds_name:<15} {'verdict':<12} ✗ no signal (Δ={delta:+.4f})")

    if overall["pretrained"] and overall["random"]:
        mean_pre = sum(overall["pretrained"]) / len(overall["pretrained"])
        mean_rand = sum(overall["random"]) / len(overall["random"])
        print()
        print(f"Overall pretrained mean accuracy: {mean_pre:.4f}")
        print(f"Overall random     mean accuracy: {mean_rand:.4f}")
        print(f"Overall Δ (pre - rand)         : {mean_pre - mean_rand:+.4f}")

    summary["overall_pretrained_mean"] = (
        sum(overall["pretrained"]) / len(overall["pretrained"]) if overall["pretrained"] else None
    )
    summary["overall_random_mean"] = (
        sum(overall["random"]) / len(overall["random"]) if overall["random"] else None
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, summary)
    print(f"\n[LinearProbe] results written to {args.output}")


if __name__ == "__main__":
    main()
