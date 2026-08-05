from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.finetune_trainer import FinetuneTrainer
from trainers.graph_trainer import GraphFinetuneTrainer
from trainers.recommendation_trainer import RecommendationTrainer
from utils.common import load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-domain transfer.")
    parser.add_argument("--config", required=True, help="Path to the finetune config.")
    parser.add_argument("--heldout_domain", required=True, help="Held-out domain alias or full name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["training"]["seed"]))
    task_name = config["training"]["task_name"]
    print(
        "[HyperFounder][Transfer] Start:"
        f" task={task_name}"
        f" heldout_domain={args.heldout_domain}"
        f" config={args.config}"
        f" seed={config['training']['seed']}"
    )
    if task_name == "node":
        trainer = FinetuneTrainer(config)
    elif task_name in {"rec", "recommendation"}:
        trainer = RecommendationTrainer(config)
    elif task_name in {"graph", "graph_cls"}:
        trainer = GraphFinetuneTrainer(config)
    else:
        raise ValueError(f"Unsupported task_name '{task_name}'.")
    summary = trainer.run(task_name=task_name, heldout_domain=args.heldout_domain)
    # Record which checkpoint was used so compare_transfer_results.py can distinguish pretrained vs scratch
    summary["pretrained_checkpoint"] = config["training"].get("pretrained_checkpoint")
    save_json(f"outputs/results/transfer_{task_name}_{args.heldout_domain}.json", summary)
    print(
        "[HyperFounder][Transfer] Finished:"
        f" task={task_name}"
        f" heldout_domain={args.heldout_domain}"
        f" result_json=outputs/results/transfer_{task_name}_{args.heldout_domain}.json"
    )
    print(summary)


if __name__ == "__main__":
    main()
