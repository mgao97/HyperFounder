from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.pretrain_trainer import PretrainTrainer
from utils.common import load_yaml, set_seed
from utils.eval import write_ablation_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-task ablation.")
    parser.add_argument("--config", required=True, help="Path to the pretraining config.")
    parser.add_argument(
        "--drop_task",
        required=True,
        choices=["struct", "node", "edge", "motif", "community", "global", "cross"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["training"]["seed"]))
    trainer = PretrainTrainer(config, drop_tasks={args.drop_task})
    summary = trainer.train()
    rows = [
        {"task": args.drop_task, "checkpoint_path": summary["checkpoint_path"], "loss_history_path": summary["loss_history_path"]},
    ]
    write_ablation_csv("outputs/results/ablation.csv", rows)
    print(rows[0])


if __name__ == "__main__":
    main()
