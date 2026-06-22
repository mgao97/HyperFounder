from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.pretrain_trainer import PretrainTrainer
from utils.common import load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hypergraph pretraining.")
    parser.add_argument("--config", required=True, help="Path to the pretraining config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["training"]["seed"]))
    trainer = PretrainTrainer(config)
    summary = trainer.train()
    save_json("outputs/results/pretrain_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
