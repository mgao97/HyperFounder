"""
Transductive baseline trainer on the 8 LightHGNN hypergraph datasets
(News20, CA-Cora, CC-Cora, CC-Citeseer, DBLP-Paper/Term/Conf, IMDB-AW).

Protocol mirrors iMoonLab/LightHGNN `trans_train.py` exactly:
  - split: 20 train / 100 val vertices per class, rest test (dhg split_by_num),
    re-drawn per seed after dhg.random.set_seed;
  - hypergraph models: dhg.models {HGNN, HGNNP, HNHN, UniGCN}
    with hid=32, use_bn=False; hypergraph = dhg.Hypergraph + self-hyperedges
    for isolated vertices (fix_iso_v);
  - MLP: plain 2-layer feature-only MLP;
  - optimizer Adam(lr=1e-2, weight_decay=5e-4), full batch, 200 epochs,
    best-val-accuracy checkpointing, test metrics reported at best epoch;
  - 5 seeds (default 0..4), report mean ± std.

Usage:
  python run_hyper_baseline.py --model hgnn --dataset ca_cora --device cuda:0
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dhg import Hypergraph
from dhg.random import set_seed
from dhg.utils import split_by_num
from dhg.models import HGNN, HGNNP, HNHN, UniGCN

ROOT = "/home/user/GSK/mgao/HyperFounder"
DATA_DIR = os.path.join(ROOT, "v3", "datasets")
RES_DIR = os.path.join(ROOT, "v3", "baselines", "results", "hyper")

DATASETS = [
    "news20", "ca_cora", "cc_cora", "cc_citeseer",
    "dblp4k_paper", "dblp4k_term", "dblp4k_conf", "imdb_aw",
]
HGNN_MODELS = ["hgnn", "hgnnp", "hnhn", "unigcn"]
ALL_MODELS = ["mlp"] + HGNN_MODELS


# ---------------------------------------------------------------------------
# data / structure
# ---------------------------------------------------------------------------
def load_dataset(name):
    d = os.path.join(DATA_DIR, name)
    assert os.path.isdir(d), f"missing {d}; run v3/make_hyper_datasets.py first"
    X = torch.load(os.path.join(d, "features.pt"))
    y = torch.load(os.path.join(d, "labels.pt"))
    edge_list = torch.load(os.path.join(d, "edge_list.pt"))
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return X, y, edge_list, meta


def fix_iso_v(G: Hypergraph):
    """Add a self-hyperedge for every isolated vertex (LightHGNN `fix_iso_v`)."""
    iso_v = np.array(G.deg_v) == 0
    if np.any(iso_v):
        extra_e = [tuple([int(e)]) for e in np.where(iso_v)[0]]
        G.add_hyperedges(extra_e)
    return G


# ---------------------------------------------------------------------------
# MLP baseline (feature-only)
# ---------------------------------------------------------------------------
class PlainMLP(nn.Module):
    def __init__(self, in_dim, hid, n_cls, dropout=0.5, input_drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, n_cls)
        self.dropout = dropout
        self.input_drop = input_drop

    def forward(self, X):
        X = F.dropout(X, self.input_drop, training=self.training)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout, training=self.training)
        return self.fc2(X)


# ---------------------------------------------------------------------------
# single seed experiment
# ---------------------------------------------------------------------------
def run_seed(model_name, X, y, edge_list, seed, device, hp):
    set_seed(seed)
    n_cls = int(y.max().item()) + 1

    train_mask, val_mask, test_mask = split_by_num(
        X.shape[0], y, hp["num_train"], hp["num_val"]
    )

    if model_name == "mlp":
        net = PlainMLP(X.shape[1], hp["mlp_hidden"], n_cls,
                       dropout=hp["mlp_dropout"], input_drop=hp["mlp_input_drop"])
        G = None
    else:
        G = Hypergraph(X.shape[0], edge_list)
        G = fix_iso_v(G)
        # dhg 0.9.3 quirk: lazily-built sparse caches (H, H_T, D_*, L_HGNN, ...)
        # are created on CPU regardless of `self.device`, and `.to(device)`
        # only moves tensors that are already cached. Force-build every cache
        # first, then move the whole structure.
        for _key in ("H", "H_T", "D_e", "D_e_neg_1", "D_v", "D_v_neg_1",
                     "D_v_neg_1_2", "L_HGNN"):
            try:
                getattr(G, _key)
            except Exception:
                pass
        G = G.to(device)
        # fail fast if any cache is still on the wrong device
        _dev = torch.device(device)
        for _key in ("H", "H_T", "D_e_neg_1", "D_v_neg_1", "L_HGNN"):
            _t = getattr(G, _key)
            assert _t.device == _dev, f"dhg cache {_key} is on {_t.device}, expected {_dev}"
        cls = {"hgnn": HGNN, "hgnnp": HGNNP, "hnhn": HNHN, "unigcn": UniGCN}[model_name]
        net = cls(X.shape[1], hp["hid"], n_cls, use_bn=False)

    net = net.to(device)
    Xd, yd = X.to(device), y.to(device)
    tm, vm, sm = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    optimizer = optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])

    def forward():
        return net(Xd, G) if G is not None else net(Xd)

    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(hp["epochs"]):
        # train
        net.train()
        optimizer.zero_grad()
        out = forward()
        loss = F.nll_loss(F.log_softmax(out[tm], dim=1), yd[tm])
        loss.backward()
        optimizer.step()
        # validate
        with torch.no_grad():
            net.eval()
            out = forward()
            val_acc = (out[vm].argmax(dim=1) == yd[vm]).float().mean().item()
            if val_acc > best_val:
                best_val, best_epoch = val_acc, epoch
                best_state = copy.deepcopy(net.state_dict())

    # test at best-val checkpoint
    net.load_state_dict(best_state)
    with torch.no_grad():
        net.eval()
        out = forward()
        test_acc = (out[sm].argmax(dim=1) == yd[sm]).float().mean().item()
        val_acc = (out[vm].argmax(dim=1) == yd[vm]).float().mean().item()
        train_loss = F.nll_loss(F.log_softmax(out[tm], dim=1), yd[tm]).item()

    return {
        "seed": seed,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "final_train_loss": train_loss,
        "num_train": int(tm.sum().item()),
        "num_val": int(vm.sum().item()),
        "num_test": int(sm.sum().item()),
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run(model_name, dataset_name, num_seeds, device, hp, output=None):
    assert model_name in ALL_MODELS, f"unknown model {model_name}"
    assert dataset_name in DATASETS, f"unknown dataset {dataset_name}"

    X, y, edge_list, meta = load_dataset(dataset_name)
    print(f"dataset {dataset_name}: {meta['num_vertices']} vertices, "
          f"{meta['num_hyperedges']} hyperedges, {meta['num_classes']} classes, "
          f"features {meta['dim_features']}-d")

    results = []
    t0 = time.time()
    for i, seed in enumerate(range(num_seeds)):
        r = run_seed(model_name, X, y, edge_list, seed, device, hp)
        results.append(r)
        print(f"  seed {seed}: val {r['val_acc']:.4f}, test {r['test_acc']:.4f} "
              f"(best epoch {r['best_epoch']}, train {r['num_train']}/"
              f"{r['num_val']}/{r['num_test']})")
    elapsed = time.time() - t0

    val_accs = np.array([r["val_acc"] for r in results])
    test_accs = np.array([r["test_acc"] for r in results])
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_seeds": num_seeds,
        "val_acc_mean": float(val_accs.mean()),
        "val_acc_std": float(val_accs.std()),
        "test_acc_mean": float(test_accs.mean()),
        "test_acc_std": float(test_accs.std()),
        "individual_results": results,
        "hyperparameters": hp,
        "dataset_meta": meta,
        "elapsed_seconds": elapsed,
    }

    print("=" * 64)
    print(f"Results: {model_name} / {dataset_name}")
    print(f"  Val  Acc: {val_accs.mean():.4f} ± {val_accs.std():.4f}")
    print(f"  Test Acc: {test_accs.mean():.4f} ± {test_accs.std():.4f}")
    print("=" * 64)

    out_dir = output or RES_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Results saved to", out_path)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--num-train", type=int, default=20, help="train vertices per class")
    p.add_argument("--num-val", type=int, default=100, help="val vertices per class")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--hid", type=int, default=32, help="hidden dim of HGNN models")
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-dropout", type=float, default=0.5)
    p.add_argument("--mlp-input-drop", type=float, default=0.5)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    hp = {
        "num_train": args.num_train,
        "num_val": args.num_val,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "hid": args.hid,
        "mlp_hidden": args.mlp_hidden,
        "mlp_dropout": args.mlp_dropout,
        "mlp_input_drop": args.mlp_input_drop,
    }
    run(args.model, args.dataset, args.num_seeds, args.device, hp, args.output)


if __name__ == "__main__":
    main()
