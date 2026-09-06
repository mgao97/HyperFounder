"""
Standard transductive node-classification trainer for hypergraph baselines.

Reuses the proven hypergraph models from baselines/HNN (MLP, HGNN/HCHA, HNHN) and
adds HGNN+ (attention) and UniGCN (GCN-style) variants (see models_extra.py).

Behaviour:
  * Loads a v3 coverage-split dataset from <data_root>/<dataset>/ (edge_list /
    features / labels / *mask pickles). Full node features are kept (no truncation).
  * Builds the hypergraph incidence (hyperedge_index = [node, edge]).
  * Trains with validation early stopping; reports test accuracy,
    5 seeds x mean +/- std, and writes a JSON summary.

Run per model, each pinned to its own GPU, e.g.:
    python run_baseline.py --model hgnn --dataset cora --device cuda:0
"""

import os
import sys
import json
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines.HNN.preprocessing import algo_preprocessing

# ---------------------------------------------------------------------------
# Model configurations (default hyperparameters shared across the 5 baselines)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "mlp":    {"hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.01, "wd": 5e-4,
               "epochs": 100, "input_drop": 0.6, "normalization": "None", "InputNorm": False},
    "hgnn":   {"hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.01, "wd": 5e-4,
               "epochs": 200, "input_drop": 0.6, "HCHA_symdegnorm": True},
    "hgnnp":  {"hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.01, "wd": 5e-4,
               "epochs": 200, "input_drop": 0.6, "heads": 8},
    "hnhn":   {"hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.01, "wd": 5e-4,
               "epochs": 200, "input_drop": 0.6, "alpha": -1.5, "beta": -0.5},
    "unigcn": {"hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.01, "wd": 5e-4,
               "epochs": 200, "input_drop": 0.6},
}


# ---------------------------------------------------------------------------
# Dataset loading (self-contained; keeps full node features)
# ---------------------------------------------------------------------------
def load_coverage_dataset(data_root, name, target_dim=None):
    import pickle

    d = os.path.join(data_root, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Dataset not found: {d}")

    def lp(f):
        with open(os.path.join(d, f), "rb") as fh:
            return pickle.load(fh)

    edge_list = lp("edge_list.pkl")
    feat = lp("features.pkl")
    labels = np.asarray(lp("labels.pkl")).flatten()
    tr = np.asarray(lp("train_mask.pkl")).astype(bool).flatten()
    va = np.asarray(lp("val_mask.pkl")).astype(bool).flatten()
    te = np.asarray(lp("test_mask.pkl")).astype(bool).flatten()

    if hasattr(feat, "toarray"):
        feat = feat.toarray()
    feat = np.asarray(feat, dtype=np.float32)
    if target_dim and feat.shape[1] > target_dim:
        feat = feat[:, :target_dim]

    x = torch.tensor(feat, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # hyperedge_index = [node_ids, edge_ids]
    nodes, edges = [], []
    for e, members in enumerate(edge_list):
        for v in members:
            nodes.append(v)
            edges.append(e)
    hyperedge_index = torch.tensor([nodes, edges], dtype=torch.long)

    data = Data(x=x, y=y)
    data.hyperedge_index = hyperedge_index
    data.num_nodes = x.shape[0]
    data.num_edges = len(edge_list)
    data.train_mask = torch.tensor(tr)
    data.val_mask = torch.tensor(va)
    data.test_mask = torch.tensor(te)
    return data


# ---------------------------------------------------------------------------
# Preprocessing / model construction (mirrors baselines/run_hnn_benchmark.py)
# ---------------------------------------------------------------------------
def preprocess_data(data, model_name, params):
    class PreprocessArgs:
        def __init__(self):
            self.method = model_name
            self.device = torch.device("cpu")
            self.dname = "cora"
            self.task_type = "node_cls"
            self.mediator = False
            self.chunk_size = 1000
            self.threshold = 0.0
            self.norm_type = 0
            self.init_val = 1.0
            self.init_type = 1
            self.HNHN_alpha = params.get("alpha", params.get("HNHN_alpha", -1.5))
            self.HNHN_beta = params.get("beta", params.get("HNHN_beta", -0.5))
            self.lam0 = params.get("lam0", 10)
            self.lam1 = params.get("lam1", 10)
            self.M = params.get("M", 32)
            self.HyperGCN_fast = params.get("HyperGCN_fast", True)
            self.HyperGCN_mediators = params.get("HyperGCN_mediators", False)

    args = PreprocessArgs()
    return algo_preprocessing(data, args)


def create_model_wrapper(num_features, num_classes, model_name, params):
    from baselines.HNN.mlp import MLP
    from baselines.HNN.hgnn import HCHA
    from baselines.HNN.hnhn import HNHN
    from baselines.HNN.unignn import UniGNN
    from models_extra import HGNNP, UniGCN

    class Args:
        pass

    args = Args()
    args.MLP_hidden = params["hidden"]
    args.All_num_layers = params["layers"]
    args.dropout = params["dropout"]
    args.HCHA_symdegnorm = params.get("HCHA_symdegnorm", True)
    args.HNHN_nonlinear_inbetween = True
    args.normalization = params.get("normalization", "None")
    args.InputNorm = params.get("InputNorm", False)
    args.heads = params.get("heads", 8)
    args.method = "UniGCN" if model_name == "unigcn" else "UniGIN"
    args.uni_heads = 1
    args.first_aggregate = "mean" if model_name == "unigcn" else "sum"
    args.second_aggregate = "sum"
    args.use_norm = False
    args.use_attention = False
    args.attn_drop = 0.0
    args.activation = "relu"
    args.input_drop = params.get("input_drop", 0.6)

    if model_name == "mlp":
        return MLP(
            num_features, params["hidden"], num_classes, params["layers"],
            dropout=params["dropout"],
            Normalization=params.get("normalization", "None"),
            InputNorm=params.get("InputNorm", False),
        )
    elif model_name == "hgnn":
        return HCHA(num_features, num_classes, args)
    elif model_name == "hgnnp":
        return HGNNP(num_features, num_classes, args)
    elif model_name == "hnhn":
        return HNHN(num_features, num_classes, args)
    elif model_name == "unigcn":
        return UniGCN(num_features, num_classes, args)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def forward_model(model, data, model_name):
    if model_name == "mlp":
        return model(data.x)
    return model(data)


def train_epoch(model, data, optimizer, model_name):
    model.train()
    optimizer.zero_grad()
    out = forward_model(model, data, model_name)
    if isinstance(out, tuple):
        out = out[0]
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, model_name):
    model.eval()
    out = forward_model(model, data, model_name)
    if isinstance(out, tuple):
        out = out[0]
    pred = out.argmax(dim=1)
    res = {}
    if data.val_mask is not None:
        res["val_acc"] = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    if data.test_mask is not None:
        res["test_acc"] = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return res


def run_single_experiment(model_name, data, num_classes, params,
                          max_epochs, patience, seed, device):
    set_seed(seed)
    data = preprocess_data(data, model_name, params)
    num_features = data.x.size(1)
    model = create_model_wrapper(num_features, num_classes, model_name, params)
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params.get("wd", 0.0)
    )

    best_val_acc, best_test_acc, bad_epochs, last_loss = 0.0, 0.0, 0, 0.0
    for epoch in range(max_epochs):
        last_loss = train_epoch(model, data, optimizer, model_name)
        metrics = evaluate(model, data, model_name)
        val_acc = metrics.get("val_acc", 0.0)
        test_acc = metrics.get("test_acc", 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    return {
        "val_acc": best_val_acc,
        "test_acc": best_test_acc,
        "epochs_trained": epoch + 1,
        "final_train_loss": last_loss,
    }


def run_baseline(model_name, dataset_name, data_root, num_seeds=5,
                 max_epochs=None, patience=50, device="cuda:0",
                 target_dim=None, output="v3/baselines/results",
                 overrides=None):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    params = copy.deepcopy(MODEL_CONFIGS[model_name])
    params["epochs"] = min(params.get("epochs", 100), max_epochs or params.get("epochs", 100))

    # apply CLI overrides (only keys explicitly provided)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                params[k] = v

    print(f"\n{'='*64}\nRunning {model_name} on {dataset_name}\n{'='*64}")
    print(f"Hyperparameters: {params}")

    data = load_coverage_dataset(data_root, dataset_name, target_dim)
    num_classes = int(data.y.max().item()) + 1
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
          f"Features: {data.x.size(1)}, Classes: {num_classes}")
    print(f"  Train: {int(data.train_mask.sum())}, Val: {int(data.val_mask.sum())}, "
          f"Test: {int(data.test_mask.sum())}")

    results = []
    seeds = [7 + i * 100 for i in range(num_seeds)]
    for i, seed in enumerate(seeds):
        print(f"\n  Seed {i+1}/{num_seeds} (seed={seed})...")
        try:
            res = run_single_experiment(
                model_name, data, num_classes, params,
                params["epochs"], patience, seed, device,
            )
            results.append(res)
            print(f"    Val Acc: {res['val_acc']:.4f}, Test Acc: {res['test_acc']:.4f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

    if not results:
        return {"error": f"All experiments failed for {model_name} on {dataset_name}"}

    val_accs = [r["val_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_seeds": len(results),
        "val_acc_mean": float(np.mean(val_accs)),
        "val_acc_std": float(np.std(val_accs)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "individual_results": results,
        "hyperparameters": params,
    }

    print(f"\n{'='*64}\nResults Summary: {model_name} / {dataset_name}")
    print(f"  Val  Acc: {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f}")
    print(f"  Test Acc: {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"{'='*64}")

    os.makedirs(output, exist_ok=True)
    out_file = os.path.join(output, f"{model_name}_{dataset_name}.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_file}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transductive hypergraph baseline trainer")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()), help="Baseline model name")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name (folder under --data-root)")
    parser.add_argument("--data-root", type=str, default=os.path.join(ROOT, "v3", "datasets"),
                        help="Root dir containing per-dataset folders")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Cap on training epochs (default: per-model config)")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target-dim", type=int, default=None,
                        help="Feature dim cap (None = keep full features)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--wd", type=float, default=None, help="Override weight decay")
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden dim")
    parser.add_argument("--layers", type=int, default=None, help="Override num layers")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    parser.add_argument("--input-drop", type=float, default=None, help="Override input dropout")
    parser.add_argument("--heads", type=int, default=None, help="Override attention heads (hgnnp)")
    parser.add_argument("--alpha", type=float, default=None, help="Override HNHN alpha")
    parser.add_argument("--beta", type=float, default=None, help="Override HNHN beta")
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "v3", "baselines", "results"))
    args = parser.parse_args(argv)

    overrides = {
        "lr": args.lr, "wd": args.wd, "hidden": args.hidden, "layers": args.layers,
        "dropout": args.dropout, "input_drop": args.input_drop, "heads": args.heads,
        "alpha": args.alpha, "beta": args.beta,
    }

    run_baseline(
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        num_seeds=args.num_seeds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        target_dim=args.target_dim,
        overrides=overrides,
        output=args.output,
    )


if __name__ == "__main__":
    main()
