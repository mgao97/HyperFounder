"""
LightHGNN reproduction, aligned with the other baselines' experimental protocol.

LightHGNN (iMoonLab/LightHGNN) = a lightweight MLP ("student") distilled from a
hypergraph teacher (HGNN by default) using (a) a KL knowledge-distillation term
over the full transductive graph and (b) a High-Order Constraint (HOC) term that
aligns the teacher/student high-order (hyperedge) predictions. The distilled MLP
is the reported model; it keeps HGNN-level accuracy at MLP inference cost.

This script reuses the EXACT data / split / optimizer settings of
`run_hyper_baseline.py` so the number is directly comparable:
  - split: 20 train / 100 val vertices per class, rest test (dhg split_by_num),
    re-drawn per seed after dhg.random.set_seed;
  - teacher HGNN: hid=32, use_bn=False;  hypergraph = dhg.Hypergraph +
    self-hyperedges for isolated vertices (fix_iso_v);
  - student MyMLPs: 2-layer MLP (in -> hid=128 -> n_cls), no dropout;
  - optimizer Adam(lr=1e-2, weight_decay=5e-4), full batch, 200 epochs,
    best-val-accuracy checkpointing, test metrics reported at best epoch;
  - 5 seeds (default 0..4), report mean +/- std.

Faithful to LightHGNNs-src/trans_train.py:
  loss_x = NLL(softmax(outs[train]), y[train])
  loss_k = KL(log_softmax(outs), softmax(out_t))            # full-graph KD
  loss_k = loss_k + HOC(outs, out_t, G)                    # high-order constraint
  loss   = loss_x * lamb + loss_k * (1 - lamb)             # default lamb=0 -> pure KD
Default teacher=hgnn, student=dis_hgnnp (HOC on), lamb=0, tau=1.0, hid=128,
hc_noise_level=1.0  (matches trans_config.yaml).

Usage:
  python run_lighthgnn.py --dataset news20 --device cuda:4
  python run_lighthgnn.py --teacher hgnn --dataset all --device cuda:4
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
from dhg.models import HGNN

ROOT = "/home/user/GSK/mgao/HyperFounder"
DATA_DIR = os.path.join(ROOT, "v3", "datasets")
RES_DIR = os.path.join(ROOT, "v3", "baselines", "results", "lighthgnn")

DATASETS = [
    "news20", "ca_cora", "cc_cora", "cc_citeseer",
    "dblp4k_paper", "dblp4k_term", "dblp4k_conf", "imdb_aw",
]


# ---------------------------------------------------------------------------
# data / structure  (identical to run_hyper_baseline.py)
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


def build_hypergraph(X, edge_list, device):
    G = Hypergraph(X.shape[0], edge_list)
    G = fix_iso_v(G)
    for _key in ("H", "H_T", "D_e", "D_e_neg_1", "D_v", "D_v_neg_1",
                 "D_v_neg_1_2", "L_HGNN"):
        try:
            getattr(G, _key)
        except Exception:
            pass
    G = G.to(device)
    _dev = torch.device(device)
    for _key in ("H", "H_T", "D_e_neg_1", "D_v_neg_1", "L_HGNN"):
        _t = getattr(G, _key)
        assert _t.device == _dev, f"dhg cache {_key} on {_t.device}, expected {_dev}"
    return G


# ---------------------------------------------------------------------------
# LightHGNN student MLP (mirrors MyMLPs: MLP([in, hid]) -> Linear(hid, cls))
# ---------------------------------------------------------------------------
class MyMLPs(nn.Module):
    def __init__(self, in_dim, hid, n_cls):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, n_cls)

    def forward(self, X, get_emb=False):
        h = F.relu(self.fc1(X))
        if get_emb:
            return h
        return self.fc2(h)


# ---------------------------------------------------------------------------
# High-Order Constraint (HOC) — ported verbatim from trans_train.py
# ---------------------------------------------------------------------------
class HighOrderConstraint(nn.Module):
    def __init__(self, model, X, G, noise_level=1.0, tau=1.0):
        super().__init__()
        model.eval()
        self.tau = tau
        pred = model(X, G).softmax(dim=-1).detach()
        entropy_x = -(pred * pred.log()).sum(1, keepdim=True)
        entropy_x[entropy_x.isnan()] = 0
        entropy_e = G.v2e(entropy_x, aggr="mean")

        X_noise = X.clone() * (torch.randn_like(X) + 1) * noise_level
        pred_ = model(X_noise, G).softmax(dim=-1).detach()
        entropy_x_ = -(pred_ * pred_.log()).sum(1, keepdim=True)
        entropy_x_[entropy_x_.isnan()] = 0
        entropy_e_ = G.v2e(entropy_x_, aggr="mean")

        self.delta_e_ = (entropy_e_ - entropy_e).abs()
        self.delta_e_ = 1 - self.delta_e_ / self.delta_e_.max()
        self.delta_e_ = self.delta_e_.squeeze()

    def forward(self, pred_s, pred_t, G):
        pred_s, pred_t = F.softmax(pred_s, dim=1), F.softmax(pred_t, dim=1)
        e_mask = torch.bernoulli(self.delta_e_).bool()
        pred_s_e = G.v2e(pred_s, aggr="mean")
        pred_s_e = pred_s_e[e_mask]
        pred_t_e = G.v2e(pred_t, aggr="mean")
        pred_t_e = pred_t_e[e_mask]
        loss = F.kl_div(torch.log(pred_s_e / self.tau), pred_t_e / self.tau,
                        reduction="batchmean", log_target=True)
        return loss


# ---------------------------------------------------------------------------
# single seed experiment
# ---------------------------------------------------------------------------
def run_seed(teacher_name, X, y, edge_list, seed, device, hp):
    set_seed(seed)
    n_cls = int(y.max().item()) + 1
    train_mask, val_mask, test_mask = split_by_num(
        X.shape[0], y, hp["num_train"], hp["num_val"])

    G = build_hypergraph(X, edge_list, device)
    Xd, yd = X.to(device), y.to(device)
    tm, vm, sm = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    # ---- teacher ----
    net_t = HGNN(X.shape[1], hp["teacher_hid"], n_cls, use_bn=False).to(device)
    opt_t = optim.Adam(net_t.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(hp["epochs"]):
        net_t.train(); opt_t.zero_grad()
        out = net_t(Xd, G)
        loss = F.nll_loss(F.log_softmax(out[tm], dim=1), yd[tm])
        loss.backward(); opt_t.step()
        with torch.no_grad():
            net_t.eval()
            val = (net_t(Xd, G)[vm].argmax(1) == yd[vm]).float().mean().item()
            if val > best_val:
                best_val, best_epoch, best_state = val, epoch, copy.deepcopy(net_t.state_dict())
    net_t.load_state_dict(best_state)
    teacher_test = (net_t(Xd, G)[sm].argmax(1) == yd[sm]).float().mean().item()

    with torch.no_grad():
        out_t = net_t(Xd, G).detach()

    # ---- student (LightHGNN) ----
    if hp["use_hoc"]:
        hc = HighOrderConstraint(net_t, Xd, G,
                                 noise_level=hp["hc_noise_level"], tau=hp["tau"]).to(device)
    else:
        hc = None

    net_s = MyMLPs(X.shape[1], hp["student_hid"], n_cls).to(device)
    opt_s = optim.Adam(net_s.parameters(), lr=hp["lr"], weight_decay=hp["wd"])

    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(hp["epochs"]):
        net_s.train(); opt_s.zero_grad()
        outs = net_s(Xd)
        loss_x = F.nll_loss(F.log_softmax(outs[tm], dim=1), yd[tm])
        loss_k = F.kl_div(F.log_softmax(outs, dim=1), F.softmax(out_t, dim=1),
                          reduction="batchmean", log_target=True)
        if hc is not None:
            loss_h = hc(outs, out_t, G)
            loss_k = loss_h + loss_k
        loss = loss_x * hp["lamb"] + loss_k * (1 - hp["lamb"])
        loss.backward(); opt_s.step()
        with torch.no_grad():
            net_s.eval()
            val = (net_s(Xd)[vm].argmax(1) == yd[vm]).float().mean().item()
            if val > best_val:
                best_val, best_epoch, best_state = val, epoch, copy.deepcopy(net_s.state_dict())
    net_s.load_state_dict(best_state)
    with torch.no_grad():
        net_s.eval()
        student_test = (net_s(Xd)[sm].argmax(1) == yd[sm]).float().mean().item()
        student_val = (net_s(Xd)[vm].argmax(1) == yd[vm]).float().mean().item()

    return {
        "seed": seed,
        "teacher_test_acc": teacher_test,
        "student_test_acc": student_test,
        "student_val_acc": student_val,
        "best_epoch": best_epoch,
        "num_train": int(tm.sum().item()),
        "num_val": int(vm.sum().item()),
        "num_test": int(sm.sum().item()),
    }


def run(teacher_name, dataset_name, num_seeds, device, hp, output=None):
    X, y, edge_list, meta = load_dataset(dataset_name)
    print(f"[LightHGNN] dataset {dataset_name}: {meta['num_vertices']} vertices, "
          f"{meta['num_hyperedges']} hyperedges, {meta['num_classes']} classes")

    results = []
    t0 = time.time()
    for i, seed in enumerate(range(num_seeds)):
        r = run_seed(teacher_name, X, y, edge_list, seed, device, hp)
        results.append(r)
        print(f"  seed {seed}: teacher_test {r['teacher_test_acc']:.4f}, "
              f"student(LightHGNN)_test {r['student_test_acc']:.4f} "
              f"(best ep {r['best_epoch']})")
    elapsed = time.time() - t0

    t_accs = np.array([r["teacher_test_acc"] for r in results])
    s_accs = np.array([r["student_test_acc"] for r in results])
    summary = {
        "method": "LightHGNN",
        "teacher": teacher_name,
        "student": "dis_hgnnp" if hp["use_hoc"] else "mlp",
        "dataset": dataset_name,
        "num_seeds": num_seeds,
        "teacher_test_acc_mean": float(t_accs.mean()),
        "teacher_test_acc_std": float(t_accs.std()),
        "test_acc_mean": float(s_accs.mean()),
        "test_acc_std": float(s_accs.std()),
        "individual_results": results,
        "hyperparameters": hp,
        "dataset_meta": meta,
        "elapsed_seconds": elapsed,
    }

    print("=" * 64)
    print(f"LightHGNN (teacher={teacher_name}): {dataset_name}")
    print(f"  Teacher Test Acc: {t_accs.mean():.4f} +/- {t_accs.std():.4f}")
    print(f"  Student Test Acc: {s_accs.mean():.4f} +/- {s_accs.std():.4f}")
    print("=" * 64)

    out_dir = output or RES_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lighthgnn_{teacher_name}_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Results saved to", out_path)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="hgnn", choices=["hgnn"])
    p.add_argument("--dataset", required=True, choices=DATASETS + ["all"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--num-train", type=int, default=20)
    p.add_argument("--num-val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--teacher-hid", type=int, default=32)
    p.add_argument("--student-hid", type=int, default=128)
    p.add_argument("--lamb", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--hc-noise-level", type=float, default=1.0)
    p.add_argument("--no-hoc", action="store_true", help="disable High-Order Constraint")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    hp = {
        "num_train": args.num_train,
        "num_val": args.num_val,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "teacher_hid": args.teacher_hid,
        "student_hid": args.student_hid,
        "lamb": args.lamb,
        "tau": args.tau,
        "hc_noise_level": args.hc_noise_level,
        "use_hoc": not args.no_hoc,
    }
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run(args.teacher, ds, args.num_seeds, args.device, hp, args.output)


if __name__ == "__main__":
    main()
