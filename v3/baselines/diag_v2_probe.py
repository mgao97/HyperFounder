"""
诊断 v2 预训练 encoder 在 8 数据集协议下的表示质量。

1) 校验输入特征管线是否与预训练一致 (用 data/cache 里的原始 pickles 复算对比)
2) 冻结 encoder -> node_tok, 用 v2 官方口径 (StandardScaler + LBFGS logistic,
   训 tr|va) 做 linear probe, 与 140-train 口径对比
3) 输出 embedding 统计 (是否有 NaN / 塌缩)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = "/home/user/GSK/mgao/HyperFounder"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dhg.random import set_seed
from dhg.utils import split_by_num

from v2.utils.dhg_datasets import _resize_features
from v3.baselines.run_v2_main import (DATA_DIR, apply_structure_caching, build_incidence,
                                      fix_iso_edges, load_dataset, load_raw_features)


def lbfgs_probe(emb: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, n_cls: int):
    Xtr = torch.from_numpy(emb[tr]).float()
    ytr = torch.from_numpy(y[tr]).long()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (torch.from_numpy(emb[te]).float() - mu) / sd
    model = nn.Linear(Xtr.size(1), n_cls)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500,
                            tolerance_grad=1e-4, tolerance_change=1e-5,
                            line_search_fn="strong_wolfe")
    crit = nn.CrossEntropyLoss()
    l2 = 0.5 / Xtr.size(0)

    def closure():
        opt.zero_grad()
        loss = crit(model(Xtr), ytr) + l2 * sum((p ** 2).sum() for p in model.parameters())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == torch.from_numpy(y[te]).long()).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cc_cora")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs_v2_fromscratch/checkpoints/pretrain_best_v2.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe-raw-baseline", action="store_true")
    args = ap.parse_args()

    set_seed(0)
    X, y, edge_list, meta = load_dataset(args.dataset)
    dev = torch.device(args.device)

    # ---- (1) 输入特征管线一致性校验 -------------------------------------
    alias = {"cc_cora": "cocitation_cora", "cc_citeseer": "cocitation_citeseer"}
    cache_dir = os.path.join(ROOT, "data", "cache", alias.get(args.dataset, args.dataset))
    if os.path.isdir(cache_dir) and os.path.isfile(os.path.join(cache_dir, "features.pkl")):
        with open(os.path.join(cache_dir, "features.pkl"), "rb") as f:
            raw = pickle.load(f)
        if hasattr(raw, "toarray"):          # scipy sparse
            raw = raw.toarray()
        raw_t = torch.as_tensor(np.asarray(raw))
        ref = _resize_features(raw_t, target_dim=128, seed=7)
        mine = _resize_features(X, target_dim=128, seed=7)
        same = torch.allclose(ref, mine, atol=1e-6)
        print(f"[input-check] cache={cache_dir} exists; projected features match: {same} "
              f"(ref {tuple(ref.shape)} vs mine {tuple(mine.shape)})")
    else:
        print(f"[input-check] no data/cache features for {args.dataset} at {cache_dir}; skip")

    # ---- (2) 两种 incidence (fix_iso / no) 下的冻结表示探针 --------------
    from v2.models.encoder_v2 import build_encoder_v2
    ck = torch.load(args.ckpt, map_location="cpu")
    mc = ck["config"]["model"]
    x = load_raw_features(args.dataset, 128, 7, fallback=X).to(dev)
    N = x.size(0)
    n_cls = int(y.max().item()) + 1

    for tag, el in (("no_fixiso", edge_list), ("fixiso", fix_iso_edges(N, edge_list))):
        H = build_incidence(N, el, dev)
        enc = build_encoder_v2(
            in_dim=int(mc["input_dim"]), hidden_dim=int(mc["hidden_dim"]),
            num_layers=int(mc.get("num_layers", 3)), num_heads=int(mc.get("num_heads", 4)),
            dropout=float(mc.get("dropout", 0.1)), pe_dim=int(mc.get("pe_dim", 32)),
            hca_topk=int(mc.get("hca_topk", 16)), use_hor=bool(mc.get("use_hor", True)),
        )
        apply_structure_caching(enc)
        enc.load_state_dict(ck["encoder"], strict=False)
        enc = enc.to(dev).eval()
        with torch.no_grad():
            node_t, *_ = enc(x, H)
        emb = node_t.detach().cpu().numpy().astype(np.float32)
        print(f"[{tag}] emb: nan={np.isnan(emb).sum()} std={emb.std():.4f} "
              f"mean|v|={np.abs(emb).mean():.4f} distinct_rows~{len(np.unique(emb.round(5), axis=0))}/{N}")

        ynp = y.numpy().astype(np.int64)
        tr, va, te = split_by_num(N, y, 20, 100)
        tr, va, te = tr.numpy(), va.numpy(), te.numpy()

        # 官方口径: 训 tr|va, 测 te
        acc_trva = lbfgs_probe(emb, ynp, tr | va, te, n_cls)
        # 本协议口径: 只训 tr, 测 te
        acc_tr = lbfgs_probe(emb, ynp, tr, te, n_cls)
        if args.probe_raw_baseline:
            raw_x = x.detach().cpu().numpy().astype(np.float32)
            acc_raw = lbfgs_probe(raw_x, ynp, tr | va, te, n_cls)
            print(f"[{tag}] probe train=tr|va: {acc_trva:.4f} | train=tr only: {acc_tr:.4f} "
                  f"| raw-feature probe(tr|va): {acc_raw:.4f}")
        else:
            print(f"[{tag}] probe train=tr|va: {acc_trva:.4f} | train=tr only: {acc_tr:.4f}")
        del enc, emb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
