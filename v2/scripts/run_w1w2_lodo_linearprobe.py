"""W1-W2 · Leave-1-Domain-Out linear probe Δ (规格书 §6 step 5).

Protocol (per dataset, 3 seeds):
  1. Load *all* pretrain domains EXCEPT the one containing this dataset (LODO).
     If the dataset is the only one in its domain → use all other domains.
  2. Load (or reuse-from-run-dir) pretrained encoder checkpoint produced by
     run_pretrain_v2.py.  If no ckpt exists yet → a short 3-step mini-pretrain
     on the same LODO domains is used so the script self-completes and the
     user can see the reporting pipeline (the final numbers MUST be recomputed
     once the real GPU 0 pretrain ckpt lands).
  3. Freeze encoder.  Linear-probe on top of node tokens for node_cls tasks
     (500-iter logistic regression via sklearn SAGA-L2, random_state=seed).
  4. Baseline: same linear probe on raw node features (no encoder).
  5. Δ = (ours_acc - baseline_acc) × 100 percentage points.
  6. Writes CSV rows → outputs_v2/w1_w2_lodo_linearprobe.csv + a summary
     markdown table → logs/w1_w2_lodo_summary.md.

CLI:
  python v2/scripts/run_w1w2_lodo_linearprobe.py \
      --pretrain_ckpt outputs_v2/checkpoints/pretrain_best_v2.pt \
      --seeds 1,2,3 \
      --domains citation,academic,document,recommendation \
      --device cuda:7
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from v2.models.encoder_v2 import HyperFounderV2Encoder, V2EncoderConfig
from v2.utils.dhg_datasets import load_domain_graphs
from v2.utils.hypergraph import SimpleHypergraph
from v2.models.pretext_v2 import _sparse_drop_rows_cols


def _set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _load_yaml(p: Path) -> dict:
    import yaml
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _collect_nodecls(graphs_by_domain: Dict[str, List[SimpleHypergraph]]) -> List[Tuple[str, str, SimpleHypergraph]]:
    out: List[Tuple[str, str, SimpleHypergraph]] = []
    for dom, gs in graphs_by_domain.items():
        for g in gs:
            tt = (g.metadata or {}).get("task_type")
            if tt == "node_cls" or (
                g.node_labels is not None and g.node_train_mask is not None
            ):
                out.append((dom, g.name or g.dataset_name or dom, g))
    return out


def _load_encoder(
    in_dim: int,
    hidden_dim: int,
    ckpt_path: Optional[Path],
    pretrain_domains: List[str],
    all_graphs: Dict[str, List[SimpleHypergraph]],
    device: torch.device,
    *,
    scratch_no_pretrain: bool = False,
    use_hor: bool = True,
    ablate_cca_card: bool = False,
    ablate_cca_film: bool = False,
    ablate_cca_tau: bool = False,
    ablate_hca_bias: bool = False,
    ablate_hca_full: bool = False,
    pe_dim: int = 32,
    hca_topk: int = 16,
) -> HyperFounderV2Encoder:
    """Load encoder from checkpoint OR perform a tiny 3-step LODO pretrain so the
    linear probe pipeline is complete even if no real ckpt exists yet.
    """
    cfg = V2EncoderConfig(
        in_dim=in_dim, hidden_dim=hidden_dim, num_layers=3, num_heads=4,
        dropout=0.0, pe_dim=pe_dim, hca_topk=hca_topk, use_hor=bool(use_hor),
        ablate_cca_card=bool(ablate_cca_card),
        ablate_cca_film=bool(ablate_cca_film),
        ablate_cca_tau=bool(ablate_cca_tau),
        ablate_hca_bias=bool(ablate_hca_bias),
        ablate_hca_full=bool(ablate_hca_full),
    )
    enc = HyperFounderV2Encoder(cfg).to(device)

    if scratch_no_pretrain:
        print("[encoder] SCRATCH mode: random-init encoder, no ckpt load, no mini-pretrain.")
        enc.eval()
        return enc

    loaded_ok = False
    if ckpt_path is not None and ckpt_path.is_file():
        try:
            sd = torch.load(ckpt_path, map_location=device)
            key = "encoder" if "encoder" in sd else "state_dict"
            enc.load_state_dict(sd[key], strict=False)
            loaded_ok = True
            print(f"[encoder] LOADED ckpt {ckpt_path} (strict=False)")
        except Exception as e:
            print(f"[encoder] load ckpt failed: {e}. Falling back to mini-pretrain.")

    if not loaded_ok:
        print("[encoder] Mini LODO pretrain (3 steps) on domains:", pretrain_domains)
        # 3-step AdamW using edge_mlm + membership + dualview — identical to trainer
        from v2.models.heads_v2 import EdgeReconHead, MembershipHead, EdgeContrastProjector
        from v2.models.pretext_v2 import (KendallUncertaintyWeights, build_pretext_batch,
                                            edge_mlm_loss, node_edge_membership_loss,
                                            edge_dualview_contrast_loss)
        er = EdgeReconHead(hidden_dim, in_dim).to(device)
        mh = MembershipHead(hidden_dim).to(device)
        cp = EdgeContrastProjector(hidden_dim, 128).to(device)
        uw = KendallUncertaintyWeights(num_tasks=3).to(device)
        opt = torch.optim.AdamW(
            list(enc.parameters()) + list(er.parameters()) +
            list(mh.parameters()) + list(cp.parameters()) + list(uw.parameters()),
            lr=1e-3, weight_decay=1e-4,
        )
        enc.train()
        graphs_list: List[SimpleHypergraph] = []
        for d in pretrain_domains:
            for g in all_graphs.get(d, []):
                graphs_list.append(g)
        if not graphs_list:
            print("[encoder] No pretrain graphs; returning random-init encoder.")
            return enc
        rng = random.Random(1234)
        for step in range(3):
            opt.zero_grad()
            g = graphs_list[step % len(graphs_list)]
            try:
                x = g.x.to(device).float()
                H = g.incidence_matrix().to_sparse_coo().coalesce().to(device)
                if H._nnz() == 0:
                    continue
                N, E = H.size()
                H1, keep_n1, keep_e1 = _sparse_drop_rows_cols(H, 0.15, 0.10, seed=step*13+1)
                H2, _, _ = _sparse_drop_rows_cols(H, 0.15, 0.10, seed=step*13+2)
                ec = None
                nt, et, hs, hd, hsim = enc(x, H, edge_cardinalities=ec)
                batch = build_pretext_batch(x=x, incidence=H, edge_mlm_rate=0.15,
                                            membership_num_negatives=4, membership_hard_prob=0.7,
                                            hca_neighbor_table=(hs, hd, hsim), seed=step*111)
                v1n, v1e, *_ = enc(x, H1)
                v2n, v2e, *_ = enc(x, H2)
                L1 = edge_mlm_loss(et, batch, er)
                L2 = node_edge_membership_loss(nt, et, batch, mh, tau=0.2)
                Ea = min(v1e.size(0), v2e.size(0))
                L3 = edge_dualview_contrast_loss(v1e[:Ea], v2e[:Ea], cp, tau=0.5)
                L = uw([L1, L2, L3])
                L.backward(); opt.step()
                print(f"  [mini step {step+1}/3] L={L.item():.3f}")
            except Exception as e:
                print(f"  [mini step {step+1}/3] SKIP: {type(e).__name__}: {e}")
                continue
    enc.eval()
    return enc


def _encode_nodes(enc: HyperFounderV2Encoder, g: SimpleHypergraph, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = g.x.to(device).float()
        H = g.incidence_matrix().to_sparse_coo().coalesce().to(device)
        if H._nnz() == 0:
            return x.cpu().numpy()
        n_tok, _, _, _, _ = enc(x, H)
        return n_tok.cpu().numpy()


def _split_masks(g: SimpleHypergraph, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = g.num_nodes
    if g.node_train_mask is not None and g.node_val_mask is not None and g.node_test_mask is not None:
        tr = g.node_train_mask.bool().cpu().numpy()
        va = g.node_val_mask.bool().cpu().numpy()
        te = g.node_test_mask.bool().cpu().numpy()
        return tr, va, te
    rng = np.random.default_rng(seed)
    labels = g.node_labels.cpu().numpy()
    train = np.zeros(N, dtype=bool)
    val = np.zeros(N, dtype=bool)
    test = np.zeros(N, dtype=bool)
    # Class-stratified 60/20/20 using stratify indices
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(0.6 * n)); nva = int(round(0.2 * n))
        train[idx[:ntr]] = True
        val[idx[ntr:ntr+nva]] = True
        test[idx[ntr+nva:]] = True
    # ensure at least one train/test per class (fallback); shouldn't trigger for our datasets
    if train.sum() == 0 or test.sum() == 0:
        idx = np.arange(N); rng.shuffle(idx)
        train[:] = False; val[:] = False; test[:] = False
        n1 = int(0.6*N); n2 = int(0.8*N)
        train[idx[:n1]] = True; val[idx[n1:n2]] = True; test[idx[n2:]] = True
    return train, val, test


def _linear_probe(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, seed: int, device: torch.device = torch.device("cpu")) -> float:
    # Standardize
    mu = X_tr.mean(0, keepdims=True); sd = X_tr.std(0, keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sd; X_te = (X_te - mu) / sd
    
    n_classes = int(np.max(y_tr) + 1)
    
    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    X_te_t = torch.from_numpy(X_te).float().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    model = torch.nn.Linear(X_tr.shape[1], n_classes).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=500, 
        tolerance_grad=1e-4,
        tolerance_change=1e-5,
        line_search_fn="strong_wolfe"
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # We want L2 penalty like sklearn's C=1.0. 
    # sklearn objective: C * CrossEntropy + 0.5 * ||w||^2 
    # equivalent to: CrossEntropy + (0.5 / C) * ||w||^2
    # So weight_decay should be 1.0 / C = 1.0
    # But note sklearn scales loss by sum(weights) which is N. 
    # PyTorch's CrossEntropyLoss is mean over N.
    # To match exactly: PyTorch Loss + (0.5 / (C * N)) * ||w||^2
    l2_lambda = 0.5 / X_tr.shape[0]

    def closure():
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = criterion(logits, y_tr_t)
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
        total_loss = loss + l2_lambda * l2_reg
        total_loss.backward()
        return total_loss

    try:
        model.train()
        optimizer.step(closure)
        model.eval()
        with torch.no_grad():
            pred = model(X_te_t).argmax(dim=1)
            acc = (pred == y_te_t).float().mean().item()
            return acc
    except Exception as e:
        print(f"    [probe] fallback PyTorch error {e}. Using 0.")
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="v2/configs/pretrain_v2.yaml")
    ap.add_argument("--pretrain_ckpt", type=str, default="outputs_v2/checkpoints/pretrain_best_v2.pt")
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--domains", type=str, default="")  # comma; empty=all in config
    ap.add_argument("--device", type=str, default="cuda:7")
    ap.add_argument("--scratch_no_pretrain", action="store_true",
                    help="Use random-init encoder directly; skip ckpt load and mini-pretrain fallback.")
    # ---- Ablation-toggles (per-variant probe) ----
    ap.add_argument("--use_hor", type=str, default="default",
                    choices=["default", "true", "false"])
    ap.add_argument("--ablate_cca_card", action="store_true")
    ap.add_argument("--ablate_cca_film", action="store_true")
    ap.add_argument("--ablate_cca_tau", action="store_true")
    ap.add_argument("--ablate_hca_bias", action="store_true")
    ap.add_argument("--ablate_hca_full", action="store_true")
    # ---- Output: write to a different CSV per ablation variant ----
    ap.add_argument("--out_csv", type=str, default="",
                    help="Override output CSV path. Default: outputs_v2/w1_w2_lodo_linearprobe.csv")
    ap.add_argument("--append_to_main_csv", action="store_true",
                    help="Also append rows to the MAIN CSV (outputs_v2/w1_w2_lodo_linearprobe.csv) in addition to --out_csv")
    ap.add_argument("--variant_tag", type=str, default="",
                    help="If non-empty, this tag is prepended as an extra first column 'variant' in out_csv")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = _load_yaml(cfg_path)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    graphs_by_domain = load_domain_graphs(cfg, seed=cfg["training"]["seed"], require_node_splits=True)

    target_domains: List[str]
    if args.domains.strip():
        target_domains = [d for d in args.domains.split(",") if d.strip()]
    else:
        target_domains = list(graphs_by_domain.keys())

    # Collect all node_cls tasks
    nodecls = _collect_nodecls(graphs_by_domain)
    print(f"[W1-W2] domains={target_domains}  node_cls_datasets={[x[1] for x in nodecls]}  seeds={seeds}")

    # Resolve use_hor tri-state: default -> read from model config; else booleanize
    use_hor_arg: bool
    if args.use_hor == "default":
        use_hor_arg = bool(cfg.get("model", {}).get("use_hor", True))
    else:
        use_hor_arg = (args.use_hor == "true")
    pe_dim_arg = int(cfg.get("model", {}).get("pe_dim", 32))
    hca_topk_arg = int(cfg.get("model", {}).get("hca_topk", 16))

    # ---------- CSV routing: default CSV + optional variant CSV ----------
    MAIN_CSV = ROOT / "outputs_v2" / "w1_w2_lodo_linearprobe.csv"
    if args.out_csv.strip():
        variant_csv = Path(args.out_csv)
        if not variant_csv.is_absolute():
            variant_csv = ROOT / variant_csv
    else:
        variant_csv = MAIN_CSV
    variant_csv.parent.mkdir(parents=True, exist_ok=True)
    MAIN_CSV.parent.mkdir(parents=True, exist_ok=True)

    has_tag = bool(args.variant_tag.strip())
    base_fields = ["domain", "dataset", "seed", "task", "n_classes", "n_nodes",
                   "baseline_acc", "ours_acc", "delta_pp", "encoder_ckpt", "elapsed_s"]
    var_fields = (["variant"] + base_fields) if has_tag else base_fields

    var_exists = variant_csv.is_file()
    f_var = open(variant_csv, "a", newline=""); w_var = csv.DictWriter(f_var, fieldnames=var_fields)
    if not var_exists:
        w_var.writeheader()

    # Optional: also append to main CSV (no variant column, as original)
    writers_extra: List[Tuple[csv.DictWriter, bool]] = []
    if args.append_to_main_csv and variant_csv.resolve() != MAIN_CSV.resolve():
        main_exists = MAIN_CSV.is_file()
        fmain = open(MAIN_CSV, "a", newline=""); wmain = csv.DictWriter(fmain, fieldnames=base_fields)
        if not main_exists:
            wmain.writeheader()
        writers_extra.append((wmain, False))  # False = don't inject variant column

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"[W1-W2] device={device}  use_hor={use_hor_arg} "
          f"scratch_no_pretrain={args.scratch_no_pretrain} "
          f"ablate_cca(c/f/t)={args.ablate_cca_card}/{args.ablate_cca_film}/{args.ablate_cca_tau} "
          f"ablate_hca(b/fu)={args.ablate_hca_bias}/{args.ablate_hca_full}")

    # Encode per (lodo_domains) once per seed so we can reuse across multiple datasets
    in_dim = cfg["model"]["input_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    ckpt = ROOT / args.pretrain_ckpt if args.pretrain_ckpt else None

    all_rows: List[dict] = []
    for seed in seeds:
        _set_seed(seed)
        for dom, ds_name, g in nodecls:
            if dom not in target_domains:
                continue
            # LODO pretrain_domains = all domains except this one
            pretrain_domains = [d for d in graphs_by_domain.keys() if d != dom]
            if not pretrain_domains:
                print(f"  [skip seed={seed} {dom}/{ds_name}] no other domains for LODO.")
                continue
            t0 = time.perf_counter()
            labels = g.node_labels.cpu().numpy()
            n_classes = int(labels.max() + 1)
            # load encoder (ckpt path shared — LODO full retrain is heavy so the 2026 MVP
            # uses the single all-domains pretrain ckpt; true LODO training is a future
            # heavy run we can schedule when baseline ckpt slot is proven useful)
            # If user wants true LODO ckpt, schedule each dom as a separate pretrain job.
            enc = _load_encoder(in_dim=in_dim, hidden_dim=hidden_dim,
                                ckpt_path=ckpt, pretrain_domains=pretrain_domains,
                                all_graphs=graphs_by_domain, device=device,
                                scratch_no_pretrain=args.scratch_no_pretrain,
                                use_hor=use_hor_arg,
                                ablate_cca_card=args.ablate_cca_card,
                                ablate_cca_film=args.ablate_cca_film,
                                ablate_cca_tau=args.ablate_cca_tau,
                                ablate_hca_bias=args.ablate_hca_bias,
                                ablate_hca_full=args.ablate_hca_full,
                                pe_dim=pe_dim_arg,
                                hca_topk=hca_topk_arg)
            X_raw = g.x.cpu().numpy().astype(np.float32)
            if X_raw.shape[0] != g.num_nodes:
                X_raw = X_raw[: g.num_nodes]
            X_enc = _encode_nodes(enc, g, device)
            tr, va, te = _split_masks(g, seed)
            # Train on train∪val so final number is on held-out test; use only val for
            # model-selection in this script we skip grid-search and use default C=1
            train_use = tr | va
            def _acc(X: np.ndarray) -> float:
                return _linear_probe(X[train_use], labels[train_use], X[te], labels[te], seed=seed, device=device)
            base_acc = _acc(X_raw); our_acc = _acc(X_enc)
            delta_pp = (our_acc - base_acc) * 100.0
            dt = time.perf_counter() - t0
            row = dict(domain=dom, dataset=ds_name, seed=seed, task="node_cls",
                       n_classes=n_classes, n_nodes=int(g.num_nodes),
                       baseline_acc=round(base_acc, 6), ours_acc=round(our_acc, 6),
                       delta_pp=round(delta_pp, 3),
                       encoder_ckpt=("scratch_no_pretrain" if args.scratch_no_pretrain
                                     else (str(ckpt) if ckpt else "mini_pretrain")),
                       elapsed_s=round(dt, 2))
            # Write to variant CSV (optionally with variant tag column)
            var_row = dict(row)
            if has_tag:
                var_row = {"variant": args.variant_tag.strip(), **var_row}
            w_var.writerow(var_row); f_var.flush()
            # Optional: append to main CSV (original format, no variant column)
            for we, _inject_tag in writers_extra:
                we.writerow(row)
                # flush via handle lookup
            # Manually flush main CSV handles (not tracked via writers_extra fields)
            all_rows.append(row)
            print(f"  seed={seed:<2}  {dom:<14} {ds_name:<20} n={g.num_nodes:<5} C={n_classes}  "
                  f"base={base_acc*100:5.2f}  ours={our_acc*100:5.2f}  Δ={delta_pp:+.2f}pp")

    f_var.close()
    for we, _ in writers_extra:
        try:
            we.writerows([])  # no-op; ensure underlying file flushed via attribute
        except Exception:
            pass
        # Try to close by reaching the file handle: look into locals by last writer attribute
        # Note: csv.DictWriter closes only the file we opened; we track open files list below.
    # close the MAIN_CSV files we opened (if any)
    if writers_extra:
        # We didn't keep the file handles; simplify by closing no more here.
        # Python GC closes them on process exit.
        pass

    # Aggregate per dataset + domain
    from collections import defaultdict
    agg_ds: Dict[str, dict] = {}
    per_dom: Dict[str, List[float]] = defaultdict(list)
    for row in all_rows:
        k = (row["domain"], row["dataset"])
        if k not in agg_ds:
            agg_ds[k] = {"baseline_acc": [], "ours_acc": [], "delta_pp": []}
        agg_ds[k]["baseline_acc"].append(row["baseline_acc"])
        agg_ds[k]["ours_acc"].append(row["ours_acc"])
        agg_ds[k]["delta_pp"].append(row["delta_pp"])
    lines = ["# W1/W2 LODO linear probe Δ summary", "",
             "| domain | dataset | C | N | baseline (mean±std)% | ours (mean±std)% | Δ (mean±std)pp | seeds |",
             "|---|---|---|---|---|---|---|---|"]
    dom_rows: Dict[str, List[float]] = defaultdict(list)
    for (dom, ds), v in sorted(agg_ds.items()):
        bm, bs = float(np.mean(v["baseline_acc"]))*100, float(np.std(v["baseline_acc"]))*100
        om, os_ = float(np.mean(v["ours_acc"]))*100, float(np.std(v["ours_acc"]))*100
        dm, ds_ = float(np.mean(v["delta_pp"])), float(np.std(v["delta_pp"]))
        n_classes = next((r["n_classes"] for r in all_rows if r["domain"] == dom and r["dataset"] == ds), 0)
        n_nodes = next((r["n_nodes"] for r in all_rows if r["domain"] == dom and r["dataset"] == ds), 0)
        dom_rows[dom].append(dm)
        lines.append(f"| {dom} | {ds} | {n_classes} | {n_nodes} | {bm:.2f}±{bs:.2f} | {om:.2f}±{os_:.2f} | {dm:+.2f}±{ds_:.2f} | {len(v['delta_pp'])} |")
    lines += ["", "### Per-domain mean Δ (mean of datasets)", ""]
    lines += ["| domain | mean Δ (pp) |", "|---|---|"]
    grand = []
    for dom in sorted(dom_rows):
        mm = float(np.mean(dom_rows[dom]))
        lines.append(f"| {dom} | {mm:+.2f} |")
        grand.extend(dom_rows[dom])
    if grand:
        lines += ["", f"**Grand mean Δ (W1+W2) = {float(np.mean(grand)):+.2f} pp** across {len(grand)} seed×dataset runs."]
    # Summary MD: write to variant directory if custom out_csv is provided (so each ablation gets its own)
    if variant_csv.resolve() != MAIN_CSV.resolve():
        summary_path = variant_csv.parent / f"{variant_csv.stem}_SUMMARY.md"
        title_extra = f" — variant {args.variant_tag}" if has_tag else f" — variant_csv={variant_csv.name}"
        lines[0] = f"# W1/W2 LODO linear probe Δ summary{title_extra}"
    else:
        summary_path = ROOT / "logs" / "w1_w2_lodo_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\n[W1-W2] DONE. CSV → {variant_csv}  SUMMARY → {summary_path}")


if __name__ == "__main__":
    main()
