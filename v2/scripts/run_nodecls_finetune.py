"""Node classification downstream evaluation with full finetune or frozen-head.

P0-2 / P1-3 shared runner:
  - protocol = full finetune (default) or frozen encoder + linear head
  - scratch control or pretrained ckpt
  - same dataset loading / masks as W1-W2 linear probe
  - writes per-run CSV + per-run curve CSV + markdown summary

Example:
  python v2/scripts/run_nodecls_finetune.py \
      --pretrain_ckpt outputs_v2/checkpoints/pretrain_best_v2.pt \
      --seeds 1,2,3 --device cuda:0 --out_csv outputs_v2/p0/finetune_full.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]

from v2.scripts.run_w1w2_lodo_linearprobe import (  # reuse exact LODO dataset protocol
    _collect_nodecls,
    _linear_probe,
    _load_encoder,
    _load_yaml,
    _set_seed,
    _split_masks,
)
from v2.utils.dhg_datasets import load_domain_graphs


def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def _evaluate_raw_feature_baseline(
    x_raw: np.ndarray,
    labels: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    seed: int,
) -> float:
    train_use = tr | va
    return _linear_probe(x_raw[train_use], labels[train_use], x_raw[te], labels[te], seed=seed)


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "cuda oom" in msg)


def _clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _clone_state_dict_to_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _train_one_phase(enc, clf, x, H, tr_mask, va_mask, te_mask, labels, opt, epochs,
                     patience, device, autocast_ctx, scaler, clip_enc):
    """Run one training phase; return best (by val) state dicts + test/val/epoch + curve."""
    best_val = -1.0
    best_epoch = -1
    best_test = 0.0
    best_state = None
    bad = 0
    curve_rows: List[dict] = []
    for ep in range(epochs):
        clf.train()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            node_t, _, _, _, _ = enc(x, H)
            logits = clf(node_t)
            loss = F.cross_entropy(logits[tr_mask], labels[tr_mask])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        if clip_enc:
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        enc.eval()
        clf.eval()
        with torch.no_grad():
            with autocast_ctx():
                node_t, _, _, _, _ = enc(x, H)
                logits = clf(node_t)
                tr_acc = _acc(logits[tr_mask], labels[tr_mask])
                va_acc = _acc(logits[va_mask], labels[va_mask])
                te_acc = _acc(logits[te_mask], labels[te_mask])
        print(f"  [ep{ep + 1:02d}] loss={float(loss.item()):.4f} tr={tr_acc:.4f} va={va_acc:.4f} te={te_acc:.4f}")
        curve_rows.append({
            "epoch": ep + 1,
            "train_acc": round(tr_acc, 6),
            "val_acc": round(va_acc, 6),
            "test_acc": round(te_acc, 6),
            "loss": round(float(loss.item()), 6),
        })
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = ep + 1
            best_test = te_acc
            best_state = {
                "clf": _clone_state_dict_to_cpu(clf),
                "enc": _clone_state_dict_to_cpu(enc) if any(p.requires_grad for p in enc.parameters()) else None,
            }
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"  early stop at ep{ep + 1}")
            break
    return best_state, best_test, best_val, best_epoch, curve_rows


def _init_head_via_lbfgs(enc, x, H, train_mask, labels, device, seed=0):
    """Faithful re-implementation of the linear-probe head from
    run_w1w2_lodo_linearprobe._linear_probe (StandardScaler + LBFGS + L2, C=1.0).

    Trains a linear head on the *frozen* encoder's node embeddings so that a frozen
    run reproduces the ~56.8% probe number, then returns (mu, sd, W, b) so the
    standardization can be *folded* into an nn.Linear that consumes raw embeddings
    (required for the subsequent fine-tune phase).
    """
    with torch.no_grad():
        node_t, _, _, _, _ = enc(x, H)
        emb = node_t.detach().cpu().numpy().astype(np.float32)
    train_idx = np.where(train_mask.detach().cpu().numpy())[0]
    X_tr = emb[train_idx]
    y_tr = labels.detach().cpu().numpy()[train_idx].astype(np.int64)
    mu = X_tr.mean(0, keepdims=True).astype(np.float32)
    sd = X_tr.std(0, keepdims=True).astype(np.float32) + 1e-8
    Xz = (X_tr - mu) / sd
    n_classes = int(y_tr.max() + 1)
    Xz_t = torch.from_numpy(Xz).float()
    y_t = torch.from_numpy(y_tr).long()
    model = torch.nn.Linear(Xz_t.shape[1], n_classes)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500,
                                  tolerance_grad=1e-4, tolerance_change=1e-5,
                                  line_search_fn="strong_wolfe")
    criterion = torch.nn.CrossEntropyLoss()
    l2_lambda = 0.5 / Xz_t.shape[0]  # sklearn C=1.0 objective

    def closure():
        optimizer.zero_grad()
        logits = model(Xz_t)
        loss = criterion(logits, y_t)
        reg = sum((p ** 2).sum() for p in model.parameters())
        total = loss + l2_lambda * reg
        total.backward()
        return total

    try:
        model.train()
        optimizer.step(closure)
    except Exception as e:
        print(f"    [lft-head] LBFGS failed ({e}); falling back to zeros init")
    with torch.no_grad():
        W = model.weight.detach().cpu().numpy().astype(np.float32)  # [C, H]
        b = model.bias.detach().cpu().numpy().astype(np.float32)     # [C]
    mu_f = mu.astype(np.float32).reshape(-1)   # [H]
    sd_f = sd.astype(np.float32).reshape(-1)   # [H]
    return mu_f, sd_f, W, b


def _run_single_dataset(
    *,
    g,
    dom: str,
    ds_name: str,
    seed: int,
    args,
    cfg,
    graphs_by_domain,
    in_dim: int,
    hidden_dim: int,
    ckpt,
    use_hor_arg: bool,
    pe_dim_arg: int,
    hca_topk_arg: int,
    variant_tag: str,
    protocol: str,
    device: torch.device,
) -> tuple[dict, List[dict]]:
    t0 = time.perf_counter()
    labels_np = g.node_labels.cpu().numpy()
    labels = g.node_labels.to(device).long()
    n_classes = int(labels_np.max() + 1)
    tr, va, te = _split_masks(g, seed)
    tr_mask = torch.from_numpy(tr).to(device)
    va_mask = torch.from_numpy(va).to(device)
    te_mask = torch.from_numpy(te).to(device)

    pretrain_domains = [d for d in graphs_by_domain.keys() if d != dom]
    enc = _load_encoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        ckpt_path=ckpt,
        pretrain_domains=pretrain_domains,
        all_graphs=graphs_by_domain,
        device=device,
        scratch_no_pretrain=args.scratch_no_pretrain,
        use_hor=use_hor_arg,
        ablate_cca_card=args.ablate_cca_card,
        ablate_cca_film=args.ablate_cca_film,
        ablate_cca_tau=args.ablate_cca_tau,
        ablate_hca_bias=args.ablate_hca_bias,
        ablate_hca_full=args.ablate_hca_full,
        pe_dim=pe_dim_arg,
        hca_topk=hca_topk_arg,
    )
    # Mitigate overfitting on small graphs: when fine-tuning (not freeze_encoder),
    # freeze all but the last K encoder layers; only they + the head receive gradients.
    # Mitigate overfitting on small graphs: when fine-tuning (not freeze_encoder),
    # freeze all but the last K encoder layers; only they + the head receive gradients.
    # NOTE: stacked layers live in enc.cca_layers (not enc.layers).
    if not args.freeze_encoder and args.unfreeze_last_k > 0 and hasattr(enc, "cca_layers"):
        total = len(enc.cca_layers)
        k = min(args.unfreeze_last_k, total)
        freeze_n = total - k
        for i in range(freeze_n):
            for p in enc.cca_layers[i].parameters():
                p.requires_grad_(False)
        print(f"[finetune] backbone frozen: enc.cca_layers[0:{freeze_n}]; "
              f"trainable: enc.cca_layers[{freeze_n}:{total}] + head")

    clf = nn.Linear(hidden_dim, n_classes).to(device)

    x = g.x.to(device).float()
    H = g.incidence_matrix().to_sparse_coo().coalesce().to(device)

    amp_enabled = bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext

    # Fine-tune training mask. Default trva matches the linear-probe protocol
    # (probe trains on tr|va); --train_on tr restricts to tr only.
    train_mask = tr_mask if args.train_on == "tr" else (tr_mask | va_mask)

    if args.freeze_encoder:
        # Frozen encoder + linear head == faithful linear probe (sklearn LBFGS+L2).
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        mu, sd, W, b = _init_head_via_lbfgs(enc, x, H, train_mask, labels, device, seed)
        Wf = W / sd
        bf = b - Wf @ mu
        with torch.no_grad():
            clf.weight.copy_(torch.from_numpy(Wf))
            clf.bias.copy_(torch.from_numpy(bf))
            node_t, _, _, _, _ = enc(x, H)
            best_test = float((clf(node_t).argmax(dim=-1) == labels).float().mean().item())
        best_state = None
        best_val = float("nan")
        best_epoch = -1
        curve_rows = []
    elif args.lft:
        # LP-FT: (1) linear-probe init of head with encoder frozen (on train_mask);
        #        (2) unfreeze last K layers, load probe head, light fine-tune.
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        mu, sd, W, b = _init_head_via_lbfgs(enc, x, H, train_mask, labels, device, seed)
        Wf = W / sd
        bf = b - Wf @ mu
        with torch.no_grad():
            clf.weight.copy_(torch.from_numpy(Wf))
            clf.bias.copy_(torch.from_numpy(bf))
        # Phase 2: fine-tune last K encoder layers + folded head.
        if args.unfreeze_last_k > 0 and hasattr(enc, "cca_layers"):
            total = len(enc.cca_layers)
            k = min(args.unfreeze_last_k, total)
            freeze_n = total - k
            for i in range(freeze_n):
                for p in enc.cca_layers[i].parameters():
                    p.requires_grad_(False)
            print(f"[lft] phase2: fine-tune enc.cca_layers[{freeze_n}:{total}] + head "
                  f"(encoder lr={args.lr_encoder:.0e})")
        else:
            print("[lft] phase2: fine-tune full encoder + head")
        opt_f = torch.optim.AdamW(
            [{"params": clf.parameters(), "lr": args.lr_head},
             {"params": enc.parameters(), "lr": args.lr_encoder}],
            weight_decay=args.weight_decay)
        best_state, best_test, best_val, best_epoch, curve_rows = _train_one_phase(
            enc, clf, x, H, train_mask, va_mask, te_mask, labels, opt_f,
            args.ft_epochs, args.patience, device, autocast_ctx, scaler, clip_enc=True)
    else:
        # Full fine-tune (known to overfit on small graphs; kept for ablation).
        enc.train()
        opt = torch.optim.AdamW(
            [{"params": clf.parameters(), "lr": args.lr_head},
             {"params": enc.parameters(), "lr": args.lr_encoder}],
            weight_decay=args.weight_decay)
        best_state, best_test, best_val, best_epoch, curve_rows = _train_one_phase(
            enc, clf, x, H, train_mask, va_mask, te_mask, labels, opt,
            args.epochs, args.patience, device, autocast_ctx, scaler, clip_enc=True)

    if best_state is not None:
        clf.load_state_dict(best_state["clf"])
        if best_state["enc"] is not None:
            enc.load_state_dict(best_state["enc"])

    x_raw = g.x.cpu().numpy().astype(np.float32)
    raw_acc = _evaluate_raw_feature_baseline(x_raw, labels_np, tr, va, te, seed)
    dt = time.perf_counter() - t0

    row = {
        "variant": variant_tag,
        "protocol": protocol,
        "domain": dom,
        "dataset": ds_name,
        "seed": seed,
        "n_classes": n_classes,
        "n_nodes": int(g.num_nodes),
        "raw_baseline_acc": round(raw_acc, 6),
        "test_acc": round(best_test, 6),
        "delta_vs_raw_pp": round((best_test - raw_acc) * 100.0, 3),
        "best_val_acc": round(best_val, 6),
        "best_epoch": best_epoch,
        "encoder_ckpt": ("scratch_no_pretrain" if args.scratch_no_pretrain else str(ckpt)),
        "elapsed_s": round(dt, 2),
    }

    for _nm in ("opt", "opt_p", "opt_f"):
        if _nm in locals():
            del locals()[_nm]
    del clf, enc, x, H, labels, tr_mask, va_mask, te_mask, best_state
    gc.collect()
    if device.type == "cuda":
        _clear_cuda_memory()
    return row, curve_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="v2/configs/pretrain_v2.yaml")
    ap.add_argument("--pretrain_ckpt", type=str, default="outputs_v2/checkpoints/pretrain_best_v2.pt")
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--domains", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr_encoder", type=float, default=5e-5)
    ap.add_argument("--lr_head", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true",
                    help="Enable AMP autocast. Disabled by default: the encoder relies on "
                         "sparse matmul + torch.linalg.solve (CCA cardinality), which is "
                         "numerically unstable under float16; keep float32 to match pretraining.")
    ap.add_argument("--train_on", type=str, default="trva", choices=["tr", "trva"],
                    help="Train classifier on tr-only or tr|va (default trva, matching linear probe).")
    ap.add_argument("--unfreeze_last_k", type=int, default=1,
                    help="Fine-tuning: train only the last K encoder layers + head; earlier "
                         "layers stay frozen to avoid overfitting small graphs. Ignored if --freeze_encoder.")
    ap.add_argument("--lft", action="store_true", default=True,
                    help="LP-FT (Linear Probe then Fine-Tune): initialize head by linear probe on "
                         "frozen encoder, then lightly fine-tune last K layers. Avoids overfitting "
                         "small graphs. Disable with --no_lft.")
    ap.add_argument("--no_lft", dest="lft", action="store_false",
                    help="Disable LP-FT; do plain (overfitting-prone) full fine-tune.")
    ap.add_argument("--probe_epochs", type=int, default=25,
                    help="LP-FT phase-1 linear-probe epochs (head init, encoder frozen).")
    ap.add_argument("--ft_epochs", type=int, default=25,
                    help="LP-FT phase-2 fine-tune epochs (last K layers + head).")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Train only the classification head; unlike linear probe, this still uses SGD on the head.")
    ap.add_argument("--scratch_no_pretrain", action="store_true",
                    help="Use random-init encoder directly; skip ckpt load and mini-pretrain fallback.")
    ap.add_argument("--use_hor", type=str, default="default", choices=["default", "true", "false"])
    ap.add_argument("--ablate_cca_card", action="store_true")
    ap.add_argument("--ablate_cca_film", action="store_true")
    ap.add_argument("--ablate_cca_tau", action="store_true")
    ap.add_argument("--ablate_hca_bias", action="store_true")
    ap.add_argument("--ablate_hca_full", action="store_true")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--variant_tag", type=str, default="")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = _load_yaml(cfg_path)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    graphs_by_domain = load_domain_graphs(cfg, seed=cfg["training"]["seed"], require_node_splits=True)
    nodecls = _collect_nodecls(graphs_by_domain)

    target_domains: List[str]
    if args.domains.strip():
        target_domains = [d for d in args.domains.split(",") if d.strip()]
    else:
        target_domains = list(graphs_by_domain.keys())

    if args.use_hor == "default":
        use_hor_arg = bool(cfg.get("model", {}).get("use_hor", True))
    else:
        use_hor_arg = (args.use_hor == "true")
    pe_dim_arg = int(cfg.get("model", {}).get("pe_dim", 32))
    hca_topk_arg = int(cfg.get("model", {}).get("hca_topk", 16))

    out_csv = Path(args.out_csv) if args.out_csv else (ROOT / "outputs_v2" / "nodecls_finetune.csv")
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    curves_dir = out_csv.parent / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "variant", "protocol", "domain", "dataset", "seed", "n_classes", "n_nodes",
        "raw_baseline_acc", "test_acc", "delta_vs_raw_pp",
        "best_val_acc", "best_epoch", "encoder_ckpt", "elapsed_s",
    ]
    csv_exists = out_csv.is_file()
    f = open(out_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if not csv_exists:
        w.writeheader()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"[nodecls-finetune] device={device} freeze_encoder={args.freeze_encoder} "
          f"scratch_no_pretrain={args.scratch_no_pretrain} use_hor={use_hor_arg}")

    in_dim = cfg["model"]["input_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    ckpt = None if not args.pretrain_ckpt.strip() else (ROOT / args.pretrain_ckpt)
    all_rows: List[dict] = []
    variant_tag = args.variant_tag.strip() or (
        "scratch" if args.scratch_no_pretrain else (
            "full" if not any([args.ablate_cca_card, args.ablate_cca_film, args.ablate_cca_tau,
                               args.ablate_hca_bias, args.ablate_hca_full]) and use_hor_arg
            else "custom"
        )
    )
    protocol = "frozen_head" if args.freeze_encoder else "finetune"

    for seed in seeds:
        _set_seed(seed)
        for dom, ds_name, g in nodecls:
            if dom not in target_domains:
                continue
            try:
                row, curve_rows = _run_single_dataset(
                    g=g,
                    dom=dom,
                    ds_name=ds_name,
                    seed=seed,
                    args=args,
                    cfg=cfg,
                    graphs_by_domain=graphs_by_domain,
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    ckpt=ckpt,
                    use_hor_arg=use_hor_arg,
                    pe_dim_arg=pe_dim_arg,
                    hca_topk_arg=hca_topk_arg,
                    variant_tag=variant_tag,
                    protocol=protocol,
                    device=device,
                )
            except RuntimeError as exc:
                if device.type != "cuda" or not _is_cuda_oom(exc):
                    raise
                print(f"[nodecls-finetune][OOM] seed={seed} domain={dom} dataset={ds_name} on {device}; retry on cpu")
                _clear_cuda_memory()
                row, curve_rows = _run_single_dataset(
                    g=g,
                    dom=dom,
                    ds_name=ds_name,
                    seed=seed,
                    args=args,
                    cfg=cfg,
                    graphs_by_domain=graphs_by_domain,
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    ckpt=ckpt,
                    use_hor_arg=use_hor_arg,
                    pe_dim_arg=pe_dim_arg,
                    hca_topk_arg=hca_topk_arg,
                    variant_tag=variant_tag,
                    protocol=protocol,
                    device=torch.device("cpu"),
                )
            w.writerow(row)
            f.flush()
            all_rows.append(row)

            curve_path = curves_dir / f"{variant_tag}_{protocol}_{dom}_{ds_name}_seed{seed}.csv"
            with open(curve_path, "w", newline="") as cf:
                cw = csv.DictWriter(cf, fieldnames=list(curve_rows[0].keys()))
                cw.writeheader()
                cw.writerows(curve_rows)
            print(f"  seed={seed:<2} {protocol:<11} {dom:<14} {ds_name:<20} "
                  f"raw={float(row['raw_baseline_acc'])*100:5.2f} "
                  f"test={float(row['test_acc'])*100:5.2f} "
                  f"Δ={float(row['delta_vs_raw_pp']):+5.2f}pp "
                  f"best_ep={row['best_epoch']}")

    f.close()

    # Summary markdown
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_rows:
        agg[(r["variant"], r["protocol"], r["domain"], r["dataset"])].append(r)
    lines = ["# NodeCls finetune summary", "",
             "| variant | protocol | domain | dataset | raw baseline (mean±std)% | test acc (mean±std)% | Δ vs raw (mean±std)pp | best epoch (mean) |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    grand = defaultdict(list)
    for (variant, protocol, dom, ds), rows in sorted(agg.items()):
        rb = [float(r["raw_baseline_acc"]) * 100.0 for r in rows]
        ta = [float(r["test_acc"]) * 100.0 for r in rows]
        dd = [float(r["delta_vs_raw_pp"]) for r in rows]
        be = [float(r["best_epoch"]) for r in rows]
        grand[(variant, protocol)].extend(dd)
        lines.append(
            f"| {variant} | {protocol} | {dom} | {ds} | "
            f"{np.mean(rb):.2f}±{np.std(rb):.2f} | {np.mean(ta):.2f}±{np.std(ta):.2f} | "
            f"{np.mean(dd):+.2f}±{np.std(dd):.2f} | {np.mean(be):.1f} |"
        )
    lines += ["", "## Grand mean Δ by variant/protocol", "", "| variant | protocol | grand mean Δ (pp) |", "|---|---|---:|"]
    for (variant, protocol), vals in sorted(grand.items()):
        lines.append(f"| {variant} | {protocol} | {np.mean(vals):+.2f} ± {np.std(vals):.2f} |")
    summary_path = out_csv.parent / f"{out_csv.stem}_SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\n[nodecls-finetune] DONE. CSV → {out_csv}  SUMMARY → {summary_path}")


if __name__ == "__main__":
    main()
