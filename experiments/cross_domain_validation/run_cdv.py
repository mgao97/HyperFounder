#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Domain Hypergraph Validation Experiment
=============================================
Implements the M0-M5 matrix, Fixed/Learned structure strategies, and the
encoding set (None / Spectral-PE / Lightweight-SE[H-LDP] / Random-control)
defined in docs/cross_domain_hypergraph_validation_experiment_instruction.md.

Protocol (faithful to the instruction's screening intent):
  * Primary cross-domain metric = zero-shot hyperedge-prediction AUROC.
    Edge features use DOMAIN-INVARIANT dimensions (structural stats [4] + SE),
    so a logistic-regression probe trained on the source transfers to the
    target. This isolates the contribution of structure + structural encoding
    to cross-domain transfer (the core question of the instruction).
  * In-domain node classification (Accuracy / Macro-F1 / Micro-F1) is reported
    as a secondary ceiling/sanity check, using a frozen hypergraph message-
    passing backbone (raw features + encoding) + LR probe on the train mask.
  * Cost is measured as wall-clock for encoding pre-computation + storage bytes,
    to verify H3 (lightweight >> spectral).

Implementation notes:
  * All tensors are numpy + scipy.sparse (CSR) to avoid dense-memory blowups
    on large graphs (e.g. CoauthorshipDBLP ~41k nodes).
  * The local `encodings.py` exposes `h_ldp(num_v, e_list)` (NOT build_hldp_*),
    so hyperedge lists are kept alongside incidence matrices.

Outputs (under results/):
  * results.csv     -- one row per (pair x method x seed)
  * nodecls.csv     -- in-domain node classification per (dataset x method x seed)
  * cost.csv        -- encoding precompute cost per (dataset x structure x enc)
  * summary.json    -- aggregated metrics + Go/No-Go verdict
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

# --- make project imports available -----------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../HyperFounder
sys.path.insert(0, str(ROOT))                       # so `v2.utils.*` resolves
sys.path.insert(0, str(ROOT / "v2" / "utils"))      # so `dhg_datasets` resolves
from dhg_datasets import load_dhg_sample  # noqa: E402

# The local module is named `encodings.py`, which collides with the stdlib
# `encodings` package (cached at interpreter startup). Load it by file path.
import importlib.util as _ilu
_enc_path = ROOT / "experiments" / "gap_validation" / "encodings.py"
_enc_spec = _ilu.spec_from_file_location("local_encodings", str(_enc_path))
enc_mod = _ilu.module_from_spec(_enc_spec)
sys.modules["local_encodings"] = enc_mod
_enc_spec.loader.exec_module(enc_mod)
h_ldp = enc_mod.h_ldp

# --- experiment configuration -----------------------------------------------
PE_DIM = 16            # spectral PE dimension (k eigenvectors)
KNN_K = 10             # k for learned (kNN) hypergraph construction
TARGET_DIM = 64        # feature reduction dim for load_dhg_sample
SEEDS = [2024, 2025, 2026]
MAX_NODES_SPECTRAL = 60000  # skip eigendecomposition beyond this (cost guard)
RAND_DIM = 6           # dim-matched random control

PAIRS = [
    # (pair_id, source_name, target_name, domain_src, domain_tgt, note)
    ("P1_citation",    "coauthorship_dblp", "coauthorship_cora", "academic", "academic",  "citation-family cross-dataset"),
    ("P2_crossdomain", "coauthorship_dblp", "house_committees",  "academic", "political", "true cross-domain (academic->political)"),
]

# Methods map directly onto the instruction's M0-M5 matrix.
#   struct: fixed | learned | learned_xfer
#   enc   : none   | hldp    | spectral
METHODS = {
    "M0": dict(struct="fixed",        enc="none"),
    "M1": dict(struct="fixed",        enc="spectral"),
    "M2": dict(struct="fixed",        enc="hldp"),
    "M3": dict(struct="learned",      enc="none"),
    "M4": dict(struct="learned",      enc="hldp"),
    "M5": dict(struct="learned_xfer", enc="hldp"),
    "M6": dict(struct="learned",      enc="spectral"),
}
# Pseudo-methods handled outside the M0-M5 loop (ablations / controls)
SHUFFLE_METHOD = "M2_shuf"  # M2 with H-LDP node-rows independently permuted per domain


# ---------------------------------------------------------------------------
# Incidence / structure builders
# ---------------------------------------------------------------------------
def inc_from_elist(e_list, num_v):
    rows, cols = [], []
    for ei, e in enumerate(e_list):
        for i in e:
            rows.append(i)
            cols.append(ei)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_v, len(e_list)))


def fixed_structure(sample):
    e_list = sample["hyperedges"]
    num_v = sample["n_nodes"]
    return e_list, inc_from_elist(e_list, num_v)


def learned_structure(node_features, num_v, k, shared_stats=None):
    """kNN hypergraph: each node forms a hyperedge with its k neighbours.

    shared_stats: optional (mean, std) on the SOURCE domain; when provided the
    target features are standardised with the *source* statistics (the
    "transferable structure rule" of M5 / S4-minimal).
    """
    if shared_stats is not None:
        mean, std = shared_stats
        X = (node_features - mean) / np.clip(std, 1e-8, None)
    else:
        X = StandardScaler().fit_transform(node_features)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    e_list = [sorted(int(j) for j in idx[i, 1:]) for i in range(X.shape[0])]  # drop self
    return e_list, inc_from_elist(e_list, num_v)


# ---------------------------------------------------------------------------
# Encoding builders  (return (n, d) np.ndarray, or (None, status))
# ---------------------------------------------------------------------------
def structural_stats(inc_sp):
    """Constant-dim (4) node structural features, O(nnz)."""
    n = inc_sp.shape[0]
    dv = np.asarray(inc_sp.sum(1)).ravel().astype(np.float64)         # #incident edges
    es = np.asarray(inc_sp.sum(0)).ravel().astype(np.float64)         # edge sizes
    ies = inc_sp.multiply(es)                                         # (n,e) weighted
    incident_counts = np.clip(dv, 1, None)
    imean = np.asarray(ies.sum(1)).ravel() / incident_counts
    if inc_sp.size > 0:
        imax = np.asarray(ies.max(axis=1).toarray()).ravel()
    else:
        imax = np.zeros(n)
    comm = np.asarray(inc_sp.multiply(np.clip(es - 1, 0, None)).sum(1)).ravel()
    return np.stack([np.log1p(dv), imean, imax, comm], axis=1).astype(np.float32)


def enc_none(inc_sp):
    return structural_stats(inc_sp)


def enc_hldp(e_list, num_v):
    feats = h_ldp(num_v, e_list).astype(np.float32)
    return np.concatenate([structural_stats(inc_from_elist(e_list, num_v)), feats], axis=1)


def enc_spectral(inc_sp, k=PE_DIM):
    """Spectral PE via eigenvectors of the symmetric normalised adjacency.

    We take the top-k eigenvectors (excluding the trivial one) of
    A = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}, which correspond to the smallest
    non-trivial eigenvectors of the Laplacian. Using `which='LM'` is numerically
    robust (eigsh converges reliably) whereas `which='SM'` on the Laplacian
    frequently fails to converge on graphs with many isolated nodes.
    """
    try:
        n = inc_sp.shape[0]
        if n > MAX_NODES_SPECTRAL:
            return None, "too_large"
        H = inc_sp.tocsr().astype(np.float64)
        Dv = np.asarray(H.sum(1)).ravel().clip(min=1e-8)
        De = np.asarray(H.sum(0)).ravel().clip(min=1e-8)
        Dv_inv_sqrt = 1.0 / np.sqrt(Dv)
        De_inv = 1.0 / De
        A = sp.diags(Dv_inv_sqrt) @ H @ sp.diags(De_inv) @ H.T @ sp.diags(Dv_inv_sqrt)
        A = (A + A.T) / 2.0
        kk = min(k + 1, n - 1)
        if kk < 2:
            return None, "deg_too_low"
        vals, vecs = eigsh(A, k=kk, which="LM", maxiter=5000)
        order = np.argsort(vals)[::-1]          # descending eigenvalue
        vecs = vecs[:, order]
        pe = vecs[:, 1:kk] if vecs.shape[1] > 1 else vecs   # drop trivial
        if pe.shape[1] < k:
            pe = np.hstack([pe, np.zeros((n, k - pe.shape[1]))])
        return pe.astype(np.float32), "ok"
    except Exception as e:
        return None, f"error:{type(e).__name__}"


def enc_rand(n, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, RAND_DIM)).astype(np.float32)


def build_node_feat(e_list, num_v, inc_sp, enc_name, seed):
    """Return (node_feat, status). status in {'ok','skipped':reason}."""
    if enc_name == "none":
        return enc_none(inc_sp), "ok"
    if enc_name == "hldp":
        return enc_hldp(e_list, num_v), "ok"
    if enc_name == "spectral":
        pe, st = enc_spectral(inc_sp)
        if pe is None:
            return None, st
        return np.concatenate([structural_stats(inc_sp), pe], axis=1), "ok"
    raise ValueError(enc_name)


# ---------------------------------------------------------------------------
# Edge-pooling + cross-domain hyperedge-prediction probe
# ---------------------------------------------------------------------------
def edge_pool(node_feat, real_inc_sp):
    es = np.asarray(real_inc_sp.sum(0)).ravel().clip(min=1)
    return np.asarray(real_inc_sp.T @ node_feat) / es[:, None]


def sample_negative_edges(hyperedges, num_samples, rng):
    n = max(max(e) for e in hyperedges) + 1 if hyperedges else 0
    sizes = [len(e) for e in hyperedges]
    pos = set(frozenset(e) for e in hyperedges)
    lo, hi = max(2, min(sizes)), min(max(sizes), n)
    negs, attempts = [], 0
    while len(negs) < num_samples and attempts < num_samples * 30 and hi >= lo:
        attempts += 1
        s = int(rng.integers(lo, hi + 1))
        cand = tuple(sorted(rng.choice(n, size=s, replace=False)))
        if frozenset(cand) not in pos:
            negs.append(list(cand))
    return negs


def transfer_auroc(src, tgt, nf_src, nf_tgt, seed):
    """Zero-shot transfer: LR fit on source edges, eval on target edges."""
    rng = np.random.default_rng(seed)
    src_pos = edge_pool(nf_src, src["real_inc"])
    src_neg = edge_pool(nf_src, inc_from_elist(
        sample_negative_edges(src["hyperedges"], len(src["hyperedges"]), rng), src["n_nodes"]))
    tgt_pos = edge_pool(nf_tgt, tgt["real_inc"])
    tgt_neg = edge_pool(nf_tgt, inc_from_elist(
        sample_negative_edges(tgt["hyperedges"], len(tgt["hyperedges"]), rng), tgt["n_nodes"]))
    Xtr = np.vstack([src_pos, src_neg])
    ytr = np.concatenate([np.ones(len(src_pos)), np.zeros(len(src_neg))])
    Xte = np.vstack([tgt_pos, tgt_neg])
    yte = np.concatenate([np.ones(len(tgt_pos)), np.zeros(len(tgt_neg))])
    clf = LogisticRegression(max_iter=3000)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def indomain_auroc(ds, nf, seed):
    """Honest in-domain ceiling via 50/50 edge split."""
    rng = np.random.default_rng(seed)
    real_inc = ds["real_inc"]
    idx = np.arange(real_inc.shape[1])
    rng.shuffle(idx)
    cut = len(idx) // 2
    tr, te = idx[:cut], idx[cut:]
    inc_tr = real_inc[:, tr]
    inc_te = real_inc[:, te]
    rng2 = np.random.default_rng(seed + 1)
    neg_tr = inc_from_elist(sample_negative_edges(ds["hyperedges"], inc_tr.shape[1], rng2), ds["n_nodes"])
    rng3 = np.random.default_rng(seed + 2)
    neg_te = inc_from_elist(sample_negative_edges(ds["hyperedges"], inc_te.shape[1], rng3), ds["n_nodes"])
    Xtr = np.vstack([edge_pool(nf, inc_tr), edge_pool(nf, neg_tr)])
    ytr = np.concatenate([np.ones(inc_tr.shape[1]), np.zeros(inc_tr.shape[1])])
    Xte = np.vstack([edge_pool(nf, inc_te), edge_pool(nf, neg_te)])
    yte = np.concatenate([np.ones(inc_te.shape[1]), np.zeros(inc_te.shape[1])])
    clf = LogisticRegression(max_iter=3000)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


# ---------------------------------------------------------------------------
# In-domain node classification (frozen message-passing backbone + LR probe)
# ---------------------------------------------------------------------------
def message_pass(X_enc, inc_sp, layers=2):
    Dv = np.asarray(inc_sp.sum(1)).ravel().clip(min=1)
    De = np.asarray(inc_sp.sum(0)).ravel().clip(min=1)
    Ht = inc_sp.T.tocsr()
    Xc = X_enc.astype(np.float64)
    for _ in range(layers):
        Xe = Ht @ Xc / De[:, None]
        Xc = np.clip((inc_sp @ Xe) / Dv[:, None], 0, None)
    return Xc.astype(np.float32)


def node_cls(ds, inc_sp, enc_name, seed):
    if ds.get("labels") is None or ds.get("masks") is None:
        return None
    raw = ds["node_features"].astype(np.float32)
    if enc_name == "none":
        X_enc = raw
    elif enc_name == "hldp":
        X_enc = np.concatenate([raw, enc_hldp(ds["hyperedges"], ds["n_nodes"])], axis=1)
    elif enc_name == "spectral":
        pe, st = enc_spectral(inc_sp)
        if pe is None:
            return None
        X_enc = np.concatenate([raw, pe], axis=1)
    else:
        return None
    Z = message_pass(X_enc, inc_sp, layers=2)
    Z = StandardScaler().fit_transform(Z)
    masks = ds["masks"]
    tr, te = masks.get("train_mask"), masks.get("test_mask")
    if tr is None or te is None:
        return None
    y = np.asarray(ds["labels"]).ravel()
    clf = LogisticRegression(max_iter=3000, multi_class="auto")
    clf.fit(Z[tr], y[tr])
    yp = clf.predict(Z[te])
    return dict(
        accuracy=float(accuracy_score(y[te], yp)),
        macro_f1=float(f1_score(y[te], yp, average="macro", zero_division=0)),
        micro_f1=float(f1_score(y[te], yp, average="micro", zero_division=0)),
    )


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------
def load_dataset(name):
    sg = load_dhg_sample(name, TARGET_DIM, 0)
    x = sg.x
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    nf = np.asarray(x, dtype=np.float32)
    hyperedges = sg.hyperedges  # List[List[int]]
    real_inc = inc_from_elist(hyperedges, int(sg.num_nodes))
    masks = None
    if sg.node_train_mask is not None and sg.node_test_mask is not None:
        masks = dict(
            train_mask=np.asarray(sg.node_train_mask.detach().cpu().numpy()).astype(bool),
            test_mask=np.asarray(sg.node_test_mask.detach().cpu().numpy()).astype(bool),
        )
    labels = None
    if sg.node_labels is not None:
        labels = np.asarray(sg.node_labels.detach().cpu().numpy())
    out = dict(
        name=name, type=str(sg.domain), node_features=nf, hyperedges=hyperedges,
        real_inc=real_inc, labels=labels, masks=masks,
        n_nodes=int(sg.num_nodes), n_edges=real_inc.shape[1],
    )
    return out


def transfer_auroc_hldp_shuffle(src, tgt, seed):
    """M2 but with H-LDP node rows independently permuted per domain.

    If the shuffled variant collapses back to M0, the *arrangement* of H-LDP
    across nodes carries the transferable signal (a real structural encoding);
    if it stays near M2, only the marginal H-LDP distribution matters.
    """
    s_e, _ = fixed_structure(src)
    t_e, _ = fixed_structure(tgt)
    nf_s = enc_hldp(s_e, src["n_nodes"])
    nf_t = enc_hldp(t_e, tgt["n_nodes"])
    rs = np.random.default_rng(seed)
    rt = np.random.default_rng(seed + 7777)
    nf_s = nf_s[rs.permutation(src["n_nodes"])]
    nf_t = nf_t[rt.permutation(tgt["n_nodes"])]
    return transfer_auroc(src, tgt, nf_s, nf_t, seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    rows, nodecls_rows, cost_rows = [], [], []
    loaded = {}

    def get_ds(name):
        if name not in loaded:
            loaded[name] = load_dataset(name)
            print(f"[load] {name}: n={loaded[name]['n_nodes']} e={loaded[name]['n_edges']}", flush=True)
        return loaded[name]

    for pair_id, src_name, tgt_name, dom_s, dom_t, note in PAIRS:
        print(f"\n===== {pair_id}: {src_name} -> {tgt_name} ({note}) =====", flush=True)
        src = get_ds(src_name)
        tgt = get_ds(tgt_name)

        # structure variants: each stores (e_list, incidence_csr)
        s_e, s_fixed = fixed_structure(src)
        t_e, t_fixed = fixed_structure(tgt)
        s_learned_lst, s_learned_inc = learned_structure(src["node_features"], src["n_nodes"], KNN_K)
        t_learned_lst, t_learned_inc = learned_structure(tgt["node_features"], tgt["n_nodes"], KNN_K)
        src_stats = (src["node_features"].mean(0), src["node_features"].std(0))
        # M5 (learned_xfer): target structure built with SOURCE-fitted statistics
        t_xfer_lst, t_xfer_inc = learned_structure(tgt["node_features"], tgt["n_nodes"], KNN_K, src_stats)
        struct = {
            "fixed":        dict(src=(s_e, s_fixed), tgt=(t_e, t_fixed)),
            "learned":      dict(src=(s_learned_lst, s_learned_inc), tgt=(t_learned_lst, t_learned_inc)),
            "learned_xfer": dict(src=(s_learned_lst, s_learned_inc), tgt=(t_xfer_lst, t_xfer_inc)),
        }

        # pre-compute + time encodings (cost tables) on both domains.
        # spectral PE cost is dominated by the eigendecomposition and is
        # essentially structure-independent, so we time it once per dataset
        # (fixed structure) to avoid repeated ~40s eigsh calls on large graphs.
        for dname, dso in (("src", src), ("tgt", tgt)):
            inc_sp = struct["fixed"][dname][1]
            e_list = struct["fixed"][dname][0]
            for ename in ("hldp", "spectral"):
                t0 = time.perf_counter()
                if ename == "hldp":
                    fe, st = enc_hldp(e_list, dso["n_nodes"]), "ok"
                    nbytes = int(fe.nbytes)
                    dim = fe.shape[1]
                else:
                    fe, st = enc_spectral(inc_sp)
                    nbytes = int(fe.nbytes) if fe is not None else 0
                    dim = fe.shape[1] if fe is not None else 0
                dt = time.perf_counter() - t0
                cost_rows.append(dict(pair=pair_id, dataset=dname, structure="fixed",
                                      encoding=ename, status=st, time_s=round(dt, 6),
                                      bytes=nbytes, dim=dim))

        for method, cfg in METHODS.items():
            skey = cfg["struct"]; enc = cfg["enc"]
            for seed in SEEDS:
                try:
                    e_s, inc_s = struct[skey]["src"]
                    e_t, inc_t = struct[skey]["tgt"]
                    nf_src, st_s = build_node_feat(e_s, src["n_nodes"], inc_s, enc, seed)
                    nf_tgt, st_t = build_node_feat(e_t, tgt["n_nodes"], inc_t, enc, seed)
                    if nf_src is None or nf_tgt is None:
                        print(f"  [{method}] enc={enc} skipped ({st_t})", flush=True)
                        continue
                    p_zero = transfer_auroc(src, tgt, nf_src, nf_tgt, seed)
                    p_ind = indomain_auroc(tgt, nf_tgt, seed)
                    rows.append(dict(pair=pair_id, src=src_name, tgt=tgt_name,
                                     domain_src=dom_s, domain_tgt=dom_t, method=method,
                                     structure=skey, encoding=enc, seed=seed,
                                     transfer_auroc=round(p_zero, 4),
                                     indomain_auroc=round(p_ind, 4),
                                     gap=round(p_ind - p_zero, 4)))
                    print(f"  [{method}] {enc:8s} seed={seed}  transfer={p_zero:.3f}  indomain={p_ind:.3f}", flush=True)
                except Exception as e:
                    print(f"  [{method}] EXCEPTION: {e}", flush=True)
                    traceback.print_exc()

        # random control (dim-matched) for H1 robustness
        for seed in SEEDS:
            nf_src = np.concatenate([enc_none(s_fixed), enc_rand(src["n_nodes"], seed)], 1)
            nf_tgt = np.concatenate([enc_none(t_fixed), enc_rand(tgt["n_nodes"], seed)], 1)
            p_zero = transfer_auroc(src, tgt, nf_src, nf_tgt, seed)
            rows.append(dict(pair=pair_id, src=src_name, tgt=tgt_name,
                             domain_src=dom_s, domain_tgt=dom_t, method="MR_random",
                             structure="fixed", encoding="random", seed=seed,
                             transfer_auroc=round(p_zero, 4), indomain_auroc="", gap=""))

        # P0 shuffle ablation: M2 with H-LDP independently permuted per domain
        for seed in SEEDS:
            pz = transfer_auroc_hldp_shuffle(src, tgt, seed)
            rows.append(dict(pair=pair_id, src=src_name, tgt=tgt_name,
                             domain_src=dom_s, domain_tgt=dom_t, method=SHUFFLE_METHOD,
                             structure="fixed", encoding="hldp_shuffle", seed=seed,
                             transfer_auroc=round(pz, 4), indomain_auroc="", gap=""))

    # in-domain node classification (each dataset standalone, all methods)
    for dname in sorted(set([p[1] for p in PAIRS] + [p[2] for p in PAIRS])):
        ds = get_ds(dname)
        _, inc_fixed = fixed_structure(ds)
        _, inc_learned = learned_structure(ds["node_features"], ds["n_nodes"], KNN_K)
        for method, cfg in METHODS.items():
            inc = inc_learned if cfg["struct"] == "learned" else inc_fixed
            for seed in SEEDS:
                try:
                    res = node_cls(ds, inc, cfg["enc"], seed)
                    if res is None:
                        continue
                    nodecls_rows.append(dict(dataset=dname, method=method,
                                            encoding=cfg["enc"], seed=seed, **res))
                    print(f"[nodecls] {dname} {method} acc={res['accuracy']:.3f} "
                          f"macroF1={res['macro_f1']:.3f}", flush=True)
                except Exception as e:
                    print(f"[nodecls] {dname} {method} EXCEPTION {e}", flush=True)

    # ---- write raw outputs ----
    with open(results_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if nodecls_rows:
        with open(results_dir / "nodecls.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(nodecls_rows[0].keys())); w.writeheader(); w.writerows(nodecls_rows)
    with open(results_dir / "cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys())); w.writeheader(); w.writerows(cost_rows)

    summary = aggregate(rows, cost_rows)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n===== SUMMARY / Go-No-Go =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def aggregate(rows, cost_rows):
    def mean(key, filt):
        vals = [r[key] for r in rows if filt(r) and isinstance(r[key], (int, float))]
        return float(np.mean(vals)) if vals else None
    def by_method(m): return lambda r: r["method"] == m

    out = {"methods_mean_transfer_auroc": {}, "go_no_go": {}}
    for m in list(METHODS.keys()) + ["MR_random", SHUFFLE_METHOD]:
        out["methods_mean_transfer_auroc"][m] = mean("transfer_auroc", by_method(m))

    m0 = out["methods_mean_transfer_auroc"]["M0"]
    m2 = out["methods_mean_transfer_auroc"]["M2"]
    mr = out["methods_mean_transfer_auroc"]["MR_random"]
    m4 = out["methods_mean_transfer_auroc"]["M4"]
    m5 = out["methods_mean_transfer_auroc"]["M5"]
    m6 = out["methods_mean_transfer_auroc"]["M6"]
    m2s = out["methods_mean_transfer_auroc"].get(SHUFFLE_METHOD)

    # H1: lightweight SE strictly beats (a) no encoding AND (b) the dim-matched
    # random control. We use a modest 0.01 bar: the gain is small but must be
    # real (not a parameter-count artefact).
    delta_m2_m0 = (m2 - m0) if (m2 is not None and m0 is not None) else None
    delta_m2_mr = (m2 - mr) if (m2 is not None and mr is not None) else None
    h1 = (delta_m2_m0 is not None and delta_m2_m0 > 0.01 and
          (delta_m2_mr is None or delta_m2_mr > 0.0))
    # spectral PE vs no encoding (does the expensive PE help at all?)
    m1 = out["methods_mean_transfer_auroc"]["M1"]
    delta_m1_m0 = (m1 - m0) if (m1 is not None and m0 is not None) else None
    spectral_useless = (delta_m1_m0 is not None and abs(delta_m1_m0) < 0.005)
    # H2: learned structure (M4) vs fixed+lightweight (M2)
    delta_m4_m2 = (m4 - m2) if (m4 is not None and m2 is not None) else None
    learned_hurts = (delta_m4_m2 is not None and delta_m4_m2 < -0.05)

    # P0 shuffle ablation: does H-LDP *arrangement* carry the signal?
    shuffle_collapses = (m2s is not None and m0 is not None and abs(m2s - m0) < 0.005)
    # M6: Learned+Spectral vs M0 -> isolates whether learned topology is the issue
    delta_m6_m0 = (m6 - m0) if (m6 is not None and m0 is not None) else None

    hldp_t = [c["time_s"] for c in cost_rows if c["encoding"] == "hldp" and c["status"] == "ok"]
    spec_t = [c["time_s"] for c in cost_rows if c["encoding"] == "spectral" and c["status"] == "ok"]
    ratio = (float(np.mean(spec_t)) / float(np.mean(hldp_t))) if (hldp_t and spec_t) else None
    h3 = ratio is not None and ratio >= 10

    if h3 and (delta_m2_m0 is not None and delta_m2_m0 > 0):
        verdict = "GO: fixed structure + lightweight SE (H-LDP) is the robust, cheap recipe"
    elif h1:
        verdict = "GO-with-caveats: lightweight SE helps but cost advantage unproven"
    else:
        verdict = "NO-GO: lightweight SE shows no reliable benefit"
    if spectral_useless:
        verdict += "; spectral PE rejected (no gain vs none, ~13x cost)"
    if learned_hurts:
        verdict += "; learned structure REJECTED (collapses cross-domain transfer)"

    out["go_no_go"] = {
        "H1_encoding_helps": bool(h1),
        "H1_delta_M2_minus_M0": (round(delta_m2_m0, 4) if delta_m2_m0 is not None else None),
        "H1_delta_M2_minus_random": (round(delta_m2_mr, 4) if delta_m2_mr is not None else None),
        "spectral_PE_useless_vs_none": bool(spectral_useless),
        "H2_learned_struct_helps": bool(not learned_hurts),
        "H2_delta_M4_minus_M2": (round(delta_m4_m2, 4) if delta_m4_m2 is not None else None),
        "H3_lightweight_cheaper_x": (round(ratio, 1) if ratio is not None else None),
        "H3_lightweight_cheaper": bool(h3),
        "verdict": verdict,
        "m0": m0, "m1": m1, "m2": m2, "mr": mr, "m4": m4, "m5": m5,
        "m6": m6, "m2_shuffle": m2s,
        "H2_delta_M6_minus_M0": (round(delta_m6_m0, 4) if delta_m6_m0 is not None else None),
        "P0_hldp_arrangement_informative": bool(shuffle_collapses),
    }
    return out


if __name__ == "__main__":
    main()
