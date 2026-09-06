"""T0 -- scale-drift demonstration (gap-validation plan).

Hypothesis (R2): existing hypergraph *structural* encodings degrade under
cross-domain / cross-scale transfer; degradation grows with the gap in average
hyperedge size between source and target.

Protocol (per pair, per encoding):
  * encode source graph (feature-free: H-LDP / H-RWPE)
  * hyperedge prediction = logistic regression on mean-pooled node encodings,
    positives = real hyperedges, negatives = popularity-matched (same size).
  * ZERO-SHOT: LR trained on SOURCE, evaluated on TARGET test edges.
  * IN-DOMAIN (upper bound): LR trained on TARGET train, evaluated on TARGET test.
  * drop = P_indomain - P_zero (higher => stronger scale drift).
x-axis = |log(avg_size_target / avg_size_source)|.
"""
from __future__ import annotations
import sys, os, time, json, math
import numpy as np

ROOT = "/home/user/GSK/mgao/HyperFounder"
sys.path.insert(0, ROOT)
from v2.utils.dhg_datasets import load_dhg_sample
import importlib.util
_spec = importlib.util.spec_from_file_location("hg_enc", os.path.join(ROOT, "experiments/gap_validation/encodings.py"))
hg_enc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hg_enc)

DATA_ROOT = os.path.join(ROOT, "data/cache")
K_RWPE = 19
ENCODINGS = ["H-LDP", "H-RWPE"]
TARGET_DIM = 16
N_CAP = 30000
RNG = np.random.default_rng(0)

PAIRS = [
    ("P1", "coauthorship_dblp", "coauthorship_cora", "close"),
    ("P2", "coauthorship_dblp", "citeseer_cc", "close"),
    ("P3", "coauthorship_dblp", "house_committees", "medium"),
    ("P4", "coauthorship_dblp", "cooking_200", "medium"),
    ("P5", "coauthorship_dblp", "movielens_1m", "extreme"),
]


def load(name):
    g = load_dhg_sample(name, target_dim=TARGET_DIM, seed=0, data_root=DATA_ROOT)
    e_list = [list(e) for e in g.hyperedges if len(e) >= 2]
    md = g.metadata
    avg = md.get("avg_hyperedge_size", sum(len(e) for e in e_list) / max(len(e_list), 1))
    return g.num_nodes, e_list, float(avg)


def edge_feats(node_enc, e_list):
    out, keep = [], []
    for e in e_list:
        if len(e) >= 2:
            out.append(node_enc[list(e)].mean(axis=0)); keep.append(e)
    return np.array(out), keep


def matched_negatives(e_list, num_nodes, rng):
    sizes = np.array([len(e) for e in e_list if len(e) >= 2])
    negs = []
    for _ in range(len(sizes)):
        s = int(rng.choice(sizes))
        if s >= num_nodes:
            s = num_nodes - 1
        nodes = rng.choice(num_nodes, size=s, replace=False)
        negs.append(tuple(sorted(int(x) for x in nodes)))
    return negs, sizes


def fit_lr(X, y, use_sklearn):
    if use_sklearn:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(sc.transform(X), y)
        return ("sklearn", sc, m)
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d); b = 0.0; lr = 0.3; reg = 1e-3
    for _ in range(2000):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        gw = Xs.T @ (p - y) + reg * w
        gb = (p - y).sum()
        w -= lr * gw / n; b -= lr * gb / n
    return ("numpy", mu, sd, w, b)


def pred_lr(params, X):
    if params[0] == "sklearn":
        _, sc, m = params
        return m.predict_proba(sc.transform(X))[:, 1]
    _, mu, sd, w, b = params
    return 1.0 / (1.0 + np.exp(-(((X - mu) / sd) @ w + b)))


def auroc(scores, y):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_r = ranks[pos].sum()
    return (sum_r - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    try:
        import sklearn
        use_sk = True
        print("[info] sklearn available")
    except Exception:
        use_sk = False
        print("[info] sklearn NOT available -> numpy LR")

    src_name = PAIRS[0][1]
    print(f"[load] source {src_name} ...", flush=True)
    t0 = time.time()
    num_v_s, e_list_s, avg_s = load(src_name)
    print(f"   source nodes={num_v_s} edges={len(e_list_s)} avg={avg_s:.2f} ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for kind in ENCODINGS:
        t0 = time.time()
        enc_s = hg_enc.encode(num_v_s, e_list_s, kind, k=K_RWPE)
        print(f"[enc] {kind} source done ({time.time()-t0:.1f}s) dim={enc_s.shape[1]}", flush=True)
        spos_f, _ = edge_feats(enc_s, e_list_s)
        sneg_e, _ = matched_negatives(e_list_s, num_v_s, RNG)
        sneg_f = np.array([enc_s[list(e)].mean(0) for e in sneg_e])
        if spos_f.shape[0] > N_CAP:
            idx = RNG.choice(spos_f.shape[0], N_CAP, replace=False)
            spos_f = spos_f[idx]; sneg_f = sneg_f[idx]
        Xs = np.vstack([spos_f, sneg_f])
        ys = np.concatenate([np.ones(len(spos_f)), np.zeros(len(sneg_f))])

        for pid, _, tgt_name, scale in PAIRS:
            t1 = time.time()
            num_v_t, e_list_t, avg_t = load(tgt_name)
            enc_t = hg_enc.encode(num_v_t, e_list_t, kind, k=K_RWPE)
            tpos_f, _ = edge_feats(enc_t, e_list_t)
            tneg_e, _ = matched_negatives(e_list_t, num_v_t, RNG)
            tneg_f = np.array([enc_t[list(e)].mean(0) for e in tneg_e])
            Xt = np.vstack([tpos_f, tneg_f])
            yt = np.concatenate([np.ones(len(tpos_f)), np.zeros(len(tneg_f))])
            n = Xt.shape[0]
            perm = RNG.permutation(n)
            cut = int(0.6 * n)
            tr, te = perm[:cut], perm[cut:]
            ps = fit_lr(Xs, ys, use_sk)
            P_zero = auroc(pred_lr(ps, Xt[te]), yt[te])
            pt = fit_lr(Xt[tr], yt[tr], use_sk)
            P_ind = auroc(pred_lr(pt, Xt[te]), yt[te])
            drop = P_ind - P_zero
            x = abs(math.log(avg_t / avg_s))
            rows.append(dict(pair=pid, target=tgt_name, scale=scale, encoding=kind,
                             avg_src=avg_s, avg_tgt=avg_t, x=round(x, 3),
                             P_zero=round(P_zero, 4), P_indomain=round(P_ind, 4),
                             drop=round(drop, 4)))
            print(f"   {kind} {pid}->{tgt_name}: avg {avg_s:.2f}->{avg_t:.2f} |Pzero={P_zero:.3f} Pindo={P_ind:.3f} drop={drop:+.3f}| ({time.time()-t1:.1f}s)", flush=True)

    out_dir = os.path.join(ROOT, "experiments/gap_validation/outputs_t0")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "t0_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n==== T0 scale-drift results ====")
    print(f"{'pair':4} {'target':18} {'scale':7} {'enc':7} {'x':>5} {'Pzero':>7} {'Pindo':>7} {'drop':>7}")
    for r in rows:
        print(f"{r['pair']:4} {r['target']:18} {r['scale']:7} {r['encoding']:7} {r['x']:5.2f} {r['P_zero']:7.3f} {r['P_indomain']:7.3f} {r['drop']:+7.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = {"H-LDP": "tab:blue", "H-RWPE": "tab:red"}
        for kind in ENCODINGS:
            xs = [r["x"] for r in rows if r["encoding"] == kind]
            ds = [r["drop"] for r in rows if r["encoding"] == kind]
            ax.scatter(xs, ds, c=cmap.get(kind, "k"), s=70, label=kind, zorder=3)
            if len(xs) >= 2:
                z = np.polyfit(xs, ds, 1)
                xx = np.linspace(min(xs), max(xs), 50)
                ax.plot(xx, np.polyval(z, xx), c=cmap.get(kind, "k"), ls="--", alpha=0.6)
        ax.axhline(0, c="grey", lw=0.8)
        ax.set_xlabel("|log(avg_size_target / avg_size_source)|  (scale gap)")
        ax.set_ylabel("drop = P_indomain - P_zero  (transfer degradation)")
        ax.set_title("T0: hypergraph encoding scale-drift (zero-shot hyperedge prediction)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figure1_scale_drift.png"), dpi=140)
        print(f"[ok] figure saved -> {out_dir}/figure1_scale_drift.png")
    except Exception as ex:
        print(f"[warn] figure skipped: {ex}")


if __name__ == "__main__":
    main()
