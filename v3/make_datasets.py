"""
Generate coverage node splits for the co-citation citation datasets and store them
under v3/datasets/<name>/ in the same pickle format that the HyperFounder loader
expects (edge_list / features / labels / train_mask / val_mask / test_mask).

Split rule ("full-node coverage split"):
    train = TRAIN_PER_CLASS nodes per class
    val   = VAL_PER_CLASS   nodes per class
    test  = all remaining nodes  (so every node is covered exactly once)

This yields:
    cora    : 20/class * 7  = 140 train, 100/class * 7 = 700 val, 1868 test  (2708 total)
    citeseer: 20/class * 6  = 120 train, 100/class * 6 = 600 val, 2592 test  (3312 total)
    pubmed  : 20/class * 3  =  60 train, 100/class * 3 = 300 val, 19357 test (19717 total)

Note: dhg's co-citation Citeseer keeps 3312 nodes (15 nodes with null features are
dropped), so the test count is 2592 rather than 2607 (= 3327 - 720). The split
covers all available nodes of the processed dataset.

Raw edge_list / features / labels are copied verbatim from the existing
co-citation cache (data/cache/cocitation_*) -- no re-download needed.
"""

import os
import argparse
import pickle
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "v3", "datasets")

SOURCES = {
    "cora": "cocitation_cora",
    "citeseer": "cocitation_citeseer",
    "pubmed": "cocitation_pubmed",
}

TRAIN_PER_CLASS = 20
VAL_PER_CLASS = 100


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_pickle(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def main():
    parser = argparse.ArgumentParser(description="Generate v3 coverage splits")
    parser.add_argument("--names", nargs="+", default=list(SOURCES.keys()))
    parser.add_argument("--train-per-class", type=int, default=TRAIN_PER_CLASS)
    parser.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name in args.names:
        src = os.path.join(CACHE, SOURCES[name])
        if not os.path.isdir(src):
            print(f"[SKIP] {name}: cache not found at {src}")
            continue

        dd = os.path.join(OUT, name)
        os.makedirs(dd, exist_ok=True)

        edge_list = load_pickle(os.path.join(src, "edge_list.pkl"))
        feat = load_pickle(os.path.join(src, "features.pkl"))
        labels = np.asarray(load_pickle(os.path.join(src, "labels.pkl"))).flatten()

        if hasattr(feat, "toarray"):
            feat = feat.toarray()
        feat = np.asarray(feat, dtype=np.float32)

        N = feat.shape[0]
        nc = int(labels.max()) + 1

        # ---- coverage split ----
        rng = np.random.default_rng(args.seed)
        train_idx, val_idx = [], []
        for c in range(nc):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            train_idx.extend(idx[: args.train_per_class].tolist())
            val_idx.extend(
                idx[args.train_per_class : args.train_per_class + args.val_per_class].tolist()
            )
        train_idx = np.array(sorted(train_idx), dtype=np.int64)
        val_idx = np.array(sorted(val_idx), dtype=np.int64)
        test_idx = np.array(
            sorted(set(range(N)) - set(train_idx.tolist()) - set(val_idx.tolist())),
            dtype=np.int64,
        )

        tr = np.zeros(N, dtype=bool)
        va = np.zeros(N, dtype=bool)
        te = np.zeros(N, dtype=bool)
        tr[train_idx] = True
        va[val_idx] = True
        te[test_idx] = True

        # ---- write v3 dataset (raw + new masks) ----
        save_pickle(edge_list, os.path.join(dd, "edge_list.pkl"))
        save_pickle(feat, os.path.join(dd, "features.pkl"))
        save_pickle(labels.astype(np.int64), os.path.join(dd, "labels.pkl"))
        save_pickle(tr, os.path.join(dd, "train_mask.pkl"))
        save_pickle(va, os.path.join(dd, "val_mask.pkl"))
        save_pickle(te, os.path.join(dd, "test_mask.pkl"))

        print(
            f"[OK] {name}: N={N} classes={nc} | "
            f"train={int(tr.sum())} val={int(va.sum())} test={int(te.sum())} "
            f"(edges={len(edge_list)}, feat_dim={feat.shape[1]})"
        )


if __name__ == "__main__":
    main()
