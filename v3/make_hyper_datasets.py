"""
Prepare the 8 hypergraph datasets used in LightHGNN (ICLR 2024, Table 1) into
self-contained processed copies under v3/datasets/<name>/.

Paper name -> source (identical to iMoonLab/LightHGNN `utils.load_data`):
  News20      -> dhg.data.News20()                       ['edge_list']
  CA-Cora     -> dhg.data.CoauthorshipCora()             ['edge_list']
  CC-Cora     -> dhg.data.CocitationCora()               ['edge_list']
  CC-Citeseer -> dhg.data.CocitationCiteseer()           ['edge_list']
  DBLP-Paper  -> DBLP4k()                                ['edge_by_paper']
  DBLP-Term   -> DBLP4k()                                ['edge_by_term']
  DBLP-Conf   -> DBLP4k()                                ['edge_by_conf']
  IMDB-AW     -> IMDB4k()                                ['edge_by_actor'] + ['edge_by_director']

Each dataset dir gets:
  features.pt   float32 [N, F]  (row L1-normalized, as shipped by dhg)
  labels.pt     int64   [N]     (remapped to 0-based contiguous classes)
  edge_list.pt  list[list[int]] hyperedge list
  meta.json     stats
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines"))

from dhg.data import News20, CoauthorshipCora, CocitationCora, CocitationCiteseer
from dhg_extra_data import DBLP4k, IMDB4k

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(ROOT, "datasets")


def build_specs():
    dblp = DBLP4k()
    imdb = IMDB4k()
    return [
        ("news20", News20(), "edge_list"),
        ("ca_cora", CoauthorshipCora(), "edge_list"),
        ("cc_cora", CocitationCora(), "edge_list"),
        ("cc_citeseer", CocitationCiteseer(), "edge_list"),
        ("dblp4k_paper", dblp, "edge_by_paper"),
        ("dblp4k_term", dblp, "edge_by_term"),
        ("dblp4k_conf", dblp, "edge_by_conf"),
        ("imdb_aw", imdb, ("edge_by_actor", "edge_by_director")),
    ]


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for name, data, key in build_specs():
        out_dir = os.path.join(OUT_ROOT, name)
        if os.path.isfile(os.path.join(out_dir, "meta.json")):
            print(f"[skip] {name} already prepared at {out_dir}")
            continue

        print(f"[load] {name} ...")
        X = data["features"]
        y = data["labels"]
        if isinstance(key, tuple):
            edge_list = data[key[0]] + data[key[1]]
        else:
            edge_list = data[key]

        # clean types
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long).view(-1)

        # remap labels to 0-based contiguous ids
        uniq = torch.unique(y).tolist()
        remap = {int(v): i for i, v in enumerate(uniq)}
        y = torch.tensor([remap[int(v)] for v in y], dtype=torch.long)

        # sanitize edge list: dedup vertices inside each edge, drop empty edges
        edge_list = [sorted(set(int(v) for v in e)) for e in edge_list]
        edge_list = [e for e in edge_list if len(e) > 0]

        n, num_classes = int(X.shape[0]), int(len(uniq))
        assert int(y.shape[0]) == n
        assert y.max().item() == num_classes - 1
        max_v = max(max(e) for e in edge_list)
        assert max_v < n, f"{name}: edge refers to vertex {max_v} >= {n}"

        os.makedirs(out_dir, exist_ok=True)
        torch.save(X, os.path.join(out_dir, "features.pt"))
        torch.save(y, os.path.join(out_dir, "labels.pt"))
        torch.save(edge_list, os.path.join(out_dir, "edge_list.pt"))
        meta = {
            "name": name,
            "num_vertices": n,
            "num_hyperedges": len(edge_list),
            "num_classes": num_classes,
            "dim_features": int(X.shape[1]),
            "num_nonzero_incidence": int(sum(len(e) for e in edge_list)),
            "source": "LightHGNN (ICLR 2024) Table 1 datasets via dhg",
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {name}: {json.dumps(meta)}")

    print("\nAll 8 hypergraph datasets ready under", OUT_ROOT)


if __name__ == "__main__":
    main()
