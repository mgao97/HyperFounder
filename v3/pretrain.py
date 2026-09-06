"""Pre-training driver and downstream (linear-probe) evaluation for HyperGFSE.

The pre-training loop mirrors GFSE Sec.3.3 / 4.1: sample graphs from multiple
domains, compute PSE with the frozen-or-trained encoder, decode the four tasks,
and optimize the uncertainty-weighted loss. Downstream usage mirrors GFSE
Sec.3.4: concatenate PSE with raw features and feed any GNN/MLP.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .encoder import HyperGFSE
from .encoding import HypergraphRandomWalkPE
from .tasks import (
    PairHead, NodeHead, EmbedHead,
    hspd_loss, motif_loss, community_loss, community_loss_pairs, gcl_loss,
    UncertaintyWeights, compute_hmotif_labels, hypergraph_community_matrix,
)


class Pretrainer:
    def __init__(self, encoder: HyperGFSE, cfg: dict | None = None):
        self.enc = encoder
        self.cfg = cfg or {}
        self.out_dim = encoder.out_proj.out_features
        self.spd_head = PairHead(self.out_dim)
        self.motif_head = NodeHead(self.out_dim, self.cfg.get("motif_k", 8))
        self.cd_head = EmbedHead(self.out_dim, self.cfg.get("cd_dim", 32))
        self.gcl_head = EmbedHead(self.out_dim, self.cfg.get("gcl_dim", 32))
        self.uw = UncertaintyWeights()
        self.opt = torch.optim.Adam(
            [p for m in (self.enc, self.spd_head, self.motif_head, self.cd_head,
                         self.gcl_head, self.uw) for p in m.parameters()],
            lr=self.cfg.get("lr", 1e-3),
        )

    # ------------------------------------------------------------------ #
    def train_step(self, batch):
        """batch: list of dicts {H(np), spd(np), motif(np), comm(np), ds(int)}.
        Pairwise tasks (spd, community) use at most `max_pairs` sampled node
        pairs to stay O(pairs) regardless of graph size (critical for news20)."""
        self.opt.zero_grad(set_to_none=True)
        losses = {}
        gcl_embs, ds_ids = [], []
        max_pairs = int(self.cfg.get("max_pairs", 20000))
        total = 0.0
        for item in batch:
            H = item["H"]
            device = next(self.enc.parameters()).device
            pse = self.enc(H)  # (N, out_dim)
            N = pse.size(0)
            spd = torch.as_tensor(item["spd"], device=device)
            motif_y = torch.as_tensor(item["motif"], device=device)
            comm = torch.as_tensor(item["comm"], device=device)

            # sample node pairs for the pairwise tasks
            if N > 1:
                ii, jj = torch.randint(0, N, (2, max_pairs))
                pairs = torch.stack([ii, jj], 1).to(device)
            else:
                pairs = None

            l_spd = hspd_loss(self.spd_head(pse), spd, pairs)
            l_motif = motif_loss(self.motif_head(pse), motif_y)
            l_cd = community_loss_pairs(self.cd_head(pse), comm, pairs, eps=self.cfg.get("cd_eps", 1.0))
            for k, v in (("spd", l_spd), ("motif", l_motif), ("community", l_cd)):
                losses[k] = losses.get(k, 0.0) + v
            gcl_embs.append(self.gcl_head(pse).mean(0, keepdim=True))
            ds_ids.append(item["ds"])

        gcl_embs = torch.cat(gcl_embs, 0)  # (B, c)
        ds_mat = torch.tensor(ds_ids, device=gcl_embs.device)
        same = (ds_mat.unsqueeze(0) == ds_mat.unsqueeze(1)).float()
        l_gcl = gcl_loss(gcl_embs, same, tau=self.cfg.get("tau", 0.1))
        losses["gcl"] = l_gcl

        loss = self.uw(losses)
        loss.backward()
        self.opt.step()
        return {k: float(v.detach()) / max(1, len(batch)) for k, v in losses.items()}, float(loss.detach())


# ------------------------------------------------------------------ #
def build_pretrain_item(H: np.ndarray, ds_id: int, spd_d: int = 8, motif_k: int = 8,
                        comm_thr: float = 0.5, device: str = "cpu") -> dict:
    """Build the four self-supervised label tensors for one hypergraph.

    Returns a dict consumable by Pretrainer.train_step:
      {H, spd(N,N), motif(N,k), comm(N,N), ds}. Computed once per graph; the
      expensive SPD/clique-expansion shortest path only runs here, not in the
      encoder forward pass.
    """
    spd = HypergraphRandomWalkPE.hypergraph_spd(H, device=device)
    motif = compute_hmotif_labels(H, k=motif_k)
    comm = hypergraph_community_matrix(H, thr=comm_thr)
    return {"H": H, "spd": spd.numpy(), "motif": motif, "comm": comm, "ds": ds_id}


# --------------------------- downstream eval ------------------------------ #
def linear_probe_eval(encoder: HyperGFSE, graphs, labels, hidden=64, epochs=100, lr=1e-3):
    """Concatenate PSE with raw node features (if any) and train a 2-layer MLP.
    graphs: list of (H_np, X_np_or_None). labels: list of (N,) int arrays.
    Returns mean test accuracy over a 1:1 train/test split (deterministic)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    Xs, ys = [], []
    with torch.no_grad():
        for (H, X), y in zip(graphs, labels):
            pse = encoder(H).cpu().numpy()
            if X is None:
                Xc = pse
            else:
                Xc = np.concatenate([X, pse], axis=1)
            Xs.append(Xc)
            ys.append(y)
    X_all = np.concatenate(Xs, 0)
    y_all = np.concatenate(ys, 0)
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.5, random_state=0, stratify=y_all)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    return acc
