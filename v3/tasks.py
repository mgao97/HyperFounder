"""Four self-supervised pre-training heads for HyperGFSE.

Each mirrors a GFSE task, adapted to hypergraphs:

  1. HSPD regression       : regress hypergraph shortest-path distance (clique-exp
                             distance by default) for node pairs.
  2. h-Motif counting       : regress node-level hypergraph motif statistics.
  3. HyperCommunity detect  : contrastive, pull same-community nodes together.
  4. Graph contrastive      : pull graphs from the same dataset together, push
                             graphs from different datasets apart (GFSE L_GCL).

Uncertainty-based loss balancing (GFSE Appendix A.3) is provided as a helper.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- heads --------------------------------------- #
class PairHead(nn.Module):
    """Concatenation pair head: (P_i || P_j) -> scalar (used for HSPD)."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, dim) -> return (N,N) scalar matrix
        N = z.size(0)
        a = z.unsqueeze(1).expand(N, N, -1)
        b = z.unsqueeze(0).expand(N, N, -1)
        pair = torch.cat([a, b], dim=-1)  # (N,N,2d)
        return self.net(pair).squeeze(-1)  # (N,N)


class NodeHead(nn.Module):
    """Node-level head: P_i -> R^k (used for h-Motif)."""

    def __init__(self, dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EmbedHead(nn.Module):
    """Node-level embedding head (used for community detection)."""

    def __init__(self, dim: int, out_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


# --------------------------- losses -------------------------------------- #
def hspd_loss(pred: torch.Tensor, target: torch.Tensor, pairs: torch.Tensor | None = None):
    """pred, target: (N,N). If pairs (M,2) given, average only over those pairs."""
    if pairs is None:
        return F.mse_loss(pred, target)
    i, j = pairs[:, 0], pairs[:, 1]
    return F.mse_loss(pred[i, j], target[i, j])


def motif_loss(pred: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(pred, target)


def community_loss(emb: torch.Tensor, same: torch.Tensor, eps: float = 1.0):
    """emb: (N,c) normalized. same: (N,N) 0/1. Margin contrastive (GFSE LCD)."""
    sim = emb @ emb.T  # (N,N) cosine because normalized
    term_same = same * (1.0 - sim)
    term_diff = (1.0 - same) * F.relu(eps - (1.0 - sim))
    mask = same + (1.0 - same)
    return (term_same + term_diff).sum() / mask.sum().clamp(min=1.0)


def community_loss_pairs(emb: torch.Tensor, same: torch.Tensor, pairs: torch.Tensor | None, eps: float = 1.0):
    """Pair-sampled version of community_loss for large N (avoids the full (N,N) matmul).

    emb: (N,c) normalized; same: (N,N) 0/1; pairs: (M,2) long tensor of node indices.
    Used by Pretrainer when graphs are large.
    """
    if pairs is None or pairs.size(0) == 0:
        return torch.zeros((), device=emb.device, requires_grad=True)
    i, j = pairs[:, 0], pairs[:, 1]
    sim = (emb[i] * emb[j]).sum(-1)  # cosine because normalized
    s = same[i, j]
    term_same = s * (1.0 - sim)
    term_diff = (1.0 - s) * F.relu(eps - (1.0 - sim))
    return (term_same + term_diff).mean()


def hypergraph_community_matrix(H: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Structural community proxy for hypergraphs (GFSE labels communities with
    shared neighbours). Two nodes belong to the same community when the Jaccard
    overlap of their hyperedge memberships exceeds `thr`. Returns (N,N) 0/1."""
    H = np.asarray(H, dtype=np.float64)
    A = H @ H.T  # co-occurrence count
    deg = A.diagonal().reshape(-1, 1)
    union = deg + deg.T - A
    J = np.where(union > 0, A / np.maximum(union, 1e-9), 0.0)
    np.fill_diagonal(J, 1.0)
    return (J > thr).astype(np.float32)


def gcl_loss(graph_emb: torch.Tensor, same_dataset: torch.Tensor, tau: float = 0.1):
    """graph_emb: (B,c). same_dataset: (B,B) 0/1. In-batch cross-dataset contrast."""
    sim = F.cosine_similarity(graph_emb.unsqueeze(1), graph_emb.unsqueeze(0), dim=-1) / tau
    B = graph_emb.size(0)
    labels = torch.arange(B, device=graph_emb.device)
    # positives = graphs from the same dataset (excluding self)
    pos = same_dataset.float() - torch.eye(B, device=graph_emb.device)
    # denominator = all except self
    denom = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=graph_emb.device), -1e9)
    logprob = sim[labels, :] - torch.logsumexp(denom, dim=1)
    denom_pos = pos.sum(1).clamp(min=1.0)
    pos_logprob = (pos * logprob.unsqueeze(0)).sum(1) / denom_pos
    return -pos_logprob.mean()


# --------------------------- uncertainty ---------------------------------- #
class UncertaintyWeights(nn.Module):
    """Task-specific homoscedastic uncertainty (GFSE Eq.7 / Kendall et al. 2018)."""

    def __init__(self, tasks=("spd", "motif", "community", "gcl")):
        super().__init__()
        self.tasks = list(tasks)
        self.log_sigma = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in self.tasks})

    def __call__(self, losses: dict) -> torch.Tensor:
        total = 0.0
        for t in self.tasks:
            s = self.log_sigma[t]
            total = total + (1.0 / (2 * torch.exp(s) ** 2)) * losses[t] + s
        return total


# --------------------------- h-motif labels ------------------------------ #
def compute_hmotif_labels(H: np.ndarray, k: int = 8) -> np.ndarray:
    """Lightweight node-level hypergraph motif statistics as regression targets.

    Returns (N, k) float matrix. The k channels capture local hypergraph
    structure around each node:
      [0] #hyperedges the node belongs to
      [1] #distinct co-occurring nodes (neighbor count)
      [2] max hyperedge size it belongs to
      [3] mean hyperedge size it belongs to
      [4] #hyperedges of size 2 it touches
      [5] #hyperedges of size 3 it touches
      [6] #hyperedges of size >=4 it touches
      [7] #(3-node cliques: two other nodes co-occurring with it in >=1 he)
    (truncated/padded to k). These are hypergraph analogues of graphlet degrees.
    """
    H = np.asarray(H, dtype=np.float64)
    N, E = H.shape
    node_deg = H.sum(1)
    sizes = H.sum(0)
    # neighbor count via clique adjacency
    A = (H @ H.T)
    np.fill_diagonal(A, 0.0)
    neigh = (A > 0).sum(1)
    max_sz = np.array([sizes[H[v, :] > 0].max() if (H[v, :] > 0).any() else 0 for v in range(N)])
    mean_sz = np.array([sizes[H[v, :] > 0].mean() if (H[v, :] > 0).any() else 0 for v in range(N)])
    cnt_sz2 = np.array([((H[v, :] > 0) & (sizes == 2)).sum() for v in range(N)]).astype(float)
    cnt_sz3 = np.array([((H[v, :] > 0) & (sizes == 3)).sum() for v in range(N)]).astype(float)
    cnt_sz4 = np.array([((H[v, :] > 0) & (sizes >= 4)).sum() for v in range(N)]).astype(float)
    # 3-node co-occurrence count per node
    tri = np.zeros(N)
    for e in range(E):
        members = np.where(H[:, e] > 0)[0]
        if len(members) >= 3:
            for v in members:
                tri[v] += (len(members) - 1)
    feats = np.stack([node_deg, neigh, max_sz, mean_sz, cnt_sz2, cnt_sz3, cnt_sz4, tri], axis=1)
    feats = feats[:, :k]
    if feats.shape[1] < k:
        feats = np.pad(feats, ((0, 0), (0, k - feats.shape[1])), mode="constant")
    # standardize across nodes
    mu, sd = feats.mean(0, keepdims=True), feats.std(0, keepdims=True)
    feats = (feats - mu) / (sd + 1e-6)
    return feats.astype(np.float32)
