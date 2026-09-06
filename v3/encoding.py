"""Hypergraph Random-Walk Positional Encoding (HyperPE).

Mirrors GFSE's absolute/relative random-walk PE (P, R) but defined on
hypergraphs instead of simple graphs. The only structural change vs GFSE is
the random-walk transition matrix M, which we define two ways:

  * walk='incidence' (default): M = D_v^-1 H D_e^-1 H^T  (node->hyperedge->node).
    Avoids clique expansion, so it does NOT quadratic-weight large hyperedges.
  * walk='clique':     M = D^-1 (H H^T)  (standard clique expansion).
    Included for ablation; reproduces GFSE exactly on 2-uniform hypergraphs.

Size-invariance (our core contribution over GFSE): when size_invariant=True
the hyperedge-degree in the denominator is raised to beta<1, damping the
disproportionate influence of very large hyperedges. This makes R comparable
across domains whose average hyperedge size differs by an order of magnitude.

P_i  = [I, M, M^2, ..., M^{d-1}]_{i,i}   -> absolute PE  (N, d)
R_ij = [I, M, M^2, ..., M^{d-1}]_{i,j}   -> relative PE  (N, N, d)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class HypergraphRandomWalkPE:
    def __init__(
        self,
        dim: int = 8,
        walk: str = "incidence",
        size_invariant: bool = False,
        beta: float = 0.5,
        device: str = "cpu",
    ):
        assert walk in ("incidence", "clique"), walk
        self.dim = dim
        self.walk = walk
        self.size_invariant = size_invariant
        self.beta = beta
        self.device = device

    # ------------------------------------------------------------------ #
    def _random_walk_matrix(self, H: np.ndarray, node_deg, edge_sizes) -> np.ndarray:
        N, E = H.shape
        if self.walk == "clique":
            A = H @ H.T
            A = A - np.diag(np.diag(A)) if N > 0 else A
            rs = A.sum(1, keepdims=True)
            rs[rs == 0] = 1.0
            return A / rs
        # incidence: M = D_v^-1 H D_e^-1 H^T, then row-normalize
        Dv = node_deg.reshape(-1, 1).astype(np.float64)
        Dv[Dv == 0] = 1.0
        exp = 1.0 if not self.size_invariant else self.beta
        De = (edge_sizes.astype(np.float64) ** exp).reshape(1, -1)
        De[De == 0] = 1.0
        He = H / De  # (N,E) node->hyperedge transfer
        M = (H @ He.T) / Dv  # (N,N)
        rs = M.sum(1, keepdims=True)
        rs[rs == 0] = 1.0
        return M / rs

    # ------------------------------------------------------------------ #
    def forward(self, H, return_spd: bool = False):
        """H: (N,E) 0/1 incidence (numpy). Returns P (N,d), R (N,N,d)[, spd (N,N)]."""
        H = np.asarray(H, dtype=np.float64)
        N, E = H.shape
        node_deg = H.sum(1)
        edge_sizes = H.sum(0)
        M = self._random_walk_matrix(H, node_deg, edge_sizes)

        I = np.eye(N)
        powers = [I]
        cur = I.copy()
        for _ in range(1, self.dim):
            cur = cur @ M
            powers.append(cur)
        powers = np.stack(powers, axis=-1)  # (N,N,d)

        P = np.array(np.diagonal(powers, axis1=0, axis2=1).T, copy=True)  # (N,d)
        R = np.array(powers, copy=True)  # (N,N,d)

        P_t = torch.as_tensor(P, dtype=torch.float32, device=self.device)
        R_t = torch.as_tensor(R, dtype=torch.float32, device=self.device)
        if not return_spd:
            return P_t, R_t

        # Hypergraph shortest-path distance via clique-expansion adjacency
        # (two nodes connected if they co-occur in >=1 hyperedge).
        adj = csr_matrix((H @ H.T) > 0)
        spd = shortest_path(adj, directed=False, unweighted=True)
        spd = np.nan_to_num(spd, nan=0.0, posinf=0.0)
        diam = spd.max() if spd.max() > 0 else 1.0
        spd = spd / diam  # normalized by diameter (GFSE Appendix A.2)
        spd_t = torch.as_tensor(spd, dtype=torch.float32, device=self.device)
        return P_t, R_t, spd_t

    # ------------------------------------------------------------------ #
    @staticmethod
    def hypergraph_spd(H: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Normalized hypergraph shortest-path distance (clique-expansion),
        without building the full PE. Diameter-normalized (GFSE Appendix A.2)."""
        H = np.asarray(H, dtype=np.float64)
        adj = csr_matrix((H @ H.T) > 0)
        spd = shortest_path(adj, directed=False, unweighted=True)
        spd = np.nan_to_num(spd, nan=0.0, posinf=0.0)
        diam = spd.max() if spd.max() > 0 else 1.0
        spd = spd / diam
        return torch.as_tensor(spd, dtype=torch.float32, device=device)
