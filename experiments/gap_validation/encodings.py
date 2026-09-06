"""Hypergraph structural encodings for the gap-validation T0 scale-drift study.

Implements two cheap encodings from Pellegrin, Fesser & Weber
(NeurIPS 2025, arXiv:2502.09570):

  * H-LDP  -- Hypergraph Local Degree Profile (node-level, R^6, NO normalization)
  * H-RWPE -- Hypergraph Random-Walk Positional Encoding (EE variant, R^k)

Both consume only the incidence structure (hyperedge list), so they are
feature-free and directly testable for cross-domain / cross-scale transfer.

H-RWPE is computed via the incidence-matrix identity
    P = normalize(B @ diag(w_e) @ B^T)      (off-diagonal)
so the O(sum |e|^2) pair enumeration is avoided entirely.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def _neighborhood_and_degree(num_v, e_list):
    deg = np.zeros(num_v, dtype=np.int64)
    neigh = [set() for _ in range(num_v)]
    for e in e_list:
        e = list(e)
        for i in e:
            deg[i] += 1
            for j in e:
                if j != i:
                    neigh[i].add(j)
    return deg, neigh


def h_ldp(num_v, e_list, out_dim=6):
    """Hypergraph Local Degree Profile.

    d_v = #hyperedges containing v; DN(v) = {d_u : u in N_v}.
    Feature = [d_v, min DN, max DN, mean DN, median DN, std DN] (R^6).
    Per the paper, NO preprocessing / normalization is applied.
    """
    assert out_dim == 6
    deg, neigh = _neighborhood_and_degree(num_v, e_list)
    feats = np.zeros((num_v, 6), dtype=np.float64)
    for i in range(num_v):
        dn = np.array([deg[u] for u in neigh[i]], dtype=np.float64)
        if dn.size == 0:
            dn = np.array([deg[i]], dtype=np.float64)
        feats[i, 0] = deg[i]
        feats[i, 1] = dn.min()
        feats[i, 2] = dn.max()
        feats[i, 3] = dn.mean()
        feats[i, 4] = np.median(dn)
        feats[i, 5] = dn.std()
    return feats


def _build_rwpe_transition(num_v, e_list, walk="EE"):
    rows, cols, data = [], [], []
    sizes = []
    for ei, e in enumerate(e_list):
        m = len(e)
        if m < 2:
            continue
        for i in e:
            rows.append(i); cols.append(ei); data.append(1.0)
        sizes.append(m)
    if not cols:
        return sp.csr_matrix((num_v, num_v)), np.zeros(num_v)
    B = sp.csr_matrix((data, (rows, cols)), shape=(num_v, len(sizes)))
    we = np.asarray(sizes, dtype=float)
    if walk == "EE":
        w = 1.0 / (we - 1.0)
        M = B @ sp.diags(w) @ B.T          # M[i,j] = sum_{e ni i,j} w_e ; diag = denom_i
        denom = np.asarray(M.diagonal()).ravel()
        M = M.tolil(); M.setdiag(0); M = M.tocsr()
        P = M.multiply(1.0 / np.maximum(denom, 1e-12)[:, None]).tocsr()
    elif walk == "WE":
        M1 = B @ B.T
        denom = np.asarray(B @ (we - 1.0)).ravel()
        M1 = M1.tolil(); M1.setdiag(0); M1 = M1.tocsr()
        P = M1.multiply(1.0 / np.maximum(denom, 1e-12)[:, None]).tocsr()
    else:  # EN
        M1 = (B @ B.T).tolil(); M1.setdiag(0); M1 = M1.tocsr()
        nbr = np.asarray((M1 > 0).sum(axis=1)).ravel()
        P = (M1 > 0).multiply(1.0 / np.maximum(nbr, 1.0)[:, None]).tocsr()
    return P, denom if walk != "EN" else nbr


def h_rwpe(num_v, e_list, k=19, walk="EE"):
    """Hypergraph Random-Walk Positional Encoding (diagonal return probs).

    Feature_i = [(P)_{ii}, (P^2)_{ii}, ..., (P^k)_{ii}] in R^k.
    P is sub-stochastic (off-diagonal normalized); we only read the diagonal.
    """
    P, _ = _build_rwpe_transition(num_v, e_list, walk=walk)
    if P.shape[0] == 0:
        return np.zeros((num_v, k))
    cap = min(20_000_000, max(2_000_000, num_v * num_v // 10))
    feats = np.zeros((num_v, k), dtype=np.float64)
    M = P
    for t in range(1, k + 1):
        feats[:, t - 1] = M.diagonal()
        if t < k:
            M = P @ M
            if M.nnz > cap:
                for tt in range(t + 1, k + 1):
                    feats[:, tt - 1] = M.diagonal()
                break
    return feats


def encode(num_v, e_list, kind, k=19):
    if kind == "H-LDP":
        return h_ldp(num_v, e_list)
    if kind in ("H-RWPE", "H-RWPE-EE"):
        return h_rwpe(num_v, e_list, k=k, walk="EE")
    raise ValueError("unknown encoding %s" % kind)
