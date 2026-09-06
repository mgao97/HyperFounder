"""
HyperFounder v2 — Unified Hypergraph Structural Encoder.

严格按照 `v2/encoder-design-spec.md` 实现（导师 2026-09-01 规格书）：

  挑战 1 (基数漂移/坍缩)  →  **CCA (Cardinality-Conditioned Attention)**
        - 基数编码 c_e = MLP([log(1+|e|),  |e|/d_max])
        - FiLM 调制 Q' = γ(c_e) ⊙ Q + β(c_e)
        - 基数温度 softmax τ(c_e) = softplus(MLP(c_e)) + ε

  挑战 2 (超边间高阶结构盲区)  →  **HCA (Hyperedge-Context Attention)**
        - 稀疏 HEDG S = Hᵀ·H，每条超边留 Top-K overlap 邻居
        - 对称归一 ô = |e_i∩e_j| / √(|e_i||e_j|)
        - logit 偏置 b(ô) 是一层 MLP（可学习函数）
        - 上下文 E'_i = E_i + Σ α_ij V_j，再回写节点 X'_v = Σ_{e∋v} E'_e / d_v

  可选模块 3  →  HOR (Higher-Order Readout，开关 use_hor)
        - 2-clique 共成员图 C = I·Iᵀ - diag，Top-K 稀疏化
        - 权重 = Σ_e A_ue·A_ve·(|e|−2)  再归一化后 re-weight attention

  前向流程（§1）:
    X, H → [模块0 Tokenize]  E = σ(mean_{v∈e}(X_v W_p) + PE_card(c_e))
         → 3 × { CCA(node→edge FiLM attn) → edge→node mean 回传 }
         → 1 × HCA(超边上下文 Top-K overlap attn + 节点回写)
         → [可选 HOR]
         → out_norm

Encoder 保持**完全 domain-blind**：不接收 domain_id，不持有任何 per-domain 参数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import nn

from v2.models.cross_domain_modules import hypergraph_rw_pe, overlap_coefficient


# ---------------------------------------------------------------------------
# 0. Module 0 — Hyperedge Tokenization （§2）
# ---------------------------------------------------------------------------

def _sparse_scatter_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    num_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter-mean `values` along dim=0 grouped by `index` ∈ [0, num_groups).

    Returns (sum_per_group, count_per_group).  Caller can divide / re-use count.
    """
    feat_dim = values.size(-1)
    device = values.device
    sum_per_group = torch.zeros(num_groups, feat_dim, dtype=values.dtype, device=device)
    sum_per_group.index_add_(0, index, values)
    count_per_group = torch.zeros(num_groups, 1, dtype=values.dtype, device=device)
    count_per_group.index_add_(0, index, torch.ones(index.numel(), 1, dtype=values.dtype, device=device))
    return sum_per_group, count_per_group


# ---------------------------------------------------------------------------
# 1. Module 1 — CCA: Cardinality-Conditioned Attention （§3）
# ---------------------------------------------------------------------------

class CardinalityConditionedAttention(nn.Module):
    """Node→Edge attention modulated by per-edge cardinality.

    规格书 §3 三件套：基数编码 → FiLM → 温度 softmax。
    对于每条超边 e，所有 node→e query 共享同一个 c_e，因此可缓存到 tokenize 阶段。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, pe_dim: int = 32):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) % num_heads ({num_heads}) != 0")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        pe_dim = max(int(pe_dim), 1)

        # 基数编码 MLP（2-dim raw → 压缩到 pe 1 → hidden
        card_inner = max(4, pe_dim // 2)
        self.card_mlp = nn.Sequential(
            nn.Linear(2, card_inner),
            nn.GELU(),
            nn.Linear(card_inner, hidden_dim),
        )
        # FiLM: γ, β per dimension； bottleneck = min(8, pe_dim/4)
        film_inner = max(4, pe_dim // 4)
        self.film_gamma = nn.Sequential(
            nn.Linear(hidden_dim, film_inner),
            nn.GELU(),
            nn.Linear(film_inner, hidden_dim),
            nn.Sigmoid(),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(hidden_dim, film_inner),
            nn.GELU(),
            nn.Linear(film_inner, hidden_dim),
        )
        # 温度 head: scalar per edge → softplus + ε
        tau_inner = 4
        self.tau_head = nn.Sequential(
            nn.Linear(hidden_dim, tau_inner),
            nn.GELU(),
            nn.Linear(tau_inner, 1),
            nn.Softplus(),
        )
        self.tau_eps = 1e-2

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_p = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def encode_cardinality(
        self,
        edge_card: torch.Tensor,          # [E]
    ) -> torch.Tensor:
        """c_e = MLP([log(1+|e|), |e|/d_max]) ∈ R^{E×d}

        每个 edge 对应一条 vector，规格书 §3.1。
        """
        d_max = edge_card.max().clamp_min(1.0)
        raw = torch.stack([
            torch.log1p(edge_card.to(torch.float32)),
            edge_card.to(torch.float32) / d_max,
        ], dim=-1)                                  # [E, 2]
        return self.card_mlp(raw)                    # [E, d]

    def forward(
        self,
        node_tokens: torch.Tensor,
        edge_tokens: torch.Tensor,
        incidence: torch.Tensor,
        c_e: Optional[torch.Tensor] = None,
        edge_card: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns *updated edge tokens* after CCA node→edge attention.

        Caller provides either (a) precomputed `c_e` from tokenize step
        or (b) raw `edge_card` so we compute c_e on-demand.
        """
        if node_tokens.numel() == 0 or edge_tokens.numel() == 0:
            return edge_tokens
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        n_idx, e_idx = sp.indices()
        num_edges = edge_tokens.size(0)
        d = self.hidden_dim
        device = node_tokens.device

        # ---- c_e / FiLM params / tau per edge ------------------------------
        if c_e is None:
            if edge_card is None:
                card = torch.zeros(num_edges, device=device, dtype=torch.float32)
                card.index_add_(0, e_idx, torch.ones_like(e_idx, dtype=torch.float32))
            else:
                card = edge_card
            c_e = self.encode_cardinality(card)
        # --- W3 ablation toggles (from HyperFounderV2Encoder global switches, propagated via c_e wrappers)
        # To keep the class signature clean the toggles are *not* stored here;
        # instead we support per-call overrides through attributes monkey-patched
        # onto the instance by the outer encoder.  See HyperFounderV2Encoder.__init__.
        ablate_card = bool(getattr(self, "ablate_card", False))
        ablate_film = bool(getattr(self, "ablate_film", False))
        ablate_tau  = bool(getattr(self, "ablate_tau", False))
        if ablate_card:
            c_e = torch.zeros_like(c_e)
        gamma = self.film_gamma(c_e)                          # [E, d]
        beta = self.film_beta(c_e)                            # [E, d]
        if ablate_film:
            gamma = torch.ones_like(gamma)
            beta = torch.zeros_like(beta)
        if ablate_tau:
            tau_e = torch.full((num_edges,), self.tau_eps, dtype=gamma.dtype, device=device)
        else:
            tau_e = self.tau_head(c_e).squeeze(-1) + self.tau_eps # [E]

        # ---- Pack edges: group node tokens by edge id ----------------------
        order = torch.argsort(e_idx, stable=True)
        e_s = e_idx[order]
        n_s = n_idx[order]
        _, counts = torch.unique_consecutive(e_s, return_counts=True)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        starts_exp = starts[:-1].repeat_interleave(counts)
        pos_in_group = torch.arange(e_s.numel(), device=device) - starts_exp

        Kmax = int(counts.max().item()) if counts.numel() else 0
        if Kmax == 0:
            return edge_tokens
        E = edge_tokens.size(0)
        H = self.num_heads
        Dh = self.head_dim
        # Q = edge token modulated by gamma/beta; K = node token; V = node token
        q = edge_tokens[e_s]                                  # [P, d]
        q = gamma[e_s] * q + beta[e_s]                        # FiLM
        k = self.k_proj(node_tokens[n_s])                     # [P, d]
        v = self.v_proj(node_tokens[n_s])                     # [P, d]
        q = self.q_proj(q).view(-1, H, Dh).transpose(0, 1)    # [H, P, Dh]
        k = k.view(-1, H, Dh).transpose(0, 1)                 # [H, P, Dh]
        v = v.view(-1, H, Dh).transpose(0, 1)                 # [H, P, Dh]

        # Per-pair score = sum_dh q[h,p]·k[h,p] / sqrt(Dh)  (but same pos_in_group edge id)
        # Simpler: pre-compute per-pair (q·k) then group softmax by e_s.
        per_pair_scores = (q * k).sum(dim=-1) / math.sqrt(Dh) # [H, P]
        # Temperature per edge: tau_e[e_s] — broadcast across H
        per_pair_tau = tau_e[e_s].unsqueeze(0).expand(H, -1)  # [H, P]
        per_pair_scores = per_pair_scores / per_pair_tau      # τ ∈ softmax denominator

        # Group-wise softmax by edge id.
        # Pack per_pair_scores [H, P] -> [H, E, Kmax] using sorted starts/pos_in_group,
        # apply stable F.softmax on dim=-1, then gather back to flat P axis.
        # This avoids index_put_ / index_add shape pitfalls with (slice, long_tensor).
        import torch.nn.functional as _F
        P = e_s.numel()
        h_idx = torch.arange(H, device=device).unsqueeze(1).expand(H, P)          # [H, P]
        e_idx_flat = e_s.unsqueeze(0).expand(H, P)                                # [H, P]
        p_flat = pos_in_group.unsqueeze(0).expand(H, P)                           # [H, P]
        NEG = torch.finfo(per_pair_scores.dtype).min
        packed = torch.full((H, E, Kmax), NEG, dtype=per_pair_scores.dtype, device=device)
        packed[h_idx, e_idx_flat, p_flat] = per_pair_scores
        packed_attn = _F.softmax(packed, dim=-1)
        attn = packed_attn[h_idx, e_idx_flat, p_flat]                             # [H, P]
        attn = self.dropout_p(attn)

        # Weighted sum of v: per edge ∈ [H, Dh]
        weighted_v = v * attn.unsqueeze(-1)                    # [H, P, Dh]
        per_edge_v = torch.zeros(H, E, Dh, dtype=weighted_v.dtype, device=device)
        per_edge_v.index_add_(1, e_s, weighted_v)              # [H, E, Dh]
        per_edge_v = per_edge_v.transpose(0, 1).contiguous().view(E, d)  # [E, d]
        new_e = self.out_proj(per_edge_v)
        return self.norm(edge_tokens + new_e)


def _edge_to_node_mean(
    node_tokens: torch.Tensor,
    edge_tokens: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    """Standard HGNN mean-pool edge→node back-propagation (规格书 §4.3, §1流程)."""
    if node_tokens.numel() == 0 or edge_tokens.numel() == 0:
        return node_tokens
    sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
    sp = sp.coalesce()
    n_idx, e_idx = sp.indices()
    N = node_tokens.size(0)
    d = node_tokens.size(-1)
    device = node_tokens.device
    sums, counts = _sparse_scatter_mean(edge_tokens[e_idx], n_idx, N)
    mean = sums / counts.clamp_min(1.0)
    out = torch.where(counts > 0, mean, torch.zeros(N, d, dtype=mean.dtype, device=device))
    return out


# ---------------------------------------------------------------------------
# 2. Module 2 — HCA: Hyperedge-Context Attention （§4）
# ---------------------------------------------------------------------------

class HyperedgeContextAttention(nn.Module):
    """Top-K overlap-aware edge↔edge cross-attention + back-write to nodes.

    规格书 §4：
      (1) 稀疏 S = Hᵀ·H；对称归一 ô = |∩| / √(|e_i||e_j|)；Top-K 邻居
      (2) logit(i,j) = Q_i·K_j /√d  +  bias(ô_ij)，bias 为 1 层 MLP 可学函数
      (3) E'_i = E_i + Σ α V_j；X'_v = Σ_{e∋v} E'_e / d_v
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        k: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) % num_heads ({num_heads}) != 0")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.k = k

        # Low-rank bottleneck 版 Q/K/V：控制 w4_no_hca 行参数增量 ≤ 10%
        inner = 8
        self.q_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )
        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )

        # Pairwise overlap → scalar bias.  1-hidden-layer MLP.
        self.bias_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout_p = nn.Dropout(dropout)
        self.norm_edge = nn.LayerNorm(hidden_dim)
        self.norm_node = nn.LayerNorm(hidden_dim)

    def topk_overlap_neighbors(
        self,
        incidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For each edge e_i, get top-K most-similar edges e_j (j ≠ i).

        Returns:
            nbr_src : [E, K]      edge ids i
            nbr_dst : [E, K]      edge ids j (K nearest neighbors of i)
            nbr_sim : [E, K, 1]   symmetric overlap ô
        """
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        if sp.dtype != torch.float32:
            sp = torch.sparse_coo_tensor(sp.indices(), sp.values().to(torch.float32), sp.size())
        E = sp.size(1)
        device = sp.device
        if E <= 1:
            return (torch.zeros(E, 0, dtype=torch.long, device=device),
                    torch.zeros(E, 0, dtype=torch.long, device=device),
                    torch.zeros(E, 0, 1, dtype=torch.float32, device=device))

        # Card(inality) + overlap
        idx = sp.indices()
        e_idx = idx[1]
        card = torch.zeros(E, dtype=torch.float32, device=device)
        card.index_add_(0, e_idx, torch.ones(e_idx.numel(), dtype=torch.float32, device=device))
        with torch.autocast(device_type=device.type, enabled=False):
            It = sp.transpose(0, 1).coalesce()
            overlap = torch.sparse.mm(It, sp).coalesce()
        oi, oj = overlap.indices()
        ov = overlap.values().to(torch.float32)

        # Exclude self
        keep = oi != oj
        oi, oj, ov = oi[keep], oj[keep], ov[keep]
        if oi.numel() == 0:
            return (torch.zeros(E, 0, dtype=torch.long, device=device),
                    torch.zeros(E, 0, dtype=torch.long, device=device),
                    torch.zeros(E, 0, 1, dtype=torch.float32, device=device))

        # Symmetric normalisation  ô = |∩| / √(|e_i| * |e_j|)
        denom = torch.sqrt((card[oi] * card[oj]).clamp_min(1e-6))
        sim = (ov / denom).clamp(0.0, 1.0)

        # Top-k per source edge.  Dense [E, E_topk_pad] for stability.
        # Order by (oi, -sim) and slice.
        order = torch.argsort(oi, stable=True)
        oi_s, oj_s, sim_s = oi[order], oj[order], sim[order]
        _, counts = torch.unique_consecutive(oi_s, return_counts=True)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])

        pad = min(int(self.k), int(counts.max().item()))
        src = torch.arange(E, dtype=torch.long, device=device).unsqueeze(-1).expand(E, pad)
        dst = torch.full((E, pad), -1, dtype=torch.long, device=device)
        sim_mat = torch.zeros(E, pad, dtype=sim_s.dtype, device=device)
        for k in range(starts.numel() - 1):
            lo = int(starts[k].item())
            hi = int(starts[k + 1].item())
            if lo == hi:
                continue
            row_id = int(oi_s[lo].item())
            sim_block = sim_s[lo:hi]
            oj_block = oj_s[lo:hi]
            kk = min(pad, sim_block.numel())
            topk = torch.topk(sim_block, kk)
            dst[row_id, :kk] = oj_block[topk.indices]
            sim_mat[row_id, :kk] = topk.values

        mask_valid = dst >= 0
        dst = dst.clamp_min(0)
        nbr_src = src
        nbr_dst = dst
        nbr_sim = sim_mat.unsqueeze(-1)
        nbr_sim = nbr_sim.masked_fill(~mask_valid.unsqueeze(-1), 0.0)
        return nbr_src, nbr_dst, nbr_sim

    def forward(
        self,
        node_tokens: torch.Tensor,
        edge_tokens: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (updated_node_tokens, updated_edge_tokens, nbr_src, nbr_dst, nbr_sim_raw).

        The Top-K overlap neighbor table (src/dst/sim) is exposed to upstream for two purposes:
          (a) reused directly as the hard-negative pool of membership contrast (spec §5.2),
          (b) logged / analysed as a structural saliency signal.
        """
        if edge_tokens.numel() == 0:
            E0 = 0
            dev = edge_tokens.device
            return (node_tokens, edge_tokens,
                    torch.zeros(E0, 0, dtype=torch.long, device=dev),
                    torch.zeros(E0, 0, dtype=torch.long, device=dev),
                    torch.zeros(E0, 0, 1, dtype=torch.float32, device=dev))
        device = edge_tokens.device
        E, d = edge_tokens.shape
        H = self.num_heads
        Dh = self.head_dim

        # (1) Top-K overlap graph
        nbr_src, nbr_dst, nbr_sim_raw = self.topk_overlap_neighbors(incidence)
        K_pad = nbr_dst.size(1)
        if K_pad == 0:
            # No overlap pairs: skip context, back-write is still identity.
            new_node = _edge_to_node_mean(node_tokens, edge_tokens, incidence)
            return (self.norm_node(node_tokens + new_node), edge_tokens,
                    nbr_src, nbr_dst, nbr_sim_raw)

        # Bias MLP over ô_ij — mapping [0,1] → R.  W4 ablate w4_no_bias → bias = 0.
        ablate_bias = bool(getattr(self, "ablate_bias", False))
        if ablate_bias:
            bias_flat = torch.zeros(E, K_pad, dtype=edge_tokens.dtype, device=device)
        else:
            bias_flat = self.bias_mlp(nbr_sim_raw.view(-1, 1)).view(E, K_pad)  # [E, K]
        key_pad_valid_mask = (nbr_dst >= 0).to(device=device, dtype=torch.bool)  # [E, K]

        # (2) QK·ᵀ/sqrt(Dh) + bias
        q_h = self.q_proj(edge_tokens).view(E, H, Dh).transpose(0, 1)        # [H, E, Dh]
        k_h = self.k_proj(edge_tokens[nbr_dst]).view(E, K_pad, H, Dh).permute(2, 0, 1, 3).contiguous()  # [H, E, K, Dh]
        v_h = self.v_proj(edge_tokens[nbr_dst]).view(E, K_pad, H, Dh).permute(2, 0, 1, 3).contiguous()  # [H, E, K, Dh]

        scores = (q_h.unsqueeze(2) * k_h).sum(dim=-1) / math.sqrt(Dh)       # [H, E, K]
        scores = scores + bias_flat.unsqueeze(0)                              # add bias (broadcast H)
        scores = scores.masked_fill(~key_pad_valid_mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout_p(attn)

        # (3) Σ α_ij V_j  →  E'_i = LN(E_i + W_proj · Σ)
        ctx = (attn.unsqueeze(-1) * v_h).sum(dim=2)                          # [H, E, Dh]
        ctx = ctx.transpose(0, 1).contiguous().view(E, d)
        new_e = self.norm_edge(edge_tokens + self.out_proj(ctx))

        # (3b) back-write nodes from context-enhanced edges
        new_n = _edge_to_node_mean(node_tokens, new_e, incidence)
        new_n = self.norm_node(node_tokens + new_n)
        return new_n, new_e, nbr_src, nbr_dst, nbr_sim_raw


# ---------------------------------------------------------------------------
# 3. Optional Module 3 — HOR (规格书 §5, 开关 use_hor)
# ---------------------------------------------------------------------------

class HigherOrderReadout(nn.Module):
    """2-clique co-membership readout; re-weight by Σ_e (|e|−2)·A_ue·A_ve.

    Set `use_hor=False` / num_layers=0  to disable entirely (§8 风险预案).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        topk: int = 64,
        pe_dim: int = 32,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) % num_heads ({num_heads}) != 0")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.topk = topk

        # 压缩版 Q/K/V：用 8 作 inner rank 控制单模块增量 ≤ encoder_without_hor × 6%
        inner = 8
        self.q_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )
        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )
        self.dropout_p = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_tokens: torch.Tensor,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        if node_tokens.numel() == 0:
            return node_tokens
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        if sp.dtype != torch.float32:
            sp = torch.sparse_coo_tensor(sp.indices(), sp.values().to(torch.float32), sp.size())
        N = sp.size(0)
        device = node_tokens.device
        d = self.hidden_dim
        H = self.num_heads
        Dh = self.head_dim

        # C = I·Iᵀ - diag  (2-clique co-membership)
        with torch.autocast(device_type=device.type, enabled=False):
            C = torch.sparse.mm(sp, sp.transpose(0, 1)).coalesce()
        c_idx = C.indices()
        c_val = C.values()
        keep = c_idx[0] != c_idx[1]
        c_idx = c_idx[:, keep]
        c_val = c_val[keep]
        if c_idx.numel() == 0:
            return self.norm(node_tokens)

        # Σ_e A_ue·A_ve·(|e|−2)  =  C[n,u]·(|e|−2)  summed over shared edges.
        # The sparse entry C_{n,u} already gives the shared-edge count.
        # Weight = (C_{n,u} − 2 · #_shared_giant_edges)...  for simplicity take
        # w_{n,u} = max(C_{n,u} − 1, 0), a proxy for (|e|−2) contribution since
        # each length-2 edge adds 1 to C but 0 to (|e|−2), subtract it out.
        w = (c_val - 1.0).clamp_min(0.0).to(c_val.dtype)

        # Top-K sparse filter per node (规格书 §5 "Top-K 稀疏化防热门节点主导")
        order = torch.argsort(c_idx[0], stable=True)
        s_idx = c_idx[0][order]
        u_idx = c_idx[1][order]
        w_s = w[order]
        _, counts = torch.unique_consecutive(s_idx, return_counts=True)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        pad = min(int(self.topk), int(counts.max().item()))
        if pad == 0:
            return self.norm(node_tokens)
        N_pad = pad
        u_mat = torch.full((N, N_pad), -1, dtype=torch.long, device=device)
        w_mat = torch.zeros((N, N_pad), dtype=w_s.dtype, device=device)
        # Note: starts only has length (num_non_empty_nodes + 1), not (N + 1).  Iterate
        # non-empty groups directly and recover node-id from the sorted s_idx.
        for k in range(starts.numel() - 1):
            i = int(s_idx[int(starts[k].item())].item())
            lo = int(starts[k].item())
            hi = int(starts[k + 1].item())
            if lo == hi:
                continue
            w_b = w_s[lo:hi]
            u_b = u_idx[lo:hi]
            kk = min(N_pad, w_b.numel())
            topk = torch.topk(w_b, kk)
            u_mat[i, :kk] = u_b[topk.indices]
            w_mat[i, :kk] = topk.values
        valid = u_mat >= 0
        u_safe = u_mat.clamp_min(0)

        # Normalise w per node  (§5 "归一化后作读出权重")
        w_sum = w_mat.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        w_norm = (w_mat / w_sum).masked_fill(~valid, 0.0)

        # Attention
        q_h = self.q_proj(node_tokens).view(N, H, Dh).transpose(0, 1)          # [H, N, Dh]
        k_ctx = node_tokens[u_safe]                                             # [N, K, d]
        v_ctx = node_tokens[u_safe]
        k_h = self.k_proj(k_ctx).view(N, N_pad, H, Dh).permute(2, 0, 1, 3)      # [H, N, K, Dh]
        v_h = self.v_proj(v_ctx).view(N, N_pad, H, Dh).permute(2, 0, 1, 3)      # [H, N, K, Dh]
        scores = (q_h.unsqueeze(2) * k_h).sum(dim=-1) / math.sqrt(Dh)
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout_p(attn)

        # Reweight by structure: attn_w = attn · w_norm  (broadcast H)，再归一
        structure = w_norm.unsqueeze(0).to(attn.dtype)                         # [1, N, K]
        attn_w = attn * structure
        attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn_w = torch.nan_to_num(attn_w, nan=0.0, posinf=0.0, neginf=0.0)

        ctx = (attn_w.unsqueeze(-1) * v_h).sum(dim=2)                           # [H, N, Dh]
        ctx = ctx.transpose(0, 1).contiguous().view(N, d)
        out = self.norm(node_tokens + self.out_proj(ctx))

        # §37 / §2: mask out isolated nodes that never saw any co-member (keeps
        # the HOR bug from reappearing — same scatter mask technique used earlier)
        has = (w_mat.sum(dim=-1, keepdim=True) > 0).to(out.dtype)
        out = out * has + node_tokens * (1 - has)
        return out


# ---------------------------------------------------------------------------
# 4. Full Encoder （规格书 §1 流程）
# ---------------------------------------------------------------------------

@dataclass
class V2EncoderConfig:
    in_dim: int
    hidden_dim: int = 256            # 导师规格书默认
    dropout: float = 0.1
    num_layers: int = 3              # 3 × { CCA + edge→node }   (§1)
    num_heads: int = 4
    pe_dim: int = 32                 # 替代旧的 structure_pe_dim
    hca_topk: int = 16               # §4 默认 Top-K 8-16
    use_hor: bool = True             # §5 开关，W1-2消融后决定是否留
    # --- W3 CCA 消融切换 (default = full spec) ---
    ablate_cca_card: bool = False    # w3_no_card: 去掉 card_mlp → c_e 全 0 向量
    ablate_cca_film: bool = False    # w3_no_film: 有 c_e，但 γ=1, β=0 恒等
    ablate_cca_tau:  bool = False    # w3_no_tau: 有 c_e + FiLM，但 τ=ε constant
    # --- W4 HCA 消融切换 (default = full spec) ---
    ablate_hca_bias: bool = False    # w4_no_bias: HCA bias_mlp 恒 0
    ablate_hca_full: bool = False    # w4_no_hca: 完全跳过 HCA (仅 LN pass-through)


class HyperFounderV2Encoder(nn.Module):
    """Domain-agnostic unified hypergraph encoder.

    Signature: forward(x: [N, in_dim], incidence: sparse [N, E]) -> node_emb: [N, hidden_dim]
    """

    def __init__(self, config: V2EncoderConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        # In-proj
        self.in_proj = nn.Linear(config.in_dim, d)
        self.edge_in_proj = nn.Linear(config.in_dim, d)

        # Structure PE (node: 6-dim raw = deg_norm + 5 RW steps; edge: 2-dim = card_norm + overlap_mean)
        pe_dim = max(int(config.pe_dim), 1)
        self.pe_dim = pe_dim
        self.node_pe_proj = nn.Linear(1 + 5, pe_dim)
        self.edge_pe_proj = nn.Linear(2, pe_dim)
        self.node_pe_align = nn.Linear(pe_dim, d) if pe_dim != d else nn.Identity()
        self.edge_pe_align = nn.Linear(pe_dim, d) if pe_dim != d else nn.Identity()

        self.ablate_cca_card = bool(config.ablate_cca_card)
        self.ablate_cca_film = bool(config.ablate_cca_film)
        self.ablate_cca_tau = bool(config.ablate_cca_tau)
        self.ablate_hca_bias = bool(config.ablate_hca_bias)
        self.ablate_hca_full = bool(config.ablate_hca_full)

        # Stacked CCA + edge→node
        self.cca_layers = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        for _ in range(config.num_layers):
            cca = CardinalityConditionedAttention(d, config.num_heads, dropout=config.dropout, pe_dim=int(config.pe_dim))
            cca.ablate_card = self.ablate_cca_card
            cca.ablate_film = self.ablate_cca_film
            cca.ablate_tau  = self.ablate_cca_tau
            self.cca_layers.append(cca)
            self.node_norms.append(nn.LayerNorm(d))

        # HCA single module (§1 一次 Top-K 上下文增强) — W4 ablations wired via instance attrs
        self.hca = HyperedgeContextAttention(d, config.num_heads, k=config.hca_topk, dropout=config.dropout)
        self.hca.ablate_bias = self.ablate_hca_bias

        # Optional HOR
        self.hor = (HigherOrderReadout(d, config.num_heads, dropout=config.dropout,
                                        topk=max(16, int(getattr(config, "hca_topk", 16)) * 4),
                                        pe_dim=int(config.pe_dim))
                     if config.use_hor else None)
        self.out_norm = nn.LayerNorm(d)

    def _structure_pe(self, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        N, E = sp.size(0), sp.size(1)
        device = sp.device
        idx = sp.indices()
        deg = torch.zeros(N, device=device, dtype=torch.float32)
        deg.index_add_(0, idx[0], torch.ones(idx.size(1), device=device, dtype=torch.float32))
        card = torch.zeros(E, device=device, dtype=torch.float32)
        card.index_add_(0, idx[1], torch.ones(idx.size(1), device=device, dtype=torch.float32))
        deg_norm = deg / deg.max().clamp_min(1e-8)
        card_norm = card / card.max().clamp_min(1e-8)

        rw = hypergraph_rw_pe(incidence, num_steps=5)                      # [N, 5]
        node_raw = torch.cat([deg_norm.unsqueeze(-1).to(rw.dtype), rw], dim=-1)  # [N, 6]

        overlap = overlap_coefficient(incidence).unsqueeze(-1)              # [E, 1]
        edge_raw = torch.cat([card_norm.unsqueeze(-1).to(overlap.dtype), overlap], dim=-1)  # [E, 2]

        node_pe = torch.nan_to_num(self.node_pe_proj(node_raw), nan=0.0)
        edge_pe = torch.nan_to_num(self.edge_pe_proj(edge_raw), nan=0.0)
        return node_pe, edge_pe

    def _tokenize_edges(
        self,
        x_compute: torch.Tensor,
        incidence: torch.Tensor,
        edge_card: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """§2 Module 0:  E = σ( mean_{v∈e}(X_v·W_p) + PE_card_proj(c_e_raw) )

        We re-use structure_pe *raw card_norm* as the cardinaliy input to card_mlp
        via the first CCA layer (c_e is passed through to CCA).

        If caller supplies explicit ``edge_card`` (shape [E], long), the card computation
        inside the helper is skipped — useful for smoke-tests with synthetic cardinalities
        and to avoid re-counting when the caller already has the vector.
        """
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        n_idx, e_idx = sp.indices()
        E = sp.size(1)
        device = x_compute.device
        if n_idx.numel() == 0:
            zero_card = torch.zeros(E, dtype=x_compute.dtype, device=device)
            return x_compute.new_zeros((E, x_compute.size(-1))), zero_card, self.cca_layers[0].encode_cardinality(zero_card)
        sum_per_e, cnt_per_e = _sparse_scatter_mean(x_compute[n_idx], e_idx, E)
        mean_xe = sum_per_e / cnt_per_e.clamp_min(1.0)
        if edge_card is None:
            card = cnt_per_e.squeeze(-1)
        else:
            card = edge_card.to(device=device, dtype=x_compute.dtype)
        return mean_xe, card, self.cca_layers[0].encode_cardinality(card)

    def forward(
        self,
        x: torch.Tensor,
        incidence: torch.Tensor,
        edge_cardinalities: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a hypergraph (per导师 spec §1 flow).

        Args
        ----
        x : [N, in_dim]              node features
        incidence : sparse [N, E]    node-hyperedge incidence matrix (H)
        edge_cardinalities : [E]     optional, per-edge |e|; if omitted, computed from H.

        Returns
        -------
        node_tokens : [N, hidden_dim]   final node embeddings, dtype == input x.dtype
        edge_tokens : [E, hidden_dim]   final hyperedge embeddings, dtype == input x.dtype
        hca_nbr_src : [E, hca_topk]    Top-K overlap src edge ids (from HCA, reuse for pretext)
        hca_nbr_dst : [E, hca_topk]    Top-K overlap dst edge ids
        hca_nbr_sim : [E, hca_topk, 1] overlap ô values (clamped [0,1])
        """
        in_dtype = x.dtype
        compute_dtype = torch.float32
        x_c = x.to(compute_dtype)

        with torch.autocast(device_type=x.device.type, enabled=False):
            node_pe, edge_pe = self._structure_pe(incidence)
            node_tokens = self.in_proj(x_c) + self.node_pe_align(node_pe.to(compute_dtype))

            # Module 0: tokenize (optionally use caller-provided cardinalities)
            edge_mean_x, edge_card, c_e = self._tokenize_edges(x_c, incidence, edge_card=edge_cardinalities)
            edge_tokens = torch.sigmoid(self.edge_in_proj(edge_mean_x) +
                                        self.edge_pe_align(edge_pe.to(compute_dtype)))  # §2  σ(·)

            # 3 × { CCA(node→edge) → edge→node mean }
            for lyr, cca in enumerate(self.cca_layers):
                edge_tokens = cca(node_tokens, edge_tokens, incidence,
                                  c_e=c_e if lyr == 0 else None,
                                  edge_card=edge_card)
                new_n = _edge_to_node_mean(node_tokens, edge_tokens, incidence)
                node_tokens = self.node_norms[lyr](node_tokens + new_n)

            # 1 × HCA (edge context + node back-write) — expose Top-K table upstream
            # W4 ablate_hca_full: skip entirely (pass-through LN), still compute overlap
            # table because downstream membership pretext expects non-empty tensors.
            if self.ablate_hca_full:
                new_n = _edge_to_node_mean(node_tokens, edge_tokens, incidence)
                node_tokens = self.hca.norm_node(node_tokens + new_n)
                edge_tokens = self.hca.norm_edge(edge_tokens)
                nbr_src, nbr_dst, nbr_sim = self.hca.topk_overlap_neighbors(incidence)
                hca_nbr_src, hca_nbr_dst, hca_nbr_sim = nbr_src, nbr_dst, nbr_sim
            else:
                node_tokens, edge_tokens, hca_nbr_src, hca_nbr_dst, hca_nbr_sim = \
                    self.hca(node_tokens, edge_tokens, incidence)

            # Optional HOR
            if self.hor is not None:
                node_tokens = self.hor(node_tokens, incidence)

            node_tokens = self.out_norm(node_tokens)
            node_tokens = torch.nan_to_num(node_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        return (node_tokens.to(in_dtype), edge_tokens.to(in_dtype),
                hca_nbr_src, hca_nbr_dst, hca_nbr_sim)


def build_encoder_v2(
    in_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.1,
    pe_dim: int = 32,
    hca_topk: int = 16,
    use_hor: bool = True,
    ablate_cca_card: bool = False,
    ablate_cca_film: bool = False,
    ablate_cca_tau: bool = False,
    ablate_hca_bias: bool = False,
    ablate_hca_full: bool = False,
    **_ignore_kwargs,
) -> HyperFounderV2Encoder:
    """Construct encoder matching signatures from V2 yaml / trainer.

    参数全部来自规格书 §3-§5；旧的 structure_pe_dim/hedg_tau 已移除。
    新增 W3 × 4 行 + W4 × 3 行消融开关 (默认全 False = 完整 spec)。
    """
    return HyperFounderV2Encoder(V2EncoderConfig(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        pe_dim=pe_dim,
        hca_topk=hca_topk,
        use_hor=use_hor,
        ablate_cca_card=ablate_cca_card,
        ablate_cca_film=ablate_cca_film,
        ablate_cca_tau=ablate_cca_tau,
        ablate_hca_bias=ablate_hca_bias,
        ablate_hca_full=ablate_hca_full,
    ))
