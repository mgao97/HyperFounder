"""v2 pretext tasks + 三参数 Kendall homoscedastic uncertainty weighting.

严格按 `v2/encoder-design-spec.md §5` + `V1 Kendall UW` 兼容设计：

  1. **edge_masked_recon** (MLM on hyperedges)
     - 随机遮罩 15% 的超边（不是节点；边信息比节点在跨域上更稳定）
     - 遮罩 token：edge_emb[masked_idx] 通过 2-layer MLP 预测该边原 mean-node-feature
     - 训练信号 = F.mse_loss(pred, target)

  2. **node_edge_membership_contrast**
     - 三元组 (n, e⁺, e⁻) InfoNCE，τ = 0.2（规格书 §5 默认）
     - 正样本：所有 I[n,e]=1 的真实 (n,e) 对
     - 负样本：对每条正边 e⁺，抽样 2 条 e⁻，70% 来自 HCA 同一个 Top-K overlap 邻居
       (复用 HCA overlap 邻居表一次计算两处受益)，30% 随机
     - head = Bilinear scorer m(N_v, E_e)

  3. **edge_dualview_contrast**
     - 对 incidence 做两种 augmentation：
         view1：node-drop ρ_n = 0.15；边-落 ρ_e = 0.1
         view2：node-drop ρ_n = 0.15；边-落 ρ_e = 0.1，独立 seed
     - 每个视图分别通过 encoder → edge projector (2-layer MLP → L2-normed)
     - Symmetric InfoNCE：L = 0.5 * CE(view1→view2) + 0.5 * CE(view2→view1)，τ = 0.5

  三任务权重 = Kendall Homoscedastic Uncertainty（3 参数 log_σ）
     L_total = Σ_i (1/(2σ_i²)) * L_i + log σ_i
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0. Kendall Uncertainty Weights (3 tasks，规格书§5继承V1 UW 简化版)
# ---------------------------------------------------------------------------

class KendallUncertaintyWeights(torch.nn.Module):
    """Three-parameter version of Kendall CVPR'18 homoscedastic uncertainty weighting.

    ```
    L_total = Σ_i [ exp(-s_i) * L_i + 0.5 * s_i ]   where s_i = log(σ_i²).
    ```
    0.5 系数对齐论文原式（s = log σ² → 权重 1/(2σ²)）。
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.log_sigma = torch.nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == self.log_sigma.numel(), "loss list must match num_tasks"
        total = 0.0
        for s_i, L_i in zip(self.log_sigma, losses):
            total = total + torch.exp(-s_i) * L_i + 0.5 * s_i
        return total

    @torch.no_grad()
    def task_weights(self) -> List[float]:
        """Return [w_1, w_2, w_3] where w_i = 1 / (2σ_i²). Useful for logging."""
        return [float((0.5 * torch.exp(-s_i)).item()) for s_i in self.log_sigma]


class ResidualUncertaintyWeights(torch.nn.Module):
    """Residual-style heteroscedastic weighting used in T4.

    L_total = Σ_i [ exp(-s_i) * L_i + s_i ]
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == self.log_var.numel(), "loss list must match num_tasks"
        total = 0.0
        for s_i, L_i in zip(self.log_var, losses):
            total = total + torch.exp(-s_i) * L_i + s_i
        return total

    @torch.no_grad()
    def task_weights(self) -> List[float]:
        return [float(torch.exp(-s_i).item()) for s_i in self.log_var]


class VariationalBottleneck(torch.nn.Module):
    """Minimal VIB bottleneck for T1.

    Input/output keep the same hidden size; KL is averaged over tokens.
    """

    def __init__(self, hidden_dim: int, latent_dim: Optional[int] = None):
        super().__init__()
        latent = int(latent_dim or hidden_dim)
        self.mu_proj = torch.nn.Linear(hidden_dim, latent)
        self.logvar_proj = torch.nn.Linear(hidden_dim, latent)
        self.out_proj = torch.nn.Identity() if latent == hidden_dim else torch.nn.Linear(latent, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            return x, x.new_tensor(0.0)
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x).clamp(min=-8.0, max=8.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        out = self.out_proj(z)
        kl_per_token = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
        kl = kl_per_token.mean()
        return out, kl


# ---------------------------------------------------------------------------
# 1. Pretext batch & augmentations
# ---------------------------------------------------------------------------

@dataclass
class PretextBatchV2:
    edge_mask_idx: torch.Tensor           # [B_mlm]
    edge_mask_target: torch.Tensor        # [B_mlm, in_dim]   — target mean-node-feature
    membership_pos: torch.Tensor          # [M, 2]       — (node_idx, edge_idx)
    membership_neg: torch.Tensor          # [M * N_neg, 2]
    # dual-view augmentations
    incidence_view1: torch.Tensor         # sparse [N, E]
    incidence_view2: torch.Tensor         # sparse [N, E]


def _sparse_drop_rows_cols(
    incidence: torch.Tensor,
    node_drop_p: float = 0.15,
    edge_drop_p: float = 0.10,
    seed: int = 0,
) -> torch.Tensor:
    """Simultaneous node-drop + edge-drop over a sparse incidence.

    规格书 §5.3：dual-view augmentation,  ρ_n=0.15, ρ_e=0.10.
    """
    sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
    sp = sp.coalesce()
    N, E = sp.size(0), sp.size(1)
    device = sp.device
    idx = sp.indices()
    n_idx, e_idx = idx[0], idx[1]

    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    if device.type == "cpu":
        keep_nodes = torch.rand(N, generator=g_cpu, device=device) >= node_drop_p
        keep_edges = torch.rand(E, generator=g_cpu, device=device) >= edge_drop_p
    else:
        keep_nodes = torch.rand(N, generator=g_cpu, device="cpu").to(device) >= node_drop_p
        keep_edges = torch.rand(E, generator=g_cpu, device="cpu").to(device) >= edge_drop_p

    ok = keep_nodes[n_idx] & keep_edges[e_idx]
    new_n = n_idx[ok]
    new_e = e_idx[ok]
    vals = torch.ones(new_n.numel(), dtype=sp.values().dtype, device=device)
    return torch.sparse_coo_tensor(torch.stack([new_n, new_e], dim=0), vals, (N, E)).coalesce()


def _edge_mean_features(
    x: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    """Target for edge MLM recon: mean of member-node features. 形状 [E, in_dim]."""
    sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
    sp = sp.coalesce()
    n_idx, e_idx = sp.indices()
    E = sp.size(1)
    device = x.device
    if n_idx.numel() == 0:
        return x.new_zeros((E, x.size(-1)))
    d = x.size(-1)
    sums = torch.zeros(E, d, device=device, dtype=x.dtype)
    cnts = torch.zeros(E, 1, device=device, dtype=x.dtype)
    sums.index_add_(0, e_idx, x[n_idx])
    cnts.index_add_(0, e_idx, torch.ones(e_idx.numel(), 1, device=device, dtype=x.dtype))
    return sums / cnts.clamp_min(1.0)


def _sample_membership_triples(
    incidence: torch.Tensor,
    hca_neighbor_table: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    num_negatives: int = 2,
    hard_prob: float = 0.7,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample (node, pos_edge, neg_edge) triples.

    If `hca_neighbor_table` is provided (from HCA.topk_overlap_neighbors), negatives are drawn
    with hard_prob from the pos-edge's Top-K overlap pool.  Otherwise 100% random negatives.
    """
    sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
    sp = sp.coalesce()
    num_nodes, num_edges = sp.size(0), sp.size(1)
    device = sp.device
    n_in, e_in = sp.indices()
    M = n_in.numel()
    if M == 0 or num_edges == 0:
        z = torch.zeros((0, 2), dtype=torch.long, device=device)
        return z, z.clone()

    # Positives: all real incidence pairs.
    pos = torch.stack([n_in, e_in], dim=1)                           # [M, 2]

    # Hard pool: pad [num_edges, K_max] from HCA neighbor dst table (exclude same edge)
    hard_pool_padded = None
    hard_Kmax = 0
    if hca_neighbor_table is not None:
        _, nbr_dst, _ = hca_neighbor_table
        if nbr_dst.numel():
            hard_pool_padded = nbr_dst                                    # [E, K]
            hard_Kmax = hard_pool_padded.size(1)

    # Repeat each positive num_negatives times to form one row per candidate negative.
    pos_rep = pos.repeat_interleave(num_negatives, dim=0)            # [M*N, 2]
    N_total = pos_rep.size(0)

    # Sample negatives
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    rand_n = torch.rand(N_total, generator=g_cpu, device="cpu").to(device)
    is_hard = (rand_n < hard_prob)

    neg_edges = torch.randint(0, num_edges, (N_total,), generator=g_cpu, device="cpu").to(device)

    if hard_pool_padded is not None and hard_Kmax > 0:
        rand_idx = torch.randint(0, hard_Kmax, (N_total,), generator=g_cpu, device="cpu").to(device)
        pos_edges_rep = pos_rep[:, 1]
        hard_choice = hard_pool_padded[pos_edges_rep.long(), rand_idx.long()]
        valid = hard_choice >= 0
        use_hard = is_hard & valid
        neg_edges = torch.where(use_hard, hard_choice.clamp_min(0), neg_edges)

    same = neg_edges == pos_rep[:, 1]
    if bool(same.any()):
        g2 = torch.Generator(device="cpu").manual_seed(seed + 7)
        repl = torch.randint(0, num_edges, (N_total,), generator=g2, device="cpu").to(device)
        neg_edges = torch.where(same, repl, neg_edges)

    neg = torch.stack([pos_rep[:, 0], neg_edges], dim=1)               # [M*N, 2]
    return pos, neg


def build_pretext_batch(
    x: torch.Tensor,
    incidence: torch.Tensor,
    edge_mlm_rate: float = 0.15,
    node_drop_p: float = 0.15,
    edge_drop_p: float = 0.10,
    membership_num_negatives: int = 2,
    membership_hard_prob: float = 0.7,
    hca_neighbor_table: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    seed: int = 0,
) -> PretextBatchV2:
    device = x.device
    E = (incidence.size(1) if incidence.is_sparse else incidence.to_sparse_coo().size(1))
    # (1) edge MLM
    g_mlm = torch.Generator(device="cpu").manual_seed(seed + 1)
    B_mlm = max(1, int(E * edge_mlm_rate))
    edge_mask_idx = torch.randperm(E, generator=g_mlm, device="cpu")[:B_mlm].to(device)
    edge_means = _edge_mean_features(x, incidence)
    edge_mask_target = edge_means[edge_mask_idx]

    # (2) membership triples (re-using HCA neighbors)
    membership_pos, membership_neg = _sample_membership_triples(
        incidence,
        hca_neighbor_table,
        num_negatives=membership_num_negatives,
        hard_prob=membership_hard_prob,
        seed=seed + 3,
    )

    # (3) dual view
    incidence_view1 = _sparse_drop_rows_cols(incidence, node_drop_p, edge_drop_p, seed=seed + 5)
    incidence_view2 = _sparse_drop_rows_cols(incidence, node_drop_p, edge_drop_p, seed=seed + 11)

    return PretextBatchV2(
        edge_mask_idx=edge_mask_idx,
        edge_mask_target=edge_mask_target,
        membership_pos=membership_pos,
        membership_neg=membership_neg,
        incidence_view1=incidence_view1,
        incidence_view2=incidence_view2,
    )


# ---------------------------------------------------------------------------
# 2. Three pretext losses
# ---------------------------------------------------------------------------

def edge_mlm_loss(
    edge_emb: torch.Tensor,
    batch: PretextBatchV2,
    head,
) -> torch.Tensor:
    """Edge MLM reconstruction loss (规格书 §5.1) — MSE on a 2-layer decoder."""
    if batch.edge_mask_idx.numel() == 0:
        return edge_emb.new_tensor(0.0)
    pred = head(edge_emb[batch.edge_mask_idx])
    return F.mse_loss(pred, batch.edge_mask_target.to(pred.dtype))


def node_edge_membership_loss(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    batch: PretextBatchV2,
    head,
    tau: float = 0.2,
) -> torch.Tensor:
    """Triplet InfoNCE for (n, e⁺) vs (n, e⁻).  Spec §5.2, τ=0.2 fixed default.

    Implementation:
        For each positive i → scores_i^+ = head(node_emb[n_i], edge_emb[e_i^+]).
        Its N_neg paired negatives → j=0..N_neg-1: scores_{i,j}^- = head(node_emb[n_i], edge_emb[e_{i,j}^-]).
        InfoNCE row-wise: logits[i] = [s_i^+, s_{i,0}^-, ..., s_{i,N-}^-] / τ.
        Label = 0 (positive at index 0).
    """
    pos = batch.membership_pos
    neg = batch.membership_neg
    if pos.numel() == 0 or neg.numel() == 0:
        return node_emb.new_tensor(0.0)
    M = pos.size(0)
    N_neg = neg.size(0) // M
    if N_neg == 0 or neg.size(0) % M != 0:
        # Fall back: match by truncating to common length
        n = min(pos.size(0), neg.size(0))
        pos_s = head(node_emb[pos[:n, 0]], edge_emb[pos[:n, 1]])
        neg_s = head(node_emb[neg[:n, 0]], edge_emb[neg[:n, 1]])
        logits = torch.stack([pos_s, neg_s], dim=1) / tau
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    s_pos = head(node_emb[pos[:, 0]], edge_emb[pos[:, 1]])             # [M]
    s_neg = head(node_emb[neg[:, 0]], edge_emb[neg[:, 1]])             # [M*N]
    s_neg = s_neg.view(M, N_neg)
    logits = torch.cat([s_pos.unsqueeze(-1), s_neg], dim=-1) / tau     # [M, 1+N]
    labels = torch.zeros(M, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def edge_dualview_contrast_loss(
    edge_emb_view1: torch.Tensor,
    edge_emb_view2: torch.Tensor,
    projector,
    tau: float = 0.5,
) -> torch.Tensor:
    """Symmetric InfoNCE on edge-level embeddings across node/edge-drop augmentation views (§5.3).

    projector maps [E, d] → [E, proj_dim] with L2-normalization.  Only trains on edges that
    survive both views (union is OK for contrastive — we just keep first min(E1,E2) rows to
    preserve alignment between same original edge id).  Encoder already outputs |E| rows so
    positional identity is preserved.
    """
    if edge_emb_view1.numel() == 0 or edge_emb_view2.numel() == 0:
        return edge_emb_view1.new_tensor(0.0)
    E_align = min(edge_emb_view1.size(0), edge_emb_view2.size(0))
    if E_align < 2:
        return edge_emb_view1.new_tensor(0.0)
    z1 = F.normalize(projector(edge_emb_view1[:E_align]), dim=-1)
    z2 = F.normalize(projector(edge_emb_view2[:E_align]), dim=-1)
    sim = z1 @ z2.transpose(0, 1) / tau
    labels = torch.arange(E_align, device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.transpose(0, 1), labels))
