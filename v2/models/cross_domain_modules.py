from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


def _dense_incidence(incidence: torch.Tensor) -> torch.Tensor:
    if incidence.is_sparse:
        return incidence.to_dense()
    return incidence


def _build_groups(keys: torch.Tensor, values: torch.Tensor, num_keys: int) -> list:
    """Group `values` by `keys` into a list of length `num_keys`.

    Returns a list where out[k] is a 1-D tensor of all `values` whose key == k
    (or None when key k is absent). Uses a sort + unique_consecutive split so it
    scales to tens of thousands of keys without any dense [N, E] allocation.
    """
    if keys.numel() == 0:
        return [None] * num_keys
    order = torch.argsort(keys, stable=True)
    keys_s = keys[order]
    vals_s = values[order]
    unique, counts = torch.unique_consecutive(keys_s, return_counts=True)
    splits = torch.split(vals_s, counts.tolist())
    out = [None] * num_keys
    for u, g in zip(unique.tolist(), splits):
        out[u] = g
    return out


class CrossDomainProjector(nn.Module):
    def __init__(self, d_out: int):
        super().__init__()
        self.d_out = d_out
        self.projectors = nn.ModuleDict()
        self.feature_types: Dict[str, str] = {}

    def register_domain(self, domain_id: str | int, d_in: int, feature_type: str) -> None:
        key = str(domain_id)
        if key in self.projectors:
            return
        if feature_type == "text":
            proj: nn.Module = nn.Sequential(
                nn.Linear(d_in, self.d_out * 2),
                nn.GELU(),
                nn.Linear(self.d_out * 2, self.d_out),
            )
        elif feature_type == "image":
            proj = nn.Sequential(
                nn.Linear(d_in, self.d_out * 2),
                nn.ReLU(),
                nn.Linear(self.d_out * 2, self.d_out),
            )
        elif feature_type == "categorical":
            proj = nn.Embedding(d_in, self.d_out)
        else:
            proj = nn.Linear(d_in, self.d_out)
        self.projectors[key] = proj
        self.feature_types[key] = feature_type

    def forward(self, x: torch.Tensor, domain_id: str | int) -> torch.Tensor:
        key = str(domain_id)
        if key not in self.projectors:
            raise KeyError(f"Domain projector '{key}' has not been registered.")
        proj = self.projectors[key]
        param = next(proj.parameters(), None)
        if param is not None and param.device != x.device:
            proj = proj.to(x.device)
            self.projectors[key] = proj
        if isinstance(proj, nn.Embedding):
            x_proj = proj(x.long())
        else:
            x_proj = proj(x)
        return torch.nan_to_num(F.layer_norm(x_proj, [self.d_out]), nan=0.0, posinf=0.0, neginf=0.0)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class DomainAlignmentLoss(nn.Module):
    def __init__(self, d_model: int, num_domains: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_domains)
        self.grl = GradientReversalLayer(alpha=1.0)

    def forward(self, h: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0:
            return h.new_tensor(0.0)
        logits = self.classifier(self.grl(h))
        return F.cross_entropy(logits, domain_labels.long())


def overlap_coefficient(incidence: torch.Tensor) -> torch.Tensor:
    # Memory-efficient, *fully sparse* overlap coefficient.
    #
    # The previous dense version built `bt_b = dense.T @ dense` (O(E^2) memory)
    # and even the chunked rewrite still relied on `_dense_incidence` which
    # materialises the full [N, E] incidence matrix -> OOM on large hypergraphs
    # (coauthorship_dblp: 19800 x 27800 -> ~2.2 GB).
    #
    # Here we stay sparse end-to-end. `It @ I` is a sparse [E, E] matrix whose
    # (i, j) entry equals |e_i ∩ e_j|. We divide each entry by min(|e_i|,|e_j|)
    # and scatter-add the per-edge numerator, so peak memory is O(nnz) and we
    # never allocate a dense [N, E] or [E, E] tensor.
    if not incidence.is_sparse:
        incidence = incidence.to_sparse_coo()
    s = incidence.coalesce()
    # torch.sparse.mm on CUDA supports only fp32/fp64; cast before the
    # sparse matrix product so a BFloat16 sparse incidence does not crash.
    if s.dtype != torch.float32:
        s = torch.sparse_coo_tensor(s.indices(), s.values().to(torch.float32), s.size())
    num_edges = s.size(1)
    if num_edges == 0:
        return s.values().new_zeros((0,))
    idx = s.indices()
    col = idx[1]
    device = idx.device
    c = torch.zeros(num_edges, device=device, dtype=torch.float32)
    c.scatter_add_(0, col, torch.ones(col.numel(), device=device))
    It = s.transpose(0, 1).coalesce()  # [E, N] sparse
    # Disable autocast: finetune runs under bf16 autocast. The incidence here
    # is a sampled sub-hypergraph (N<=256, E<=128), so a dense [E, E] product
    # is cheap and avoids the `aten::sparse_dim` op that some torch builds
    # (incl. the current 1.13.1+cu117) fail to register on any backend.
    with torch.autocast(device_type=device.type, enabled=False):
        It_d = It.to_dense().to(torch.float32)
        s_d = s.to_dense().to(torch.float32)
        EI = It_d @ s_d                      # [E, E] dense, EI[i, j] = |e_i ∩ e_j|
    ei = EI.nonzero().t()
    ev = EI[ei[0], ei[1]].to(torch.float32)
    c1 = c[ei[0]]
    c2 = c[ei[1]]
    min_c = torch.minimum(c1, c2).clamp_min(1e-8)
    div = ev / min_c
    numerator = torch.zeros(num_edges, device=device, dtype=torch.float32)
    numerator.scatter_add_(0, ei[0], div)
    overlap_mean = numerator / num_edges
    return torch.nan_to_num(overlap_mean, nan=0.0, posinf=0.0, neginf=0.0)


def hypergraph_rw_pe(incidence: torch.Tensor, num_steps: int = 5,
                      chunk_size: int = 256,
                      mem_budget_bytes: int = 256 * 1024 * 1024) -> torch.Tensor:
    # Fully sparse random-walk PE. No dense [N, E] matrix is ever materialised.
    #
    # P = D_v^-1 @ I @ D_e^-1 @ I^T. We multiply a chunk of identity rows by P
    # iteratively (dense@sparse -> dense), keeping only the diagonal of P^k for
    # each chunk. Memory peak is O(chunk_size * max(num_nodes, num_edges)).
    #
    # The dominant cost is the `torch.sparse.mm` kernel launches (2 per step, per
    # chunk). We therefore pick the *largest* chunk that fits the memory budget
    # instead of the tiny default 256, which collapses the number of kernel
    # launches (e.g. cora: 11 chunks -> 1; dblp: ~195 -> ~39) without changing a
    # single number -- chunks compute their diagonals independently.
    if not incidence.is_sparse:
        incidence = incidence.to_sparse_coo()
    s = incidence.coalesce()
    # torch.sparse.mm only supports fp32/fp64 on CUDA; the caller may feed a
    # BFloat16 sparse incidence (e.g. mixed-precision finetune). Compute in
    # float32 and cast the result back to the input dtype to stay consistent
    # with the surrounding tensors.
    in_dtype = s.dtype
    if in_dtype != torch.float32:
        # Rebuild explicitly: COO .to(dtype) does not reliably recast the
        # values tensor on some torch builds, so reconstruct from indices.
        s = torch.sparse_coo_tensor(s.indices(), s.values().to(torch.float32), s.size())
    num_nodes = s.size(0)
    num_edges = s.size(1)
    if num_nodes == 0:
        return s.values().new_zeros((0, num_steps), dtype=in_dtype)
    device = s.device
    node_degree = torch.sparse.sum(s, dim=1).to_dense().clamp_min(1.0)
    edge_degree = torch.sparse.sum(s, dim=0).to_dense().clamp_min(1.0)
    node_degree_inv = 1.0 / node_degree  # [num_nodes]
    edge_degree_inv = 1.0 / edge_degree  # [num_edges]

    rw_features = torch.zeros(num_nodes, num_steps, device=device, dtype=torch.float32)
    # Largest chunk whose dense [chunk, N] working buffer stays within budget.
    max_chunk_mem = max(1, int(mem_budget_bytes) // (num_nodes * 4))
    chunk_size = min(num_nodes, max(int(chunk_size), max_chunk_mem))
    s_t = s.transpose(0, 1).coalesce()  # [E, N] sparse (for cur @ s == s_t^T @ cur^T)

    for cs in range(0, num_nodes, chunk_size):
        ce = min(cs + chunk_size, num_nodes)
        cur_chunk = ce - cs
        chunk_idx = torch.arange(cur_chunk, device=device)
        cur = torch.zeros(cur_chunk, num_nodes, device=device, dtype=torch.float32)
        cur[chunk_idx, cs + chunk_idx] = 1.0

        for k in range(num_steps):
            cur = cur * node_degree_inv.unsqueeze(0)
            # NOTE: finetune runs under autocast(bf16); autocast forces matmul
            # ops (incl. sparse.mm) to BFloat16, which CUDA sparse kernels
            # reject. Disable autocast so the sparse product stays float32.
            with torch.autocast(device_type=device.type, enabled=False):
                # cur @ s  ==  (s_t^T @ cur^T)^T ; use sparse-left matmul (CPU/CUDA safe)
                cur = torch.sparse.mm(s_t, cur.t()).t()  # [chunk, E]
            cur = cur * edge_degree_inv.unsqueeze(0)
            with torch.autocast(device_type=device.type, enabled=False):
                # cur @ It == (It^T @ cur^T)^T == (s @ cur^T)^T
                cur = torch.sparse.mm(s, cur.t()).t()    # [chunk, N]
            rw_features[cs:ce, k] = cur[chunk_idx, cs + chunk_idx]

    rw_features = torch.nan_to_num(rw_features, nan=0.0, posinf=0.0, neginf=0.0)
    return rw_features.to(in_dtype) if in_dtype != torch.float32 else rw_features


class CrossDomainStructuralPEModule(nn.Module):
    def __init__(self, d_pe: int, num_rw_steps: int = 5):
        super().__init__()
        self.d_pe = d_pe
        self.num_rw_steps = num_rw_steps
        self.node_mlp = nn.Sequential(
            nn.Linear(1 + num_rw_steps, d_pe * 2),
            nn.ReLU(),
            nn.Linear(d_pe * 2, d_pe),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, d_pe * 2),
            nn.ReLU(),
            nn.Linear(d_pe * 2, d_pe),
        )

    def forward(
        self,
        incidence: torch.Tensor,
        node_weights: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Operate directly on the sparse COO incidence [N, E]; never build the
        # dense [N, E] matrix (that is what previously OOMed on big graphs).
        if not incidence.is_sparse:
            incidence = incidence.to_sparse_coo()
        s = incidence.coalesce()
        idx = s.indices()
        num_nodes = s.size(0)
        num_edges = s.size(1)
        device = s.device
        if node_weights is None:
            node_weights = torch.ones(num_nodes, device=device)
        if edge_weights is None:
            edge_weights = torch.ones(num_edges, device=device)
        # node_degree[n] = sum_e I[n,e] * edge_weights[e]
        node_degree = torch.zeros(num_nodes, device=device)
        node_degree.scatter_add_(0, idx[0], edge_weights.to(torch.float32)[idx[1]])
        # edge_cardinality[e] = sum_n I[n,e] * node_weights[n]
        edge_cardinality = torch.zeros(num_edges, device=device)
        edge_cardinality.scatter_add_(0, idx[1], node_weights.to(torch.float32)[idx[0]])
        node_degree_norm = node_degree / node_degree.max().clamp_min(1e-8)
        edge_cardinality_norm = edge_cardinality / edge_cardinality.max().clamp_min(1e-8)
        rw_pe = hypergraph_rw_pe(incidence, num_steps=self.num_rw_steps)
        node_raw = torch.cat([node_degree_norm.unsqueeze(-1), rw_pe], dim=-1)
        edge_overlap = overlap_coefficient(incidence).unsqueeze(-1)
        edge_raw = torch.cat([edge_cardinality_norm.unsqueeze(-1), edge_overlap], dim=-1)
        pe_node = self.node_mlp(node_raw)
        pe_edge = self.edge_mlp(edge_raw)
        return (
            torch.nan_to_num(pe_node, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(pe_edge, nan=0.0, posinf=0.0, neginf=0.0),
        )


class CrossDomainFeatureProjectionModule(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_projector = CrossDomainProjector(hidden_dim)
        self.edge_projector = CrossDomainProjector(hidden_dim)

    def register_domain(self, domain_id: str | int, node_dim: int, edge_dim: int, feature_type: str) -> None:
        self.node_projector.register_domain(domain_id, node_dim, feature_type)
        self.edge_projector.register_domain(domain_id, edge_dim, feature_type)

    def forward(self, features: torch.Tensor, domain_id: str | int, is_edge: bool) -> torch.Tensor:
        projector = self.edge_projector if is_edge else self.node_projector
        return projector(features, domain_id)


class DomainAdapter(nn.Module):
    """Domain-specific adapter for learning domain-biased deviations from shared structure."""

    def __init__(self, hidden_dim: int, adapter_dim: int = 32):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.adapter_layer = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_value = self.gate(x)
        adapted = self.adapter_layer(x)
        return gate_value * adapted


class MoEExpert(nn.Module):
    """Mixture of Experts routing expert - outputs hidden_dim directly."""

    def __init__(self, hidden_dim: int, expert_dim: int = 32):
        super().__init__()
        self.expert_net = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert_net(x)


class DomainMoE(nn.Module):
    """Mixture of Experts for domain-specific adaptation."""

    def __init__(self, hidden_dim: int, num_domains: int, expert_dim: int = 32, num_experts: int = 4):
        super().__init__()
        self.num_domains = num_domains
        self.num_experts = num_experts
        # Each expert maps hidden -> expert_dim -> hidden
        self.experts = nn.ModuleList([
            MoEExpert(hidden_dim, expert_dim) for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor, domain_id: int | None = None) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Get routing weights
        routing_logits = self.router(x)  # (batch, num_experts)
        routing_weights = F.softmax(routing_logits, dim=-1)  # (batch, num_experts)
        
        # Stack expert outputs: (num_experts, batch, hidden)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        
        # Weighted sum: (batch, num_experts) @ (num_experts, batch, hidden) -> (batch, hidden)
        # routing_weights: (batch, num_experts) -> (batch, num_experts, 1)
        # expert_outputs: (num_experts, batch, hidden) -> (batch, num_experts, hidden) after transpose
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        expert_outputs_transposed = expert_outputs.transpose(0, 1)  # (batch, num_experts, hidden)
        
        adapted = (routing_weights_expanded * expert_outputs_transposed).sum(dim=1)  # (batch, hidden)
        
        if adapted.size(0) == 1:
            adapted = adapted.squeeze(0)
        
        return adapted


class DynamicDomainAdapter(nn.Module):
    """
    Unified dynamic domain adapter supporting both Adapter and MoE modes.
    Composes shared features with domain-specific adaptations.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_domains: int,
        adapter_type: str = "adapter",
        adapter_dim: int = 32,
        num_experts: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.adapter_type = adapter_type

        if adapter_type == "adapter":
            self.domain_adapters = nn.ModuleDict({
                str(d): DomainAdapter(hidden_dim, adapter_dim) for d in range(num_domains)
            })
        elif adapter_type == "moe":
            self.domain_moe = DomainMoE(hidden_dim, num_domains, adapter_dim, num_experts)
        else:
            self.domain_adapters = nn.ModuleDict()
            self.domain_moe = None

    def forward(self, shared_emb: torch.Tensor, domain_id: int) -> torch.Tensor:
        if shared_emb.numel() == 0:
            return shared_emb

        if self.adapter_type == "adapter":
            key = str(domain_id)
            if key not in self.domain_adapters:
                key = "0"
            adapter_output = self.domain_adapters[key](shared_emb)
        elif self.adapter_type == "moe":
            adapter_output = self.domain_moe(shared_emb, domain_id)
        else:
            adapter_output = shared_emb.new_zeros_like(shared_emb)

        return adapter_output


class StructureAwareAlignment(nn.Module):
    """
    Multi-granularity structure alignment module.
    Aligns node, edge, and subgraph structures across domains.
    """

    def __init__(self, hidden_dim: int, alignment_type: str = "prototype"):
        super().__init__()
        self.alignment_type = alignment_type
        self.hidden_dim = hidden_dim

        self.structure_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if alignment_type == "prototype":
            self.prototype_projector = nn.Linear(hidden_dim, hidden_dim)
        elif alignment_type == "ot":
            self.ot_cost_projector = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def compute_structure_features(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structure-aware features from node/edge embeddings and incidence."""
        if incidence.is_sparse:
            dense_inc = incidence.to_dense()
        else:
            dense_inc = incidence

        num_nodes = node_emb.size(0)
        node_context = torch.zeros_like(node_emb)

        for i in range(num_nodes):
            incident_edges = dense_inc[i].nonzero(as_tuple=True)[0]
            if incident_edges.numel() > 0:
                node_context[i] = edge_emb[incident_edges].mean(dim=0)

        combined = torch.cat([node_emb, node_context], dim=-1)
        return self.structure_encoder(combined)

    def prototype_alignment_loss(
        self,
        node_emb_1: torch.Tensor,
        node_emb_2: torch.Tensor,
        num_prototypes: int = 8,
    ) -> torch.Tensor:
        """Align structural prototypes across views."""
        combined_1 = self.compute_structure_features(node_emb_1, node_emb_1, torch.zeros(1, 1))
        combined_2 = self.compute_structure_features(node_emb_2, node_emb_2, torch.zeros(1, 1))

        proj_1 = F.normalize(self.prototype_projector(combined_1), dim=-1)
        proj_2 = F.normalize(self.prototype_projector(combined_2), dim=-1)

        prototypes_1 = proj_1[:num_prototypes]
        prototypes_2 = proj_2[:num_prototypes]

        sim_matrix = prototypes_1 @ prototypes_2.T / 0.07
        labels = torch.arange(num_prototypes, device=sim_matrix.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def structural_alignment_loss(
        self,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        struct_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        General structural alignment loss with multi-granularity consistency.
        """
        emb_1_norm = F.normalize(emb_1, dim=-1)
        emb_2_norm = F.normalize(emb_2, dim=-1)

        alignment_loss = 2 - 2 * (emb_1_norm * emb_2_norm).sum(dim=-1).mean()

        if struct_weight > 0:
            structure_1 = self.compute_structure_features(
                emb_1, emb_1, torch.zeros(1, 1, device=emb_1.device)
            )
            structure_2 = self.compute_structure_features(
                emb_2, emb_2, torch.zeros(1, 1, device=emb_2.device)
            )
            structure_loss = 2 - 2 * (F.normalize(structure_1, dim=-1) * F.normalize(structure_2, dim=-1)).sum(dim=-1).mean()
            return alignment_loss + struct_weight * structure_loss

        return alignment_loss


class ScalableSparseHyperedgeAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, topk: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.topk = topk
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def compute_relevance(self, node_tokens: torch.Tensor, edge_tokens: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coalesced = incidence.coalesce() if incidence.is_sparse else incidence.to_sparse_coo().coalesce()
        node_idx, edge_idx = coalesced.indices()
        q = self.q(node_tokens[node_idx])
        k = self.k(edge_tokens[edge_idx])
        scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden_dim)
        return scores, node_idx, edge_idx

    def forward(self, node_tokens: torch.Tensor, edge_tokens: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if node_tokens.numel() == 0 or edge_tokens.numel() == 0:
            return node_tokens, edge_tokens.new_zeros((node_tokens.size(0), 0), dtype=torch.long)
        scores, node_idx, edge_idx = self.compute_relevance(node_tokens, edge_tokens, incidence)
        if scores.numel() == 0:
            return node_tokens, edge_tokens.new_zeros((node_tokens.size(0), 0), dtype=torch.long)
        value = self.v(edge_tokens)
        num_nodes = node_tokens.size(0)
        device = node_tokens.device
        # Group all (score, edge) candidates by node, then pack into a padded
        # [num_nodes, Kmax] tensor so per-node topk/softmax runs in one shot.
        # Padding scores are -inf -> softmax weight 0 (no contribution).
        order = torch.argsort(node_idx, stable=True)
        n_s, s_s, e_s = node_idx[order], scores[order], edge_idx[order]
        uniq, counts = torch.unique_consecutive(n_s, return_counts=True)
        k_max = int(counts.max().item())
        if k_max == 0:
            return self.norm(node_tokens), edge_tokens.new_zeros((num_nodes, 0), dtype=torch.long)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        starts_exp = starts[:-1].repeat_interleave(counts)
        pos_in_group = torch.arange(n_s.numel(), device=device) - starts_exp

        pad_scores = torch.full((num_nodes, k_max), float("-inf"), device=device)
        pad_scores[n_s, pos_in_group] = s_s
        pad_edges = torch.full((num_nodes, k_max), -1, dtype=torch.long, device=device)
        pad_edges[n_s, pos_in_group] = e_s

        row_has = (pad_scores > float("-inf")).any(dim=1)
        # Rows with no candidate (isolated nodes) would be all -inf and make
        # softmax emit NaN; neutralise them so they receive a zero update.
        pad_scores = torch.where(row_has.unsqueeze(1), pad_scores, torch.zeros_like(pad_scores))

        take = min(int(self.topk), k_max)
        top_scores, top_pos = pad_scores.topk(take, dim=1)  # [num_nodes, take]
        attn = torch.softmax(top_scores, dim=1)             # -inf padding -> 0 weight
        gathered_edges = pad_edges.gather(1, top_pos)       # [num_nodes, take]
        gathered_value = value[gathered_edges.clamp_min(0)]  # padding -> row 0 (masked below)
        pad_mask = (gathered_edges == -1).unsqueeze(-1)
        gathered_value = gathered_value.masked_fill(pad_mask, 0.0)
        agg = (attn.unsqueeze(-1) * gathered_value).sum(dim=1)  # [num_nodes, d]
        agg = agg * row_has.unsqueeze(1).to(agg.dtype)      # isolated nodes -> agg 0
        # Isolated nodes must receive *no* update at all (not even the Linear
        # bias); zero the projected residual for them to match the loop version.
        update = self.dropout(self.out(agg)) * row_has.unsqueeze(1).to(agg.dtype)
        updated = node_tokens + update
        topk_index = torch.full((num_nodes, self.topk), -1, dtype=torch.long, device=device)
        topk_index[:, :take] = gathered_edges
        return self.norm(updated), topk_index


class DualLevelAttentionModule(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, max_k: int = 512):
        super().__init__()
        # Cap the packed sequence length (k_max) used by the batched MHA calls.
        # Hypergraphs such as pubmed contain hyperedges with tens of thousands of
        # members; letting k_max follow the largest edge makes the
        # [num_edges, k_max, d] buffer explode and triggers
        # "CUDA error: invalid configuration argument" in scaled_dot_product_attention.
        # Truncating oversized groups to max_k keeps the kernel valid and bounds memory.
        self.max_k = int(max_k)
        # The batched MHA is called once for ALL hyperedges (or all nodes) at once.
        # On whole-graph forward passes this batch dimension can reach ~90k
        # (pubmed: 88,676 edges), i.e. batch*num_heads >> 65535, which exceeds the
        # 16-bit CUDA grid limit and raises "CUDA error: invalid configuration
        # argument" inside scaled_dot_product_attention. We therefore split the
        # call into chunks of at most max_batch rows (per call), which keeps the
        # kernel config valid without changing the math.
        self.num_heads = int(num_heads)
        self.max_batch = 4096
        self._safe_chunk = max(1, min(self.max_batch, 65535 // self.num_heads))
        self.intra = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.inter = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.node_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.edge_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.node_norm1 = nn.LayerNorm(hidden_dim)
        self.edge_norm1 = nn.LayerNorm(hidden_dim)
        self.node_norm2 = nn.LayerNorm(hidden_dim)
        self.edge_norm2 = nn.LayerNorm(hidden_dim)

    def _chunked_mha(self, mha: nn.MultiheadAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Run batched MHA over rows split into safe-size chunks.

        nn.MultiheadAttention -> scaled_dot_product_attention crashes with
        "CUDA error: invalid configuration argument" when batch*num_heads exceeds
        the 16-bit CUDA grid limit (65535), which happens on whole-graph forward
        passes over large hypergraphs (pubmed: 88k edges). Chunking the batch
        dimension keeps every kernel launch within the limit.
        """
        b = q.size(0)
        if b <= self._safe_chunk:
            out, _ = mha(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
            return out
        outs = []
        for i in range(0, b, self._safe_chunk):
            j = min(i + self._safe_chunk, b)
            out_i, _ = mha(q[i:j], k[i:j], v[i:j],
                           key_padding_mask=key_padding_mask[i:j] if key_padding_mask is not None else None,
                           need_weights=False)
            outs.append(out_i)
        return torch.cat(outs, dim=0)

    def forward(self, node_tokens: torch.Tensor, edge_tokens: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if node_tokens.numel() == 0 and edge_tokens.numel() == 0:
            return node_tokens, edge_tokens
        device = node_tokens.device
        # Build sparse index groupings instead of a dense [N, E] incidence matrix
        # (the dense conversion previously OOMed on large hypergraphs).
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        node_idx, edge_idx = sp.indices()
        num_nodes = sp.size(0)
        num_edges = sp.size(1)

        d = node_tokens.size(-1)
        if edge_tokens.numel():
            # Pack every edge's member nodes into [E, Kmax_nodes, d] and run the
            # intra-edge self-attention as a single batched MHA call (no per-edge
            # Python loop, no batch=1 launch). Padding positions are masked out.
            order = torch.argsort(edge_idx, stable=True)
            e_s, n_s = edge_idx[order], node_idx[order]
            _, counts = torch.unique_consecutive(e_s, return_counts=True)
            k_max = int(counts.max().item()) if counts.numel() else 0
            if k_max > 0:
                # Truncate oversized hyperedges to self.max_k to keep the batched
                # MHA kernel valid and memory bounded (pubmed-scale graphs otherwise
                # crash with "CUDA error: invalid configuration argument").
                eff_k = min(k_max, self.max_k)
                starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
                starts_exp = starts[:-1].repeat_interleave(counts)
                pos_in_group = torch.arange(e_s.numel(), device=device) - starts_exp
                keep = pos_in_group < eff_k  # drop members beyond max_k of each edge
                e_s, n_s, pos_in_group = e_s[keep], n_s[keep], pos_in_group[keep]
                edge_seq = torch.zeros(num_edges, eff_k, d, device=device)
                edge_seq[e_s, pos_in_group] = node_tokens[n_s]
                key_mask = torch.ones(num_edges, eff_k, dtype=torch.bool, device=device)
                key_mask[e_s, pos_in_group] = False
                attn_out = self._chunked_mha(self.intra, edge_seq, edge_seq, edge_seq, key_mask)
                valid = (~key_mask).unsqueeze(-1).to(attn_out.dtype)
                # Padding (query/kv) positions in attention outputs are zeroed so
                # fully-padded edges don't emit NaN into the pooled mean.
                attn_out = attn_out.masked_fill(key_mask.unsqueeze(-1), 0.0)
                pooled = (attn_out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                edge_tokens = self.edge_norm1(edge_tokens + pooled)

        if node_tokens.numel() and edge_tokens.numel():
            # Pack every node's incident edges into [N, Kmax_edges, d] (KV) and
            # run cross-attention for all nodes in one MHA call. query = node.
            order = torch.argsort(node_idx, stable=True)
            n_s, e_s = node_idx[order], edge_idx[order]
            _, counts = torch.unique_consecutive(n_s, return_counts=True)
            k_max = int(counts.max().item()) if counts.numel() else 0
            if k_max > 0:
                # Same truncation safeguard as the intra-edge branch.
                eff_k = min(k_max, self.max_k)
                starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
                starts_exp = starts[:-1].repeat_interleave(counts)
                pos_in_group = torch.arange(n_s.numel(), device=device) - starts_exp
                keep = pos_in_group < eff_k  # drop incident edges beyond max_k of each node
                n_s, e_s, pos_in_group = n_s[keep], e_s[keep], pos_in_group[keep]
                node_seq = torch.zeros(num_nodes, eff_k, d, device=device)
                node_seq[n_s, pos_in_group] = edge_tokens[e_s]
                node_mask = torch.ones(num_nodes, eff_k, dtype=torch.bool, device=device)
                node_mask[n_s, pos_in_group] = False
                query_seq = node_tokens.unsqueeze(1)  # [N, 1, d]
                attn_out = self._chunked_mha(self.inter, query_seq, node_seq, node_seq, node_mask)
                # Nodes with no incident edges have fully-padded KV -> masked
                # attention emits NaN; zero those rows (no residual update).
                kv_valid = (~node_mask).sum(dim=1)
                row_zero = (kv_valid == 0).unsqueeze(1).unsqueeze(2)
                attn_out = attn_out.masked_fill(row_zero, 0.0)
                node_tokens = self.node_norm1(node_tokens + attn_out.squeeze(1))

        if node_tokens.numel():
            node_tokens = self.node_norm2(node_tokens + self.node_ffn(node_tokens))
        if edge_tokens.numel():
            edge_tokens = self.edge_norm2(edge_tokens + self.edge_ffn(edge_tokens))
        return node_tokens, edge_tokens


class HierarchicalHypergraphPooling(nn.Module):
    def __init__(self, hidden_dim: int, pooled_nodes: int, pooled_edges: int):
        super().__init__()
        self.pooled_nodes = pooled_nodes
        self.pooled_edges = pooled_edges
        self.node_assign = nn.Linear(hidden_dim, pooled_nodes)
        self.edge_assign = nn.Linear(hidden_dim, pooled_edges)

    def forward(self, node_tokens: torch.Tensor, edge_tokens: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if node_tokens.numel() == 0 or edge_tokens.numel() == 0:
            return node_tokens, edge_tokens, incidence
        # Keep the incidence sparse: pooled_incidence = s_node^T @ I @ s_edge,
        # computed as (dense[P,N] @ sparse[N,E]) @ dense[E,P] -> [P,P], never
        # materialising the dense [N,E] incidence matrix.
        #
        # NOTE: dense @ sparse requires the *sparse* op (`aten::sparse_dim` /
        # `aten::sparse.mm`). Some torch CUDA builds do not register this op on
        # the CPU backend, so we perform the product on the ORIGINAL device
        # (CUDA Sparse backend) in float32, then move the small [P, P] result
        # back. P is the number of pooled nodes / edges (defaults 64 / 32) so
        # the cost is negligible.
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        s_node = torch.softmax(self.node_assign(node_tokens), dim=1)
        s_edge = torch.softmax(self.edge_assign(edge_tokens), dim=1)
        pooled_nodes = s_node.transpose(0, 1) @ node_tokens
        pooled_edges = s_edge.transpose(0, 1) @ edge_tokens
        orig_device = sp.device
        # Some torch builds (e.g. selective/custom CUDA builds) do not register
        # the `aten::sparse_dim` op on either CPU or CUDA backends, so a dense
        # @ sparse product fails. The incidence matrix is small ([N, E] with
        # N<=256, E<=128), so materialising it dense is cheap and backend-safe.
        with torch.autocast(device_type=orig_device.type, enabled=False):
            sp_dense = sp.to_dense().to(orig_device).to(torch.float32)
            pooled_incidence = (
                (s_node.transpose(0, 1).to(torch.float32) @ sp_dense)
                @ s_edge.to(torch.float32)
            ).to(orig_device)
        return (
            torch.nan_to_num(pooled_nodes, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(pooled_edges, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(pooled_incidence, nan=0.0, posinf=0.0, neginf=0.0),
        )


@dataclass
class EncoderLayerConfig:
    hidden_dim: int
    num_heads: int
    topk: int
    dropout: float
    max_k: int = 512


class EncoderLayer(nn.Module):
    def __init__(self, config: EncoderLayerConfig):
        super().__init__()
        self.sparse_attn = ScalableSparseHyperedgeAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            topk=config.topk,
            dropout=config.dropout,
        )
        self.dual_attn = DualLevelAttentionModule(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_k=config.max_k,
        )

    def forward(self, node_tokens: torch.Tensor, edge_tokens: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_tokens, topk_index = self.sparse_attn(node_tokens, edge_tokens, incidence)
        node_tokens, edge_tokens = self.dual_attn(node_tokens, edge_tokens, incidence)
        return node_tokens, edge_tokens, topk_index
