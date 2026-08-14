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
    dense = _dense_incidence(incidence)
    if dense.size(1) == 0:
        return dense.new_zeros((0,))
    bt_b = dense.transpose(0, 1) @ dense
    cardinality = dense.sum(dim=0)
    min_cardinality = torch.minimum(cardinality.unsqueeze(0), cardinality.unsqueeze(1))
    overlap = bt_b / min_cardinality.clamp_min(1e-8)
    return torch.nan_to_num(overlap.mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0)


def hypergraph_rw_pe(incidence: torch.Tensor, num_steps: int = 5,
                      chunk_size: int = 256) -> torch.Tensor:
    # NOTE: previous implementation allocated three O(N^2) matrices
    # (d_v_inv, d_e_inv, w) plus the full transition matrix, which OOMs
    # on large hypergraphs (e.g. gowalla with 40K nodes). This version
    # chunks the computation: it never materialises any [N, N] matrix.
    # Memory peak per chunk is O(chunk_size * max(num_nodes, num_edges)).
    dense = _dense_incidence(incidence)
    num_nodes = dense.size(0)
    num_edges = dense.size(1) if dense.dim() > 1 else 0
    if num_nodes == 0:
        return dense.new_zeros((0, num_steps))
    device = dense.device
    dtype = dense.dtype
    node_degree = dense.sum(dim=1).clamp_min(1.0)
    edge_degree = dense.sum(dim=0).clamp_min(1.0)
    node_degree_inv = (1.0 / node_degree).to(dtype=dtype)  # [num_nodes]
    edge_degree_inv = (1.0 / edge_degree).to(dtype=dtype) if num_edges > 0 else None

    # We compute diag(P^k) for k = 1..num_steps where
    # P = D_v^-1 @ D @ D_e^-1 @ D^T, by iteratively multiplying a
    # chunk of identity rows by (P). At each step we only keep the
    # diagonal elements of that chunk.
    rw_features = torch.zeros(num_nodes, num_steps, device=device, dtype=dtype)
    chunk_size = max(1, min(int(chunk_size), num_nodes))
    dense_t = dense.transpose(0, 1)  # [num_edges, num_nodes]

    for cs in range(0, num_nodes, chunk_size):
        ce = min(cs + chunk_size, num_nodes)
        cur_chunk = ce - cs
        chunk_idx = torch.arange(cur_chunk, device=device)
        # cur = identity rows for this chunk  (shape: [cur_chunk, num_nodes])
        cur = torch.zeros(cur_chunk, num_nodes, device=device, dtype=dtype)
        cur[chunk_idx, cs + chunk_idx] = 1.0

        for k in range(num_steps):
            # cur = e_i^T @ transition = e_i^T @ D_v^-1 @ D @ D_e^-1 @ D^T
            # Scale rows by 1/node_degree (broadcast over columns)
            cur = cur * node_degree_inv.unsqueeze(0)
            # cur @ D   [cur_chunk, num_edges]
            cur = cur @ dense
            # Scale cols by 1/edge_degree
            if edge_degree_inv is not None:
                cur = cur * edge_degree_inv.unsqueeze(0)
            # cur @ D^T  [cur_chunk, num_nodes]
            cur = cur @ dense_t
            # Extract diagonal for this chunk
            rw_features[cs:ce, k] = cur[chunk_idx, cs + chunk_idx]

    rw_features = torch.nan_to_num(rw_features, nan=0.0, posinf=0.0, neginf=0.0)
    return rw_features


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
        dense = _dense_incidence(incidence)
        if node_weights is None:
            node_weights = dense.new_ones((dense.size(0),))
        if edge_weights is None:
            edge_weights = dense.new_ones((dense.size(1),))
        node_degree = (dense * edge_weights.unsqueeze(0)).sum(dim=1)
        edge_cardinality = (dense * node_weights.unsqueeze(1)).sum(dim=0)
        node_degree_norm = node_degree / node_degree.max().clamp_min(1e-8)
        edge_cardinality_norm = edge_cardinality / edge_cardinality.max().clamp_min(1e-8)
        rw_pe = hypergraph_rw_pe(dense, num_steps=self.num_rw_steps)
        node_raw = torch.cat([node_degree_norm.unsqueeze(-1), rw_pe], dim=-1)
        edge_overlap = overlap_coefficient(dense).unsqueeze(-1)
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
        node_outputs = []
        topk_index = torch.full(
            (node_tokens.size(0), self.topk),
            fill_value=-1,
            device=node_tokens.device,
            dtype=torch.long,
        )
        for node_id in range(node_tokens.size(0)):
            mask = node_idx == node_id
            candidate_scores = scores[mask]
            candidate_edges = edge_idx[mask]
            if candidate_scores.numel() == 0:
                node_outputs.append(node_tokens[node_id])
                continue
            take = min(int(self.topk), int(candidate_scores.numel()))
            top_scores, top_pos = candidate_scores.topk(take)
            chosen_edges = candidate_edges[top_pos]
            attn = torch.softmax(top_scores, dim=0)
            agg = (attn.unsqueeze(-1) * value[chosen_edges]).sum(dim=0)
            updated = node_tokens[node_id] + self.dropout(self.out(agg))
            node_outputs.append(updated)
            topk_index[node_id, :take] = chosen_edges
        return self.norm(torch.stack(node_outputs, dim=0)), topk_index


class DualLevelAttentionModule(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
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

    def forward(self, node_tokens: torch.Tensor, edge_tokens: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if node_tokens.numel() == 0 and edge_tokens.numel() == 0:
            return node_tokens, edge_tokens
        dense = _dense_incidence(incidence).to(torch.bool)

        if edge_tokens.numel():
            updated_edges = []
            for edge_id in range(edge_tokens.size(0)):
                members = torch.nonzero(dense[:, edge_id], as_tuple=False).view(-1)
                if members.numel() == 0:
                    updated_edges.append(edge_tokens[edge_id])
                    continue
                node_seq = node_tokens[members].unsqueeze(0)
                attn_out, _ = self.intra(node_seq, node_seq, node_seq, need_weights=False)
                pooled = attn_out.mean(dim=1).squeeze(0)
                updated_edges.append(edge_tokens[edge_id] + pooled)
            edge_tokens = torch.stack(updated_edges, dim=0)
            edge_tokens = self.edge_norm1(edge_tokens)

        if node_tokens.numel() and edge_tokens.numel():
            updated_nodes = []
            for node_id in range(node_tokens.size(0)):
                incident_edges = torch.nonzero(dense[node_id, :], as_tuple=False).view(-1)
                if incident_edges.numel() == 0:
                    updated_nodes.append(node_tokens[node_id])
                    continue
                query = node_tokens[node_id].view(1, 1, -1)
                key_value = edge_tokens[incident_edges].unsqueeze(0)
                attn_out, _ = self.inter(query, key_value, key_value, need_weights=False)
                updated_nodes.append(node_tokens[node_id] + attn_out.view(-1))
            node_tokens = torch.stack(updated_nodes, dim=0)
            node_tokens = self.node_norm1(node_tokens)

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
        dense = _dense_incidence(incidence).to(node_tokens.dtype if node_tokens.numel() else edge_tokens.dtype)
        if node_tokens.numel() == 0 or edge_tokens.numel() == 0:
            return node_tokens, edge_tokens, incidence
        s_node = torch.softmax(self.node_assign(node_tokens), dim=1)
        s_edge = torch.softmax(self.edge_assign(edge_tokens), dim=1)
        pooled_nodes = s_node.transpose(0, 1) @ node_tokens
        pooled_edges = s_edge.transpose(0, 1) @ edge_tokens
        pooled_incidence = s_node.transpose(0, 1) @ dense @ s_edge
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
        )

    def forward(self, node_tokens: torch.Tensor, edge_tokens: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_tokens, topk_index = self.sparse_attn(node_tokens, edge_tokens, incidence)
        node_tokens, edge_tokens = self.dual_attn(node_tokens, edge_tokens, incidence)
        return node_tokens, edge_tokens, topk_index
