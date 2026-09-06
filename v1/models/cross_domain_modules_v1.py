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
    if not incidence.is_sparse:
        incidence = incidence.to_sparse_coo()
    s = incidence.coalesce()
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
    It = s.transpose(0, 1).coalesce()
    with torch.autocast(device_type=device.type, enabled=False):
        EI = torch.sparse.mm(It, s)
    ei = EI.indices()
    ev = EI.values().to(torch.float32)
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
    if not incidence.is_sparse:
        incidence = incidence.to_sparse_coo()
    s = incidence.coalesce()
    in_dtype = s.dtype
    if in_dtype != torch.float32:
        s = torch.sparse_coo_tensor(s.indices(), s.values().to(torch.float32), s.size())
    num_nodes = s.size(0)
    num_edges = s.size(1)
    if num_nodes == 0:
        return s.values().new_zeros((0, num_steps), dtype=in_dtype)
    device = s.device
    node_degree = torch.sparse.sum(s, dim=1).to_dense().clamp_min(1.0)
    edge_degree = torch.sparse.sum(s, dim=0).to_dense().clamp_min(1.0)
    node_degree_inv = 1.0 / node_degree
    edge_degree_inv = 1.0 / edge_degree

    rw_features = torch.zeros(num_nodes, num_steps, device=device, dtype=torch.float32)
    max_chunk_mem = max(1, int(mem_budget_bytes) // (num_nodes * 4))
    chunk_size = min(num_nodes, max(int(chunk_size), max_chunk_mem))
    s_t = s.transpose(0, 1).coalesce()

    for cs in range(0, num_nodes, chunk_size):
        ce = min(cs + chunk_size, num_nodes)
        cur_chunk = ce - cs
        chunk_idx = torch.arange(cur_chunk, device=device)
        cur = torch.zeros(cur_chunk, num_nodes, device=device, dtype=torch.float32)
        cur[chunk_idx, cs + chunk_idx] = 1.0

        for k in range(num_steps):
            cur = cur * node_degree_inv.unsqueeze(0)
            with torch.autocast(device_type=device.type, enabled=False):
                cur = torch.sparse.mm(s_t, cur.t()).t()
            cur = cur * edge_degree_inv.unsqueeze(0)
            with torch.autocast(device_type=device.type, enabled=False):
                cur = torch.sparse.mm(s, cur.t()).t()
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
        node_degree = torch.zeros(num_nodes, device=device)
        node_degree.scatter_add_(0, idx[0], edge_weights.to(torch.float32)[idx[1]])
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
    def __init__(self, hidden_dim: int, num_domains: int, expert_dim: int = 32, num_experts: int = 4):
        super().__init__()
        self.num_domains = num_domains
        self.num_experts = num_experts
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
        routing_logits = self.router(x)
        routing_weights = F.softmax(routing_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        routing_weights_expanded = routing_weights.unsqueeze(-1)
        expert_outputs_transposed = expert_outputs.transpose(0, 1)
        adapted = (routing_weights_expanded * expert_outputs_transposed).sum(dim=1)
        if adapted.size(0) == 1:
            adapted = adapted.squeeze(0)
        return adapted


class DynamicDomainAdapter(nn.Module):
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
        pad_scores = torch.where(row_has.unsqueeze(1), pad_scores, torch.zeros_like(pad_scores))

        take = min(int(self.topk), k_max)
        top_scores, top_pos = pad_scores.topk(take, dim=1)
        attn = torch.softmax(top_scores, dim=1)
        gathered_edges = pad_edges.gather(1, top_pos)
        gathered_value = value[gathered_edges.clamp_min(0)]
        pad_mask = (gathered_edges == -1).unsqueeze(-1)
        gathered_value = gathered_value.masked_fill(pad_mask, 0.0)
        agg = (attn.unsqueeze(-1) * gathered_value).sum(dim=1)
        agg = agg * row_has.unsqueeze(1).to(agg.dtype)
        update = self.dropout(self.out(agg)) * row_has.unsqueeze(1).to(agg.dtype)
        updated = node_tokens + update
        topk_index = torch.full((num_nodes, self.topk), -1, dtype=torch.long, device=device)
        topk_index[:, :take] = gathered_edges
        return self.norm(updated), topk_index


class DualLevelAttentionModule(nn.Module):
    """v1: chunked intra-edge and cross-attention to avoid whole-graph KV packing (OOM fix)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, max_k: int = 512):
        super().__init__()
        self.max_k = int(max_k)
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

    def _chunked_mha(self, mha, q, k, v, key_padding_mask):
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

    def _intra_edge_attention(self, edge_tokens, node_tokens, incidence, device, d):
        """Per-edge self-attention over member nodes, processed in chunks of edges."""
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        node_idx, edge_idx = sp.indices()
        num_edges = sp.size(1)
        order = torch.argsort(edge_idx, stable=True)
        e_s, n_s = edge_idx[order], node_idx[order]
        _, counts = torch.unique_consecutive(e_s, return_counts=True)
        k_max = int(counts.max().item()) if counts.numel() else 0
        if k_max == 0:
            return edge_tokens
        eff_k = min(k_max, self.max_k)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        starts_exp = starts[:-1].repeat_interleave(counts)
        pos_in_group = torch.arange(e_s.numel(), device=device) - starts_exp
        keep = pos_in_group < eff_k
        e_s, n_s, pos_in_group = e_s[keep], n_s[keep], pos_in_group[keep]

        # Build per-edge member lists (sparse-friendly) and process chunk-by-chunk.
        edge_groups = _build_groups(e_s, n_s, num_edges)
        pooled = edge_tokens.new_zeros((num_edges, d))
        valid = torch.zeros(num_edges, dtype=torch.bool, device=device)
        for start in range(0, num_edges, self._safe_chunk):
            end = min(start + self._safe_chunk, num_edges)
            seqs = []
            masks = []
            for eid in range(start, end):
                members = edge_groups[eid]
                if members is None or members.numel() == 0:
                    seqs.append(edge_tokens.new_zeros((1, eff_k, d)))
                    masks.append(torch.ones(1, eff_k, dtype=torch.bool, device=device))
                    continue
                members = members[:eff_k]
                seq = node_tokens[members].unsqueeze(0)  # [1, k, d]
                k_actual = seq.size(1)
                if k_actual < eff_k:
                    pad = edge_tokens.new_zeros((1, eff_k - k_actual, d))
                    seq = torch.cat([seq, pad], dim=1)
                seqs.append(seq)
                m = torch.zeros(1, k_actual, dtype=torch.bool, device=device)
                if eff_k > k_actual:
                    m = torch.cat([m, torch.ones(1, eff_k - k_actual, dtype=torch.bool, device=device)], dim=1)
                masks.append(m)
            seq = torch.cat(seqs, dim=0)              # [chunk, eff_k, d]
            mask = torch.cat(masks, dim=0)            # [chunk, eff_k]
            attn_out = self._chunked_mha(self.intra, seq, seq, seq, mask)
            attn_out = attn_out.masked_fill(mask.unsqueeze(-1), 0.0)
            v = (~mask).unsqueeze(-1).to(attn_out.dtype)
            p = (attn_out * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0)
            pooled[start:end] = p
            for off, eid in enumerate(range(start, end)):
                members = edge_groups[eid]
                valid[eid] = members is not None and members.numel() > 0
        pooled = torch.where(valid.unsqueeze(-1), pooled, edge_tokens)
        return self.edge_norm1(edge_tokens + pooled)

    def _node_cross_attention(self, node_tokens, edge_tokens, incidence, device, d):
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        node_idx, edge_idx = sp.indices()
        num_nodes = sp.size(0)
        order = torch.argsort(node_idx, stable=True)
        n_s, e_s = node_idx[order], edge_idx[order]
        _, counts = torch.unique_consecutive(n_s, return_counts=True)
        k_max = int(counts.max().item()) if counts.numel() else 0
        if k_max == 0:
            return node_tokens
        eff_k = min(k_max, self.max_k)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        starts_exp = starts[:-1].repeat_interleave(counts)
        pos_in_group = torch.arange(n_s.numel(), device=device) - starts_exp
        keep = pos_in_group < eff_k
        n_s, e_s, pos_in_group = n_s[keep], e_s[keep], pos_in_group[keep]

        node_groups = _build_groups(n_s, e_s, num_nodes)
        updated = node_tokens.clone()
        kv_valid = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        for start in range(0, num_nodes, self._safe_chunk):
            end = min(start + self._safe_chunk, num_nodes)
            seqs = []
            masks = []
            for nid in range(start, end):
                inc = node_groups[nid]
                if inc is None or inc.numel() == 0:
                    seqs.append(edge_tokens.new_zeros((1, eff_k, d)))
                    masks.append(torch.ones(1, eff_k, dtype=torch.bool, device=device))
                    continue
                inc = inc[:eff_k]
                seq = edge_tokens[inc].unsqueeze(0)  # [1, k, d]
                k_actual = seq.size(1)
                if k_actual < eff_k:
                    pad = edge_tokens.new_zeros((1, eff_k - k_actual, d))
                    seq = torch.cat([seq, pad], dim=1)
                seqs.append(seq)
                m = torch.zeros(1, k_actual, dtype=torch.bool, device=device)
                if eff_k > k_actual:
                    m = torch.cat([m, torch.ones(1, eff_k - k_actual, dtype=torch.bool, device=device)], dim=1)
                masks.append(m)
            seq = torch.cat(seqs, dim=0)
            mask = torch.cat(masks, dim=0)
            query = node_tokens[start:end].unsqueeze(1)  # [chunk, 1, d]
            # key_padding_mask=mask already excludes padded edges inside MHA,
            # so no post-hoc masked_fill is needed (and would be shape-mismatched).
            attn_out = self._chunked_mha(self.inter, query, seq, seq, mask)
            updated[start:end] = self.node_norm1(node_tokens[start:end] + attn_out.squeeze(1))
            for off, nid in enumerate(range(start, end)):
                inc = node_groups[nid]
                kv_valid[nid] = inc is not None and inc.numel() > 0
        return updated

    def forward(self, node_tokens, edge_tokens, incidence):
        if node_tokens.numel() == 0 and edge_tokens.numel() == 0:
            return node_tokens, edge_tokens
        device = node_tokens.device
        d = node_tokens.size(-1)
        if edge_tokens.numel():
            edge_tokens = self._intra_edge_attention(edge_tokens, node_tokens, incidence, device, d)
        if node_tokens.numel() and edge_tokens.numel():
            node_tokens = self._node_cross_attention(node_tokens, edge_tokens, incidence, device, d)
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

    def forward(self, node_tokens, edge_tokens, incidence):
        if node_tokens.numel() == 0 or edge_tokens.numel() == 0:
            return node_tokens, edge_tokens, incidence
        sp = incidence if incidence.is_sparse else incidence.to_sparse_coo()
        sp = sp.coalesce()
        s_node = torch.softmax(self.node_assign(node_tokens), dim=1)
        s_edge = torch.softmax(self.edge_assign(edge_tokens), dim=1)
        pooled_nodes = s_node.transpose(0, 1) @ node_tokens
        pooled_edges = s_edge.transpose(0, 1) @ edge_tokens
        with torch.autocast(device_type=sp.device.type, enabled=False):
            # Use sparse @ dense (supported) instead of dense @ sparse (not supported
            # in this torch build): (Pn x N) @ ((N x Ne) @ (Ne x Pe)) = (Pn x Pe).
            pooled_incidence = (s_node.transpose(0, 1)) @ (sp @ s_edge)
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

    def forward(self, node_tokens, edge_tokens, incidence):
        node_tokens, topk_index = self.sparse_attn(node_tokens, edge_tokens, incidence)
        node_tokens, edge_tokens = self.dual_attn(node_tokens, edge_tokens, incidence)
        return node_tokens, edge_tokens, topk_index
