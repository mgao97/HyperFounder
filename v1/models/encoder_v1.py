from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from v1.models.cross_domain_modules_v1 import (
    CrossDomainFeatureProjectionModule,
    CrossDomainStructuralPEModule,
    DynamicDomainAdapter,
    EncoderLayer,
    EncoderLayerConfig,
    HierarchicalHypergraphPooling,
)
from v1.models.hypergraph_data import HypergraphData
from utils.hypergraph import SimpleHypergraph
from utils.sampling import build_community_embeddings, build_cross_scale_embeddings, build_motif_embeddings, sample_communities, sample_motifs


class UnifiedHypergraphEncoderV1(nn.Module):
    """v1 encoder: OOM-safe attention + optional shared-branch return for downstream use.

    Key changes vs. encoder.py:
      * Uses cross_domain_modules_v1 (chunked intra/cross attention, no whole-graph KV packing).
      * Adds `return_shared` flag: when a disentangler is supplied, the returned
        node/edge embeddings are replaced by their *shared* projections so that the
        downstream fine-tuner consumes the transferable representation that the
        pre-training objective actually optimizes (fixes the previous mismatch where
        pre-training aligned z_shared but fine-tuning used the raw node_emb).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        num_heads: int,
        structure_pe_dim: int,
        num_domains: int = 4,
        domain_names: Optional[Sequence[str]] = None,
        topk: int = 16,
        pooled_nodes: int = 64,
        pooled_edges: int = 32,
        max_k: int = 512,
        use_domain_adapter: bool = True,
        adapter_type: str = "adapter",
        adapter_dim: int = 32,
        num_experts: int = 4,
    ):
        super().__init__()
        pe_dim = max(int(structure_pe_dim), 1)
        self.pe_module = CrossDomainStructuralPEModule(d_pe=pe_dim)
        self.projector = CrossDomainFeatureProjectionModule(hidden_dim=hidden_dim)
        self.node_pe_align = nn.Identity() if pe_dim == hidden_dim else nn.Linear(pe_dim, hidden_dim)
        self.edge_pe_align = nn.Identity() if pe_dim == hidden_dim else nn.Linear(pe_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    EncoderLayerConfig(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        topk=topk,
                        dropout=dropout,
                        max_k=max_k,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.pooling_module = HierarchicalHypergraphPooling(
            hidden_dim=hidden_dim,
            pooled_nodes=pooled_nodes,
            pooled_edges=pooled_edges,
        )
        self.readout_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.subhypergraph_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.use_domain_adapter = use_domain_adapter
        self.adapter_type = adapter_type
        if use_domain_adapter:
            self.domain_adapter = DynamicDomainAdapter(
                hidden_dim=hidden_dim,
                num_domains=num_domains,
                adapter_type=adapter_type,
                adapter_dim=adapter_dim,
                num_experts=num_experts,
            )
        else:
            self.domain_adapter = None

        if domain_names is None:
            self.domain_to_id: Dict[str, int] = {}
        else:
            self.domain_to_id = {str(name): index for index, name in enumerate(domain_names)}
        self.num_domains = int(num_domains)
        # v1: structural sub-sampling (motifs/communities) is off by default for
        # downstream fine-tuning; set True only when pre-training uses them.
        self.use_communities = False
        self.use_motifs = False

    def encode_candidate_hyperedges(self, node_emb: torch.Tensor, hyperedges: List[List[int]]) -> torch.Tensor:
        if not hyperedges:
            return node_emb.new_zeros((0, node_emb.size(-1)))
        pooled = []
        for edge in hyperedges:
            if not edge:
                pooled.append(node_emb.new_zeros(node_emb.size(-1)))
                continue
            pooled.append(node_emb[edge].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def forward(
        self,
        hg: SimpleHypergraph | HypergraphData,
        x: Optional[torch.Tensor] = None,
        motif_budget: int = 32,
        motifs: Optional[List[Dict[str, List[int]]]] = None,
        communities: Optional[List[Dict[str, List[int]]]] = None,
        motif_seed: int = 0,
        return_shared: bool = False,
        node_disentangler=None,
        edge_disentangler=None,
    ):
        if isinstance(hg, HypergraphData):
            data = hg
            feature_tensor = hg.node_features
            incidence_dense = (
                hg.incidence_matrix.to_sparse_coo()
                if hg.incidence_matrix.is_sparse
                else hg.incidence_matrix
            )
            domain_name = str(hg.domain_id)
            domain_id = int(hg.domain_id)
            edge_features = hg.edge_features
            source_hg = None
        else:
            if x is None:
                raise ValueError("Encoder forward requires `x` when input is SimpleHypergraph.")
            incidence_dense = hg.incidence_matrix().to(x.device)
            incidence = incidence_dense.to_sparse_coo()
            num_edges = incidence_dense.size(1)
            edge_features = x.new_zeros((num_edges, x.size(-1)))
            if self.domain_to_id:
                domain_id = int(self.domain_to_id.get(hg.domain, 0))
            else:
                domain_id = int(abs(hash(hg.domain)) % max(self.num_domains, 1))
            data = HypergraphData(
                node_features=x,
                edge_features=edge_features,
                incidence_matrix=incidence,
                node_labels=hg.node_labels.to(x.device) if hg.node_labels is not None else None,
                domain_id=domain_id,
                feature_type=str(getattr(hg, "feature_type", "numerical")),
            )
            feature_tensor = x
            domain_name = hg.domain
            source_hg = hg
        model_device = next(self.parameters()).device
        if data.node_features.device != model_device:
            data.node_features = data.node_features.to(model_device)
        if data.edge_features.device != model_device:
            data.edge_features = data.edge_features.to(model_device)
        if data.incidence_matrix.device != model_device:
            data.incidence_matrix = data.incidence_matrix.to(model_device)
        if incidence_dense.device != model_device:
            incidence_dense = incidence_dense.to(model_device)
        if data.node_labels is not None and data.node_labels.device != model_device:
            data.node_labels = data.node_labels.to(model_device)
        self.projector.register_domain(
            domain_id=data.domain_id,
            node_dim=int(data.node_features.size(-1)),
            edge_dim=int(edge_features.size(-1)),
            feature_type=str(data.feature_type),
        )
        incidence = data.incidence_matrix if data.incidence_matrix.is_sparse else data.incidence_matrix.to_sparse_coo()
        pe_node, pe_edge = self.pe_module(incidence)
        pe_node = self.node_pe_align(pe_node)
        pe_edge = self.edge_pe_align(pe_edge)
        node_tokens = self.projector(data.node_features, domain_id=data.domain_id, is_edge=False) + pe_node
        edge_tokens = self.projector(data.edge_features, domain_id=data.domain_id, is_edge=True) + pe_edge

        for layer in self.encoder_layers:
            node_tokens, edge_tokens, _ = layer(node_tokens, edge_tokens, incidence)

        node_emb = torch.nan_to_num(node_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        edge_emb = torch.nan_to_num(edge_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        if self.domain_adapter is not None:
            domain_id_int = int(data.domain_id) if hasattr(data, 'domain_id') else 0
            node_adapter_out = self.domain_adapter(node_emb, domain_id_int)
            edge_adapter_out = self.domain_adapter(edge_emb, domain_id_int)
            node_emb = node_emb + node_adapter_out
            edge_emb = edge_emb + edge_adapter_out

        # v1: optionally replace raw embeddings with their shared disentangled branch,
        # so downstream tasks use the transferable representation actually optimized
        # during pre-training.
        if return_shared and node_disentangler is not None and edge_disentangler is not None:
            z_node_shared, _, _ = node_disentangler(node_emb)
            z_edge_shared, _, _ = edge_disentangler(edge_emb)
            node_emb = node_emb + z_node_shared  # residual fusion keeps raw signal + shared
            edge_emb = edge_emb + z_edge_shared

        pooled_nodes, pooled_edges, pooled_incidence = self.pooling_module(node_emb, edge_emb, incidence)
        node_graph = pooled_nodes.mean(dim=0) if pooled_nodes.numel() else node_emb.mean(dim=0)
        edge_graph = pooled_edges.mean(dim=0) if pooled_edges.numel() else edge_emb.mean(dim=0)
        graph_emb = torch.nan_to_num(
            self.readout_projection(torch.cat([node_graph, edge_graph], dim=0)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        motif_items = motifs if motifs is not None else (
            sample_motifs(source_hg, budget=motif_budget, seed=motif_seed) if source_hg is not None else []
        )
        community_items = communities if communities is not None else (
            sample_communities(source_hg) if (source_hg is not None and getattr(self, "use_communities", False)) else []
        )
        motif_emb = torch.nan_to_num(
            build_motif_embeddings(node_emb, edge_emb, motif_items, self.subhypergraph_projection),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        community_emb = torch.nan_to_num(
            build_community_embeddings(node_emb, edge_emb, community_items, self.subhypergraph_projection),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        cross_emb = torch.nan_to_num(
            build_cross_scale_embeddings(motif_emb, community_emb, graph_emb),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        aux = {
            "motif_emb": motif_emb,
            "community_emb": community_emb,
            "cross_emb": cross_emb,
            "motifs": motif_items,
            "communities": community_items,
            "incidence": incidence_dense,
            "pooled_incidence": pooled_incidence,
            "pooled_node_emb": pooled_nodes,
            "pooled_edge_emb": pooled_edges,
            "node_pe": pe_node,
            "edge_pe": pe_edge,
            "domain_name": domain_name,
        }
        return node_emb, edge_emb, graph_emb, aux
