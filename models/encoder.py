from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from models.backbone import build_backbone
from utils.hypergraph import SimpleHypergraph
from utils.sampling import build_community_embeddings, build_cross_scale_embeddings, build_motif_embeddings, sample_communities, sample_motifs


class UnifiedHypergraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        num_heads: int,
        spectral_dim: int,
    ):
        super().__init__()
        self.backbone = build_backbone(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            spectral_dim=spectral_dim,
        )
        self.readout_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.subhypergraph_projection = nn.Linear(hidden_dim * 2, hidden_dim)

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
        hg: SimpleHypergraph,
        x: torch.Tensor,
        motif_budget: int = 32,
        motifs: Optional[List[Dict[str, List[int]]]] = None,
        communities: Optional[List[Dict[str, List[int]]]] = None,
        motif_seed: int = 0,
    ):
        incidence = hg.incidence_matrix().to(x.device)
        node_emb, edge_emb, structure_cache = self.backbone(x, incidence)
        node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)
        edge_emb = torch.nan_to_num(edge_emb, nan=0.0, posinf=0.0, neginf=0.0)
        node_graph = node_emb.mean(dim=0) if node_emb.numel() else x.new_zeros((self.readout_projection.out_features,))
        edge_graph = edge_emb.mean(dim=0) if edge_emb.numel() else node_graph.new_zeros(node_graph.shape)
        graph_emb = torch.nan_to_num(self.readout_projection(torch.cat([node_graph, edge_graph], dim=0)), nan=0.0, posinf=0.0, neginf=0.0)
        motif_items = motifs if motifs is not None else sample_motifs(hg, budget=motif_budget, seed=motif_seed)
        community_items = communities if communities is not None else sample_communities(hg)
        motif_emb = torch.nan_to_num(build_motif_embeddings(node_emb, edge_emb, motif_items, self.subhypergraph_projection), nan=0.0, posinf=0.0, neginf=0.0)
        community_emb = torch.nan_to_num(build_community_embeddings(node_emb, edge_emb, community_items, self.subhypergraph_projection), nan=0.0, posinf=0.0, neginf=0.0)
        cross_emb = torch.nan_to_num(build_cross_scale_embeddings(motif_emb, community_emb, graph_emb), nan=0.0, posinf=0.0, neginf=0.0)
        aux = {
            "motif_emb": motif_emb,
            "community_emb": community_emb,
            "cross_emb": cross_emb,
            "motifs": motif_items,
            "communities": community_items,
            "incidence": incidence,
            "node_bias": structure_cache["node_bias"],
            "edge_bias": structure_cache["edge_bias"],
            "node_pe": structure_cache["node_pe"],
            "edge_pe": structure_cache["edge_pe"],
        }
        return node_emb, edge_emb, graph_emb, aux
