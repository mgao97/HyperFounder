from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


def build_node_structural_features(incidence: torch.Tensor) -> torch.Tensor:
    node_degree = incidence.sum(dim=1)
    edge_sizes = incidence.sum(dim=0)
    node_overlap = incidence @ incidence.transpose(0, 1)
    incident_edge_sizes = incidence * edge_sizes.unsqueeze(0)
    incident_counts = incidence.sum(dim=1).clamp_min(1.0)
    incident_mean = incident_edge_sizes.sum(dim=1) / incident_counts
    if incidence.size(1) == 0:
        incident_max = incidence.new_zeros((incidence.size(0),))
    else:
        incident_max = incident_edge_sizes.max(dim=1).values
    return torch.stack(
        [
            torch.log1p(node_degree),
            incident_mean,
            incident_max,
            (node_overlap.sum(dim=1) - torch.diagonal(node_overlap)),
        ],
        dim=1,
    )


def build_edge_structural_features(incidence: torch.Tensor) -> torch.Tensor:
    edge_sizes = incidence.sum(dim=0)
    edge_overlap = incidence.transpose(0, 1) @ incidence
    normalized_overlap = edge_overlap.sum(dim=1) / edge_sizes.clamp_min(1.0)
    return torch.stack(
        [
            torch.log1p(edge_sizes),
            edge_sizes,
            normalized_overlap,
            (edge_overlap.sum(dim=1) - torch.diagonal(edge_overlap)),
        ],
        dim=1,
    )


def build_incidence_positional_features(
    incidence: torch.Tensor,
    node_base_features: torch.Tensor,
    edge_base_features: torch.Tensor,
    node_projection: nn.Module | None,
    edge_projection: nn.Module | None,
    structure_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes = incidence.size(0)
    num_edges = incidence.size(1)
    if structure_dim <= 0:
        return (
            incidence.new_zeros((num_nodes, 0)),
            incidence.new_zeros((num_edges, 0)),
        )

    node_degree = incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
    edge_sizes = incidence.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0)
    edge_messages = edge_projection(edge_base_features) if edge_projection is not None else incidence.new_zeros((num_edges, structure_dim))
    node_messages = node_projection(node_base_features) if node_projection is not None else incidence.new_zeros((num_nodes, structure_dim))
    node_pe = incidence @ edge_messages / node_degree
    edge_pe = incidence.transpose(0, 1) @ node_messages / edge_sizes
    return (
        torch.nan_to_num(node_pe, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(edge_pe, nan=0.0, posinf=0.0, neginf=0.0),
    )


def build_relative_bias(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    degree = torch.diagonal(matrix).clamp_min(1.0)
    norm = torch.sqrt(degree.unsqueeze(1) * degree.unsqueeze(0))
    return torch.nan_to_num(torch.log1p(matrix / norm), nan=0.0, posinf=0.0, neginf=0.0)


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features.new_zeros((features.size(0), self.net[-1].out_features))
        return self.net(features)


class StructuralSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            return tokens
        attn_out, _ = self.attn(tokens.unsqueeze(0), tokens.unsqueeze(0), tokens.unsqueeze(0), attn_mask=bias)
        hidden = self.norm1(tokens + self.dropout(attn_out.squeeze(0)))
        return self.norm2(hidden + self.dropout(self.ffn(hidden)))


@dataclass
class BackboneConfig:
    in_dim: int
    hidden_dim: int
    dropout: float
    num_layers: int
    num_heads: int
    structure_pe_dim: int


class HypergraphTransformerBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.structure_dim = max(int(config.structure_pe_dim), 0)
        self.node_projection = nn.Linear(config.in_dim, config.hidden_dim)
        self.edge_input_projection = nn.Linear(config.in_dim, config.hidden_dim)
        node_feature_dim = 4 + self.structure_dim
        edge_feature_dim = 4 + self.structure_dim
        self.node_pe = PositionalEncoder(node_feature_dim, config.hidden_dim)
        self.edge_pe = PositionalEncoder(edge_feature_dim, config.hidden_dim)
        if self.structure_dim > 0:
            self.node_incidence_projection = nn.Linear(4, self.structure_dim)
            self.edge_incidence_projection = nn.Linear(4, self.structure_dim)
        else:
            self.node_incidence_projection = None
            self.edge_incidence_projection = None
        self.node_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.node_edge_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_node_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.global_token = nn.Parameter(torch.zeros((config.hidden_dim,)))
        self.blocks = nn.ModuleList(
            [StructuralSelfAttentionBlock(config.hidden_dim, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )

    def _initialize_edge_tokens(self, x: torch.Tensor, incidence: torch.Tensor, edge_pe: torch.Tensor) -> torch.Tensor:
        if incidence.size(1) == 0:
            return x.new_zeros((0, self.hidden_dim))
        projected_nodes = self.edge_input_projection(x)
        edge_sizes = incidence.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0)
        edge_tokens = incidence.transpose(0, 1) @ projected_nodes / edge_sizes
        return edge_tokens + edge_pe

    def build_structure_cache(self, incidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        node_overlap = incidence @ incidence.transpose(0, 1)
        edge_overlap = incidence.transpose(0, 1) @ incidence
        node_features = torch.nan_to_num(
            build_node_structural_features(incidence),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        edge_features = torch.nan_to_num(
            build_edge_structural_features(incidence),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        node_incidence_pe, edge_incidence_pe = build_incidence_positional_features(
            incidence,
            node_base_features=node_features,
            edge_base_features=edge_features,
            node_projection=self.node_incidence_projection,
            edge_projection=self.edge_incidence_projection,
            structure_dim=self.structure_dim,
        )
        node_pe = torch.nan_to_num(
            self.node_pe(torch.cat([node_features, node_incidence_pe], dim=1)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        edge_pe = torch.nan_to_num(
            self.edge_pe(torch.cat([edge_features, edge_incidence_pe], dim=1)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        node_bias = torch.nan_to_num(build_relative_bias(node_overlap) * self.node_bias_scale, nan=0.0, posinf=0.0, neginf=0.0)
        edge_bias = torch.nan_to_num(build_relative_bias(edge_overlap) * self.edge_bias_scale, nan=0.0, posinf=0.0, neginf=0.0)
        node_edge_bias = torch.nan_to_num(torch.log1p(incidence) * self.node_edge_bias_scale, nan=0.0, posinf=0.0, neginf=0.0)
        edge_node_bias = torch.nan_to_num(
            torch.log1p(incidence.transpose(0, 1)) * self.edge_node_bias_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return {
            "node_pe": node_pe,
            "edge_pe": edge_pe,
            "node_incidence_pe": node_incidence_pe,
            "edge_incidence_pe": edge_incidence_pe,
            "node_bias": node_bias,
            "edge_bias": edge_bias,
            "node_edge_bias": node_edge_bias,
            "edge_node_bias": edge_node_bias,
            "node_overlap": node_overlap,
            "edge_overlap": edge_overlap,
        }

    def forward(self, x: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        structure_cache = self.build_structure_cache(incidence)
        node_tokens = torch.nan_to_num(self.node_projection(x) + structure_cache["node_pe"], nan=0.0, posinf=0.0, neginf=0.0)
        edge_tokens = torch.nan_to_num(self._initialize_edge_tokens(x, incidence, structure_cache["edge_pe"]), nan=0.0, posinf=0.0, neginf=0.0)
        global_token = self.global_token.to(node_tokens.device, dtype=node_tokens.dtype)
        if node_tokens.numel() and edge_tokens.numel():
            global_token = global_token + 0.5 * (node_tokens.mean(dim=0) + edge_tokens.mean(dim=0))
        elif node_tokens.numel():
            global_token = global_token + node_tokens.mean(dim=0)
        elif edge_tokens.numel():
            global_token = global_token + edge_tokens.mean(dim=0)
        global_token = global_token.unsqueeze(0)

        num_nodes = node_tokens.size(0)
        num_edges = edge_tokens.size(0)
        total_len = 1 + num_nodes + num_edges
        bias = node_tokens.new_zeros((total_len, total_len))
        if num_nodes:
            bias[1 : 1 + num_nodes, 1 : 1 + num_nodes] = structure_cache["node_bias"]
        if num_edges:
            start = 1 + num_nodes
            bias[start:, start:] = structure_cache["edge_bias"]
        if num_nodes and num_edges:
            start = 1 + num_nodes
            bias[1 : 1 + num_nodes, start:] = structure_cache["node_edge_bias"]
            bias[start:, 1 : 1 + num_nodes] = structure_cache["edge_node_bias"]

        tokens = torch.cat([global_token, node_tokens, edge_tokens], dim=0)
        for block in self.blocks:
            tokens = torch.nan_to_num(block(tokens, bias), nan=0.0, posinf=0.0, neginf=0.0)

        structure_cache["global_token"] = tokens[0]
        node_tokens = tokens[1 : 1 + num_nodes]
        edge_tokens = tokens[1 + num_nodes :]
        return node_tokens, edge_tokens, structure_cache


def build_backbone(
    in_dim: int,
    hidden_dim: int,
    dropout: float,
    num_layers: int,
    num_heads: int,
    structure_pe_dim: int,
) -> HypergraphTransformerBackbone:
    return HypergraphTransformerBackbone(
        BackboneConfig(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            structure_pe_dim=structure_pe_dim,
        )
    )
