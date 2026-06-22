from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


def _safe_eigen_features(matrix: torch.Tensor, num_components: int) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix.new_zeros((0, num_components))
    if matrix.size(0) == 1:
        return matrix.new_ones((1, num_components))
    sym_matrix = (matrix + matrix.transpose(0, 1)) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(sym_matrix)
    take = min(num_components, eigvecs.size(1))
    selected = eigvecs[:, -take:]
    if take < num_components:
        pad = selected.new_zeros((selected.size(0), num_components - take))
        selected = torch.cat([selected, pad], dim=1)
    return torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)


def build_node_structural_features(incidence: torch.Tensor, spectral_dim: int) -> torch.Tensor:
    node_degree = incidence.sum(dim=1)
    edge_sizes = incidence.sum(dim=0)
    node_overlap = incidence @ incidence.transpose(0, 1)
    incident_edge_sizes = incidence * edge_sizes.unsqueeze(0)
    incident_counts = incidence.sum(dim=1).clamp_min(1.0)
    incident_mean = incident_edge_sizes.sum(dim=1) / incident_counts
    incident_max = incident_edge_sizes.max(dim=1).values
    spectral = _safe_eigen_features(node_overlap, spectral_dim)
    base = torch.stack(
        [
            torch.log1p(node_degree),
            incident_mean,
            incident_max,
            (node_overlap.sum(dim=1) - torch.diagonal(node_overlap)),
        ],
        dim=1,
    )
    return torch.cat([base, spectral], dim=1)


def build_edge_structural_features(incidence: torch.Tensor, spectral_dim: int) -> torch.Tensor:
    edge_sizes = incidence.sum(dim=0)
    edge_overlap = incidence.transpose(0, 1) @ incidence
    normalized_overlap = edge_overlap.sum(dim=1) / edge_sizes.clamp_min(1.0)
    spectral = _safe_eigen_features(edge_overlap, spectral_dim)
    base = torch.stack(
        [
            torch.log1p(edge_sizes),
            edge_sizes,
            normalized_overlap,
            (edge_overlap.sum(dim=1) - torch.diagonal(edge_overlap)),
        ],
        dim=1,
    )
    return torch.cat([base, spectral], dim=1)


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
    spectral_dim: int


class HypergraphTransformerBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.node_projection = nn.Linear(config.in_dim, config.hidden_dim)
        self.edge_input_projection = nn.Linear(config.in_dim, config.hidden_dim)
        node_feature_dim = 4 + config.spectral_dim
        edge_feature_dim = 4 + config.spectral_dim
        self.node_pe = PositionalEncoder(node_feature_dim, config.hidden_dim)
        self.edge_pe = PositionalEncoder(edge_feature_dim, config.hidden_dim)
        self.node_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.node_blocks = nn.ModuleList(
            [StructuralSelfAttentionBlock(config.hidden_dim, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.edge_blocks = nn.ModuleList(
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
            build_node_structural_features(incidence, spectral_dim=self.node_pe.net[0].in_features - 4),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        edge_features = torch.nan_to_num(
            build_edge_structural_features(incidence, spectral_dim=self.edge_pe.net[0].in_features - 4),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        node_pe = torch.nan_to_num(self.node_pe(node_features), nan=0.0, posinf=0.0, neginf=0.0)
        edge_pe = torch.nan_to_num(self.edge_pe(edge_features), nan=0.0, posinf=0.0, neginf=0.0)
        node_bias = torch.nan_to_num(build_relative_bias(node_overlap) * self.node_bias_scale, nan=0.0, posinf=0.0, neginf=0.0)
        edge_bias = torch.nan_to_num(build_relative_bias(edge_overlap) * self.edge_bias_scale, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "node_pe": node_pe,
            "edge_pe": edge_pe,
            "node_bias": node_bias,
            "edge_bias": edge_bias,
            "node_overlap": node_overlap,
            "edge_overlap": edge_overlap,
        }

    def forward(self, x: torch.Tensor, incidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        structure_cache = self.build_structure_cache(incidence)
        node_tokens = torch.nan_to_num(self.node_projection(x) + structure_cache["node_pe"], nan=0.0, posinf=0.0, neginf=0.0)
        edge_tokens = torch.nan_to_num(self._initialize_edge_tokens(x, incidence, structure_cache["edge_pe"]), nan=0.0, posinf=0.0, neginf=0.0)
        for node_block, edge_block in zip(self.node_blocks, self.edge_blocks):
            node_tokens = torch.nan_to_num(node_block(node_tokens, structure_cache["node_bias"]), nan=0.0, posinf=0.0, neginf=0.0)
            edge_tokens = torch.nan_to_num(edge_block(edge_tokens, structure_cache["edge_bias"]), nan=0.0, posinf=0.0, neginf=0.0)
        return node_tokens, edge_tokens, structure_cache


def build_backbone(
    in_dim: int,
    hidden_dim: int,
    dropout: float,
    num_layers: int,
    num_heads: int,
    spectral_dim: int,
) -> HypergraphTransformerBackbone:
    return HypergraphTransformerBackbone(
        BackboneConfig(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            spectral_dim=spectral_dim,
        )
    )
