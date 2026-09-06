"""HyperGFSE encoder: stack of HyperGPS layers producing a PSE.

Public API mirrors GFSE:
  - forward(H, x_init=None) -> PSE  (N, output_dim)
  - PSE is a generic positional/structural encoding usable by any downstream
    graph model exactly like GFSE's: concat [x_init | PSE] or project to LLM space.

Differences from GFSE that matter for reproducibility:
  - initial node features are the absolute random-walk PE P (dim d=8) instead of
    randomized features; if raw node features x_init are provided they are
    projected to hidden and added to P.
  - R is the hypergraph relative PE computed once and reused across layers
    (matches GFSE, which feeds the same R^ell to every GPS layer).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .encoding import HypergraphRandomWalkPE
from .hypergps import HyperGPSLayer


class HyperGFSE(nn.Module):
    def __init__(
        self,
        pe_dim: int = 8,
        hidden: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        output_dim: int = 64,
        rw_mode: str = "incidence",
        size_invariant: bool = False,
        beta: float = 0.5,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.pe = HypergraphRandomWalkPE(
            dim=pe_dim, walk=rw_mode, size_invariant=size_invariant, beta=beta, device=device
        )
        self.pe_dim = pe_dim
        self.input_proj = nn.Linear(pe_dim, hidden)
        self.layers = nn.ModuleList(
            [HyperGPSLayer(hidden, num_heads, pe_dim, dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden, output_dim)
        self.device = device

    def forward(self, H: np.ndarray, x_init: torch.Tensor | None = None) -> torch.Tensor:
        """H: (N,E) 0/1 numpy incidence. x_init: optional (N,dx) tensor raw features.
        Returns PSE (N, output_dim)."""
        P, R = self.pe.forward(H)  # (N,pe_dim), (N,N,pe_dim)
        x = self.input_proj(P.to(self.device))
        if x_init is not None:
            if not hasattr(self, "feat_proj"):
                self.feat_proj = nn.Linear(x_init.size(-1), self.input_proj.out_features).to(self.device)
            x = x + self.feat_proj(x_init.to(self.device))
        Ht = torch.as_tensor(H, dtype=torch.float32, device=self.device)
        for layer in self.layers:
            x = layer(x, R.to(self.device), Ht)
        return self.out_proj(x)  # (N, output_dim)
