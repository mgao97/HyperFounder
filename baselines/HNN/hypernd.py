import torch
import torch.nn as nn
from .mlp import MLP
import torch_scatter


class HyperND(nn.Module):
    def __init__(self, num_features, num_targets, args):

        super().__init__()

        self.hidden_dim = args.MLP_hidden
        self.num_layers = args.MLP_num_layers 
        self.dropout = args.dropout
        self.norm = args.normalization

        self.p = args.HyperND_ord
        self.tol = args.HyperND_tol
        self.max_steps = args.HyperND_steps
        self.aggr = args.aggregate
        self.alpha = args.restart_alpha

        self.cache = None

        self.decoder = MLP(in_channels=num_features,
            hidden_channels=self.hidden_dim,
            out_channels=num_targets,
            num_layers=self.num_layers,
            dropout=self.dropout,
            Normalization=self.norm,
            InputNorm=False)

    def reset_parameters(self):
        self.cache = None
        self.decoder.reset_parameters()

    def diffusion(self, data):
        if self.cache is not None:
            return self.cache

        X = data.x
        N = X.shape[-2]
        V, E = data.hyperedge_index[0], data.hyperedge_index[1]
        device = X.device

        ones = torch.ones(V.shape[0], device=device, dtype=torch.float32)
        D = torch_scatter.scatter_add(ones, V, dim=0)

        ones = torch.ones(E.shape[0], device=device, dtype=torch.float32)
        DE = torch_scatter.scatter_add(ones, E, dim=0)

        def rho(Z):
            return torch.pow(Z, self.p)

        def sigma(Z):
            Z = Z / DE[:, None]
            return torch.pow(Z, 1. / self.p)

        def V2E(Z):
            Z = Z / torch.pow(D, 0.5)[:, None]
            Xve = rho(Z[..., V, :])  # [nnz, C] 
            Xe = torch_scatter.scatter(Xve, E, dim=-2, reduce=self.aggr)  # [E, C]
            Xe = sigma(Xe)
            return Xe

        def E2V(Z):
            Xev = Z[..., E, :]  # [nnz, C]
            Xv = torch_scatter.scatter(Xev, V, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]
            Xv = Xv / torch.pow(D, 0.5)[:, None]
            return Xv

        def phi(Z):
            P = V2E(Z)
            P = torch.linalg.norm(P, dim=-1, ord=2.) ** 2.
            return 2. * torch.sqrt(P.sum())

        U = X / phi(X)
        F = U
        for i in range(self.max_steps):
            F_last = F

            G = (1. - self.alpha) * E2V(V2E(F)) + self.alpha * U
            F = G / phi(G)

            d = torch.linalg.norm(F - F_last, ord='fro') / torch.linalg.norm(F, ord='fro')
            if d < self.tol:
                print(f'Interrupt hypergraph diffusion with {i} iterations and d={d}')
                break

        self.cache = F

        return F

    def forward(self, data):
        F = self.diffusion(data)
        return self.decoder(F), None 
