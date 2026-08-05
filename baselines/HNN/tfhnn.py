import torch
import torch.nn as nn
from .mlp import MLP
import numpy as np
import scipy.sparse as sp
from .utils import create_coo_from_edge_index, ssm2tst


class TFHNN(nn.Module):
    
    def __init__(self, num_features, num_targets, args):

        super().__init__()

        self.L = args.All_num_layers
        self.num_decoder_layers = args.MLP_num_layers 
        self.hidden_dim = args.MLP_hidden
        self.alpha = args.alpha
        self.dropout = args.dropout
        self.norm = args.normalization
        self.args = args

        self.cache = None

        self.decoder = MLP(in_channels=num_features,
            hidden_channels=self.hidden_dim,
            out_channels=num_targets,
            num_layers=self.num_decoder_layers,
            dropout=self.dropout,
            Normalization=self.norm,
            InputNorm=False)

    def reset_parameters(self):
        self.cache = None
        self.decoder.reset_parameters()

    def weighted_ce_graph(self, data, args):

        H = create_coo_from_edge_index(data.hyperedge_index) 

        # 1. H x B x H^{T}
        colsum = np.array(H.sum(0), dtype='float')
        r_inv_sqrt = np.power(colsum, -1).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        B = sp.diags(r_inv_sqrt)

        A_beta = H.dot(B).dot(H.transpose())
        I = sp.eye(A_beta.shape[0])
        A_beta += I 

        # 2. D^{-1/2} x H x B x H^{T} x D^{1/2}
        rowsum = np.array(A_beta.sum(1), dtype='float')
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        D = sp.diags(r_inv_sqrt)

        A_beta = D.dot(A_beta).dot(D)
        
        return ssm2tst(A_beta).to(args.device)
    
    def diffusion(self, data):
        
        if self.cache is not None:
            return self.cache
            
        X = data.x
        
        norm_W = self.weighted_ce_graph(data, self.args)
        
        assert norm_W.layout == torch.sparse_coo
        norm_W = norm_W.coalesce()

        N = norm_W.size(0)
        X_out = self.alpha * X  # l = 0: (1 - alpha)^0 * X

        W_power_X = X.clone()

        for l in range(1, self.L):
            W_power_X = torch.sparse.mm(norm_W, W_power_X)
            X_out += self.alpha * ((1 - self.alpha) ** l) * W_power_X

        W_L_X = torch.sparse.mm(norm_W, W_power_X)
        X_out += ((1 - self.alpha) ** self.L) * W_L_X

        self.cache = X_out

        return X_out

    def forward(self, data):
        F = self.diffusion(data)
        return self.decoder(F), None 
