import math
import torch
import torch.nn as nn
from .mlp import MLP_inner


def util_sparse_einsum(abh: torch.sparse.Tensor, bhd: torch.Tensor) -> torch.Tensor:
    """abh,bhd->ahd"""
    a, b, h = abh.size()
    b_, h_, d = bhd.size()
    assert b == b_ and h == h_
    indices, values = abh.indices(), abh.values()
    res = torch.sparse_coo_tensor(
        indices[0:1], values[..., None] * bhd[indices[1]], size=(a, h, d)
    )
    return res.coalesce().to_dense()


class EHNNTransformerConv(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_qk,
        dim_ff,
        n_heads,
        dropout,
        hypernet_info,
        inner_dim,
        pe_dim,
        hyper_dim,
        hyper_layers,
        hyper_dropout,
        force_broadcast,
        input_dropout,
        mlp_classifier,
        att0_dropout,
        att1_dropout,
    ):
        super().__init__()
        assert mlp_classifier
        self.input_dropout = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.lin_proj = nn.Linear(dim_in, dim_hidden)
        
        self.v2e_layer = TransformerV2E(
            dim_hidden,
            dim_qk,
            dim_ff,
            n_heads,
            dropout,
            hypernet_info,
            inner_dim,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
            att0_dropout,
            att1_dropout,
        )
        self.e2v_layer = TransformerE2V(
            dim_hidden,
            dim_qk,
            dim_ff,
            n_heads,
            dropout,
            hypernet_info,
            inner_dim,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
            att0_dropout,
            att1_dropout,
        )

    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        self.v2e_layer.reset_parameters()
        self.e2v_layer.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:
            
        x = self.input_dropout(x)
        x = self.lin_proj(x)

        x_v, x_e = self.v2e_layer(x, ehnn_cache)
        x_v, x_e = self.dropout(x_v), self.dropout(x_e)
        x = (x_v, x_e)
        x = self.e2v_layer(x, ehnn_cache)

        return x, x_e


class SelfAttnV2E(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_qk,
        n_heads,
        hypernet_info,
        inner_dim,
        pe_dim,
        hyper_dim,
        hyper_layers,
        hyper_dropout,
        force_broadcast,
        att0_dropout,
        att1_dropout,
    ):
        super().__init__()
        _, max_l, _ = hypernet_info
        self.max_l = max_l
        self.dim_in = dim_in
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        self.inner_dim = inner_dim
        self.dim_v = inner_dim
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.dim_v_head = self.dim_v // n_heads if self.dim_v >= n_heads else 1
        self.q = PositionalMLP(
            n_heads * self.dim_qk_head,
            2,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
        )
        self.k = nn.Linear(self.dim_v, 2 * dim_qk)
        self.v = nn.Linear(self.dim_v, self.dim_v)

        self.mlp1 = MLP_inner(dim_in, self.dim_v, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP_inner(
            self.dim_v * 2, self.dim_v, hyper_dim, hyper_layers, hyper_dropout
        )
        self.mlp3 = MLP_inner(self.dim_v * 2, dim_in, hyper_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.pe2 = PositionalEncoding(inner_dim, 2)
        self.pe3 = PositionalEncoding(inner_dim, max_l + 1)
        self.b = BiasE(
            dim_in,
            max_l,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
        )

        self.att0_dropout = nn.Dropout(att0_dropout)
        self.att1_dropout = nn.Dropout(att1_dropout)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.b.reset_parameters()

    def forward(self, x, ehnn_cache):
        incidence = ehnn_cache["incidence"]
        edge_orders = ehnn_cache["edge_orders"]

        # MLP1
        x = x + self.mlp1(self.norm1(x))

        q_0 = self.q(torch.zeros(1, dtype=torch.long, device=x.device)).view(
            self.n_heads, self.dim_qk_head
        )
        q_1 = self.q(torch.ones(1, dtype=torch.long, device=x.device)).view(
            self.n_heads, self.dim_qk_head
        )
        k = self.k(x)
        k_0, k_1 = k.split(self.dim_qk, -1)
        k_0 = torch.stack(k_0.split(self.dim_qk_head, -1), 1)
        k_1 = torch.stack(k_1.split(self.dim_qk_head, -1), 1)
        v = self.v(x)
        v = torch.stack(v.split(self.dim_v_head, -1), 1)

        # 0-overlap (global)
        logit0 = torch.einsum("hd,nhd->nh", q_0, k_0) / math.sqrt(
            self.dim_qk_head
        )
        alpha0 = torch.softmax(logit0, dim=0)
        att0 = torch.einsum("nh,nhd->hd", alpha0, v)
        att0 = torch.cat(att0.unbind(-2), -1)
        att0 = att0[None, :]
        
        # 1-overlap node-to-node
        att1_v = torch.cat(v.unbind(-2), -1)
        
        # 1-overlap node-to-edge
        logit1 = torch.einsum("hd,nhd->nh", q_1, k_1) / math.sqrt(
            self.dim_qk_head
        )
        n, e = incidence.size()
        indices = incidence.indices()[[1, 0]]
        logit1 = torch.sparse_coo_tensor(
            indices, logit1[indices[1]], size=(e, n, self.n_heads)
        )
        alpha1 = torch.sparse.softmax(logit1, dim=1).coalesce()
        att1_e = util_sparse_einsum(alpha1, v)
        att1_e = torch.cat(att1_e.unbind(-2), -1)

        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=x.device)).view(
            1, self.inner_dim
        )
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x.device)).view(
            1, self.inner_dim
        )

        t0 = self.norm2(att0)
        t1_v = self.norm2(att1_v)
        t1_e = self.norm2(att1_e)
        att0 = att0 + self.mlp2(torch.cat((t0, pe2_s0.expand(t0.shape)), dim=-1))
        att1_v = att1_v + self.mlp2(
            torch.cat((t1_v, pe2_s1.expand(t1_v.shape)), dim=-1)
        )
        att1_e = att1_e + self.mlp2(
            torch.cat((t1_e, pe2_s1.expand(t1_e.shape)), dim=-1)
        )

        x_v = self.att0_dropout(att0) + self.att1_dropout(att1_v)
        x_e = self.att0_dropout(att0) + self.att1_dropout(att1_e)

        # MLP3
        pe3_l1 = self.pe3(torch.ones(1, dtype=torch.long, device=x.device)).view(
            1, self.inner_dim
        )
        if (
            self.max_l < len(edge_orders) and self.hyper_dropout == 0
        ) or self.force_broadcast:
            indices = torch.arange(self.pe3.max_pos, device=x.device)
            pe3_l = self.pe3(indices)[edge_orders]
        else:
            pe3_l = self.pe3(edge_orders)

        t_v = self.norm3(x_v)
        t_e = self.norm3(x_e)
        x_v = x_v + self.mlp3(torch.cat((t_v, pe3_l1.expand(t_v.shape)), dim=-1))
        x_e = x_e + self.mlp3(torch.cat((t_e, pe3_l.expand(t_e.shape)), dim=-1))

        x = x_v, x_e
        x = self.b(x, edge_orders)
        return x


class SelfAttnE2V(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_qk,
        n_heads,
        hypernet_info,
        inner_dim,
        pe_dim,
        hyper_dim,
        hyper_layers,
        hyper_dropout,
        force_broadcast,
        att0_dropout,
        att1_dropout,
    ):
        super().__init__()
        max_k, _, _ = hypernet_info
        self.max_k = max_k
        self.dim_in = dim_in
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        self.dim_v = inner_dim
        self.inner_dim = inner_dim
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.dim_v_head = self.dim_v // n_heads if self.dim_v >= n_heads else 1
        self.q = PositionalMLP(
            n_heads * self.dim_qk_head,
            2,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
        )
        self.k = nn.Linear(self.dim_v, 2 * dim_qk)
        self.v = nn.Linear(self.dim_v, self.dim_v)

        self.mlp1 = MLP_inner(dim_in * 2, self.dim_v, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP_inner(
            self.dim_v * 2, self.dim_v, hyper_dim, hyper_layers, hyper_dropout
        )
        self.mlp3 = MLP_inner(self.dim_v, dim_in, hyper_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.pe1 = PositionalEncoding(dim_in, max_k + 1)
        self.pe2 = PositionalEncoding(inner_dim, 2)
        self.b = BiasV(dim_in)

        self.att0_dropout = nn.Dropout(att0_dropout)
        self.att1_dropout = nn.Dropout(att1_dropout)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.b.reset_parameters()

    def forward(self, x, ehnn_cache):
        incidence = ehnn_cache["incidence"]
        edge_orders = ehnn_cache["edge_orders"]
        indices_with_nodes = ehnn_cache["indices_with_nodes"]

        # MLP1
        x_v, x_e = x
        pe1_k1 = self.pe1(torch.ones(1, dtype=torch.long, device=x_v.device)).view(
            1, self.dim_in
        )
        if (
            self.max_k < len(edge_orders) and self.hyper_dropout == 0
        ) or self.force_broadcast:
            indices = torch.arange(self.pe1.max_pos, device=x_v.device)
            pe1_k = self.pe1(indices)[edge_orders]
        else:
            pe1_k = self.pe1(edge_orders)

        t_v = self.norm1(x_v)
        t_e = self.norm1(x_e)
        x_v = x_v + self.mlp1(torch.cat((t_v, pe1_k1.expand(t_v.shape)), dim=-1))
        x_e = x_e + self.mlp1(torch.cat((t_e, pe1_k.expand(t_e.shape)), dim=-1))

        q_0 = self.q(torch.zeros(1, dtype=torch.long, device=x_v.device)).view(
            self.n_heads, self.dim_qk_head
        )
        q_1 = self.q(torch.ones(1, dtype=torch.long, device=x_v.device)).view(
            self.n_heads, self.dim_qk_head
        )
        k_v, k_e = self.k(x_v), self.k(x_e)
        k_v_0, k_v_1 = k_v.split(self.dim_qk, -1)
        k_e_0, k_e_1 = k_e.split(self.dim_qk, -1)
        k_v_0 = torch.stack(k_v_0.split(self.dim_qk_head, -1), 1)
        k_v_1 = torch.stack(k_v_1.split(self.dim_qk_head, -1), 1)
        k_e_0 = torch.stack(k_e_0.split(self.dim_qk_head, -1), 1)
        k_e_1 = torch.stack(k_e_1.split(self.dim_qk_head, -1), 1)
        k_0 = torch.cat([k_e_0, k_v_0])
        v_v, v_e = self.v(x_v), self.v(x_e)
        v_v = torch.stack(v_v.split(self.dim_v_head, -1), 1)
        v_e = torch.stack(v_e.split(self.dim_v_head, -1), 1)

        # aggregation
        v = torch.cat([v_e, v_v])
        
        # 0-overlap
        logit0 = torch.einsum("hd,ehd->eh", q_0, k_0) / math.sqrt(
            self.dim_qk_head
        )
        alpha0 = torch.softmax(logit0, dim=0)
        att0 = torch.einsum("eh,ehd->hd", alpha0, v)
        att0 = torch.cat(att0.unbind(-2), -1)
        att0 = att0[None, :]
        
        # 1-overlap
        logit1 = torch.einsum("hd,nhd->nh", q_1, torch.cat([k_e_1, k_v_1])) / math.sqrt(
            self.dim_qk_head
        )
        n, e = incidence.size()
        logit1 = torch.sparse_coo_tensor(
            indices_with_nodes,
            logit1[indices_with_nodes[1]],
            size=(n, e + n, self.n_heads),
        )
        alpha1 = torch.sparse.softmax(logit1, dim=1).coalesce()
        att1 = util_sparse_einsum(alpha1, v)
        att1 = torch.cat(att1.unbind(-2), -1)

        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=x_v.device)).view(
            1, self.inner_dim
        )
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x_v.device)).view(
            1, self.inner_dim
        )

        t0 = self.norm2(att0)
        t1 = self.norm2(att1)
        att0 = att0 + self.mlp2(torch.cat((t0, pe2_s0.expand(t0.shape)), dim=-1))
        att1 = att1 + self.mlp2(torch.cat((t1, pe2_s1.expand(t1.shape)), dim=-1))

        x = self.att0_dropout(att0) + self.att1_dropout(att1)
        
        # MLP3
        x = x + self.mlp3(self.norm3(x))
        x = self.b(x)
        return x


class TransformerV2E(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_qk,
        dim_ff,
        n_heads,
        dropout,
        hypernet_info,
        inner_dim,
        pe_dim,
        hyper_dim,
        hyper_layers,
        hyper_dropout,
        force_broadcast,
        att0_dropout,
        att1_dropout,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim_in)
        self.attn = SelfAttnV2E(
            dim_in,
            dim_qk,
            n_heads,
            hypernet_info,
            inner_dim,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
            att0_dropout,
            att1_dropout,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_ff),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_in),
        )

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.attn.reset_parameters()
        for layer in self.ffn:
            if isinstance(layer, nn.LayerNorm):
                layer.reset_parameters()
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, ehnn_cache):
        h = self.ln(x)
        h_v, h_e = self.attn(h, ehnn_cache)
        x_v, x_e = (x + h_v, h_e)
        h_v, h_e = (self.ffn(x_v), self.ffn(x_e))
        x_v, x_e = (x_v + h_v, x_e + h_e)
        return x_v, x_e


class TransformerE2V(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_qk,
        dim_ff,
        n_heads,
        dropout,
        hypernet_info,
        inner_dim,
        pe_dim,
        hyper_dim,
        hyper_layers,
        hyper_dropout,
        force_broadcast,
        att0_dropout,
        att1_dropout,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim_in)
        self.attn = SelfAttnE2V(
            dim_in,
            dim_qk,
            n_heads,
            hypernet_info,
            inner_dim,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
            att0_dropout,
            att1_dropout,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_ff),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_in),
        )

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.attn.reset_parameters()
        for layer in self.ffn:
            if isinstance(layer, nn.LayerNorm):
                layer.reset_parameters()
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:
        x_v, x_e = x
        h = (self.ln(x_v), self.ln(x_e))
        h = self.attn(h, ehnn_cache)
        x = x_v + h
        h = self.ffn(x)
        x = x + h
        return x


class PositionalEncoding(nn.Module):
    """https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, dim, max_pos):
        super().__init__()
        self.max_pos = max_pos
        position = torch.arange(max_pos).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_pos, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.pe[x]


class PositionalMLP(nn.Module):
    def __init__(self, dim_out, max_pos, dim_pe, dim_hidden, n_layers, dropout):
        super().__init__()
        self.max_pos = max_pos
        self.pe = PositionalEncoding(dim_pe, max_pos)
        self.input = nn.Linear(dim_pe, dim_hidden)
        self.mlp = MLP_inner(dim_hidden, dim_out, dim_hidden, n_layers, dropout)

    def reset_parameters(self):
        self.input.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.input(self.pe(x))
        x = self.mlp(x)
        return x


class PositionalWeight(nn.Module):
    def __init__(self, max_pos, dim_in, dim_out):
        super().__init__()
        self.max_pos = max_pos
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = nn.Parameter(torch.Tensor(max_pos + 1, dim_in, dim_out))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_out)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.weights[x].view(x.size(0), -1)


class BiasE(nn.Module):
    def __init__(self, dim_out, max_l, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast):
        super().__init__()
        self.dim_out = dim_out
        self.max_l = max_l
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        self.b = PositionalMLP(dim_out, max_l + 1, pe_dim, hyper_dim, hyper_layers, hyper_dropout)

    def reset_parameters(self):
        self.b.reset_parameters()

    def forward(self, x, edge_orders):
        x_v, x_e = x
        if (self.max_l and self.hyper_dropout == 0) or self.force_broadcast:
            indices = torch.arange(self.b.max_pos, device=x_v.device)
            b = self.b(indices)[edge_orders]
        else:
            b = self.b(edge_orders)
        b_1 = self.b(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, self.dim_out)
        return x_v + b_1, x_e + b


class BiasV(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.b = nn.Parameter(torch.Tensor(1, dim_out))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_out)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return x + self.b
