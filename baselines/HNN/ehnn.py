import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
from .ehnn_linear import EHNNLinearConv
from .ehnn_transformer import EHNNTransformerConv

class EHNN(nn.Module):
    
    def __init__(self, num_features, num_targets, args, ehnn_cache):
        super().__init__()
        edge_orders = ehnn_cache["edge_orders"]  # [|E|,]
        overlaps = ehnn_cache["overlaps"]  # [|overlaps|,]
        max_edge_order = int(edge_orders.max().item())
        max_overlap = int(overlaps.max().item()) if overlaps is not None else 0
        hypernet_info = (max_edge_order, max_edge_order, max_overlap)
        
        self.num_layers = args.All_num_layers
        self.hidden_dim = args.ehnn_hidden_channel
        self.ehnn_type=args.ehnn_type

        self.dropout=args.dropout
        self.decoder_hidden=args.decoder_hidden
        self.decoder_num_layers =args.decoder_num_layers
        self.decoder_norm=args.normalization 

        self.convs = nn.ModuleList()
        if self.ehnn_type == "linear":
            if self.num_layers == 1:
                self.convs.append(EHNNLinearConv(num_features,self.hidden_dim,
                    args.ehnn_inner_channel,args.dropout,hypernet_info,args.ehnn_pe_dim,
                    args.ehnn_hyper_dim,args.ehnn_hyper_layers,
                    args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier)))
            else:
                self.convs.append(EHNNLinearConv(num_features,self.hidden_dim,
                    args.ehnn_inner_channel,args.dropout,hypernet_info,args.ehnn_pe_dim,
                    args.ehnn_hyper_dim,args.ehnn_hyper_layers,
                    args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier)))
                for _ in range(self.num_layers-2):
                    self.convs.append(EHNNLinearConv(self.hidden_dim,self.hidden_dim,
                    args.ehnn_inner_channel,args.dropout,hypernet_info,args.ehnn_pe_dim,
                    args.ehnn_hyper_dim,args.ehnn_hyper_layers,
                    args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier)))
                # Output heads is set to 1 as default
                self.convs.append(EHNNLinearConv(self.hidden_dim,self.hidden_dim,
                    args.ehnn_inner_channel,args.dropout,hypernet_info,args.ehnn_pe_dim,
                    args.ehnn_hyper_dim,args.ehnn_hyper_layers,
                    args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier)))
            
        elif self.ehnn_type == "transformer":
            if self.num_layers == 1:
                self.convs.append(EHNNTransformerConv(num_features,self.hidden_dim,
                    args.ehnn_qk_channel,args.ehnn_hidden_channel,args.ehnn_n_heads,
                    args.dropout,hypernet_info,args.ehnn_inner_channel,args.ehnn_pe_dim,args.ehnn_hyper_dim,
                    args.ehnn_hyper_layers,args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier),
                    args.ehnn_att0_dropout,args.ehnn_att1_dropout))
            else:
                self.convs.append(EHNNTransformerConv(num_features,self.hidden_dim,
                    args.ehnn_qk_channel,args.ehnn_hidden_channel,args.ehnn_n_heads,
                    args.dropout,hypernet_info,args.ehnn_inner_channel,args.ehnn_pe_dim,args.ehnn_hyper_dim,
                    args.ehnn_hyper_layers,args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier),
                    args.ehnn_att0_dropout,args.ehnn_att1_dropout))
                for _ in range(self.num_layers-2):
                    self.convs.append(EHNNTransformerConv(self.hidden_dim,self.hidden_dim,
                    args.ehnn_qk_channel,args.ehnn_hidden_channel,args.ehnn_n_heads,
                    args.dropout,hypernet_info,args.ehnn_inner_channel,args.ehnn_pe_dim,args.ehnn_hyper_dim,
                    args.ehnn_hyper_layers,args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier),
                    args.ehnn_att0_dropout,args.ehnn_att1_dropout))
                self.convs.append(EHNNTransformerConv(self.hidden_dim,self.hidden_dim,
                    args.ehnn_qk_channel,args.ehnn_hidden_channel,args.ehnn_n_heads,
                    args.dropout,hypernet_info,args.ehnn_inner_channel,args.ehnn_pe_dim,args.ehnn_hyper_dim,
                    args.ehnn_hyper_layers,args.ehnn_hyper_dropout,ast.literal_eval(args.ehnn_force_broadcast),
                    args.ehnn_input_dropout,ast.literal_eval(args.ehnn_mlp_classifier),
                    args.ehnn_att0_dropout,args.ehnn_att1_dropout))
        else:
            raise NotImplementedError

        self.decoder = MLP(
            in_channels=self.hidden_dim,
            hidden_channels=self.decoder_hidden,
            out_channels=num_targets,
            num_layers=self.decoder_num_layers,
            dropout=self.dropout,
            Normalization=self.decoder_norm,
            InputNorm=False,
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data):
        """forward method
        :param data:
        :param ehnn_cache:
        :return: [N, C] dense
        """
        x = data.x
        ehnn_cache = data.ehnn_cache
        
        if self.ehnn_type == 'transformer':
            with torch.no_grad():
                ehnn_cache["incidence"] = ehnn_cache["incidence"].coalesce()
                n, e = ehnn_cache["incidence"].size()
                node_indices = torch.arange(0, n, device=ehnn_cache["incidence"].device)[
                    None, :
                ].repeat(2, 1)
                node_indices[1] += e
                ehnn_cache["indices_with_nodes"] = torch.cat(
                    (ehnn_cache["incidence"].indices(), node_indices), dim=1
                )
        
        for i, conv in enumerate(self.convs[:-1]):
            x,_ = conv(x,ehnn_cache) 
            x = F.dropout(x, p=self.dropout, training=self.training)

        x,x_e = self.convs[-1](x, ehnn_cache)
        x = self.decoder(x)

        return x,x_e

