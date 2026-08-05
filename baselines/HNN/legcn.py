import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_scatter

class LEGCN(nn.Module):

    def __init__(self, num_features, num_targets, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(LEGCN, self).__init__()
        self.hidden_dim = args.MLP_hidden
        self.num_layers = args.All_num_layers
        self.dropout=args.dropout
        self.convs = nn.ModuleList([])
        if self.num_layers == 1:
            self.convs.append(GCNConv(in_channels=num_features, out_channels=num_targets))
        else:
            self.convs.append(GCNConv(in_channels=num_features, out_channels=self.hidden_dim))
            for l in range(self.num_layers - 2):
                self.convs.append(GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim))
            self.convs.append(GCNConv(in_channels=self.hidden_dim, out_channels=num_targets))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        le_adj = data.le_adj
        edge_index = data.hyperedge_index
        x = data.x[edge_index[0]]
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, le_adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, le_adj)
        x = torch_scatter.scatter(x, edge_index[0], dim=0, reduce='mean')
        return x,None 