import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainUnigencoder(nn.Module):

    def __init__(self, num_features, num_targets, args):
        super(PlainUnigencoder, self).__init__()
        
        self.in_channels = num_features
        self.hidden_channels = args.MLP_hidden
        self.out_channels = num_targets
        self.num_layers = args.All_num_layers
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(self.in_channels, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.lins.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(nn.Linear(self.hidden_channels, self.out_channels))
        self.cls = nn.Linear(self.out_channels, self.out_channels)
        self.dropout = args.dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, data):
        x = data.x
        Pv, PvT = data.Pv, data.PvT
        x = torch.spmm(Pv, x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = torch.spmm(PvT, x)
        return x, None
