
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import sys
import copy
import random
import torch.optim as optim
import pickle
import os
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from tqdm import trange, tqdm
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import dhg
from dhg.data import CocitationCora, Cooking200
from dhg import Hypergraph
from dhg.nn import HGNNConv
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from scatter_func import scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score

import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from scipy.sparse import csr_matrix
import gc
torch.manual_seed(42)
np.random.seed(42)
from utils import accuracy, normalize_features

import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg
# from .layers import HGNNConv
from dhg.nn import HGNNConv
import math


class HGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X
    

class LAHGCN(nn.Module):
    def __init__(self, concat, in_channels, hid_channels, num_classes, dropout):
        super(LAHGCN, self).__init__()

        self.hgcn1_list = nn.ModuleList()
        for _ in range(concat):
            self.hgcn1_list.append(HGNNConv(in_channels, hid_channels))
        self.hgc2 = HGNNConv(concat*hid_channels, num_classes)
        self.dropout = dropout

    def forward(self, x_list, hg):
        hidden_list = []
        for k, con in enumerate(self.hgcn1_list):
            x = F.dropout(x_list[k], self.dropout, training=self.training)
            hidden_list.append(F.relu(con(x, hg)))
        x = torch.cat((hidden_list), dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, hg)
        # print(x.shape)
        return x


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, conditional=False, conditional_size=0):
        super(CVAE, self).__init__()
        self.conditional = conditional
        # if self.conditional:
        #     latent_dim += conditional_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.latent_dim = latent_dim

        # hypergraph convolution
        # self.hg_conv = HGNNConv(input_dim+conditional_size, hidden_dim)
        self.fc1 = nn.Linear(input_dim+conditional_size, hidden_dim)
        # Encoder
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(input_dim+latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)
        h1 = F.relu(self.fc1(x))
        return self.mu(h1), self.sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)
        
        h2 = self.fc2(z)
        h2 = F.relu(h2)
        return F.sigmoid(self.fc3(h2))
        

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        return self.decode(z, c), mu, logvar, z
    
    def inference(self, z, c):
        recon_x = self.decode(z, c)
        return recon_x

class HGNNCVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, out_channels, conditional, conditional_size):
        super(HGNNCVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # CVAE
        self.cvae = CVAE(input_dim, hidden_dim, latent_dim, conditional, conditional_size)
        
        # HGNNConv
        self.hgnnconv = HGNNConv(input_dim+latent_dim, out_channels)
        
    def forward(self, X, hg):
        # CVAE
        z, mu, logvar = self.cvae(X)
        
        # Concatenate the original input features and the generated features
        X_augmented = torch.cat([X, z], dim=1)
        
        # HGNNConv
        out = self.hgnnconv(X_augmented, hg)
        
        return out

def adjacency_matrix(hg, s=1, weight=False):
        r"""
        The :term:`s-adjacency matrix` for the dual hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix

        """

        tmp_H = hg.H.to_dense().numpy()
        A = tmp_H @ (tmp_H.T)
        A[np.diag_indices_from(A)] = 0
        if not weight:
            A = (A >= s) * 1

        del tmp_H
        gc.collect()

        return csr_matrix(A)

def feature_tensor_normalize(feature):
    # feature = torch.tensor(feature)
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature

# Adapted from GRAND: https://github.com/THUDM/GRAND
def consis_loss(logps):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./0.5) / torch.sum(torch.pow(avg_p, 1./0.5), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return 1.0 * loss

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

def normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def neighbor_of_node(adj_matrix, node):
    # 找到邻接矩阵中节点i对应的行
    node_row = adj_matrix[node, :].toarray().flatten()

    # 找到非零元素对应的列索引，即邻居节点
    neighbors = np.nonzero(node_row)[0]
    return neighbors.tolist()

def aug_features_concat(concat, features, cvae_model):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), 8]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        print("6"*100)
        print(augmented_features, augmented_features.shape, type(augmented_features))
        augmented_features = feature_tensor_normalize(augmented_features).detach()
        
        X_list.append(augmented_features.to(device))
        
    return X_list

def get_augmented_features(args, hg, features, labels, idx_train, features_normalized, device):
    adj = adjacency_matrix(hg, s=1, weight=False)
    x_list, c_list = [], []
    for i in trange(adj.shape[0]):
        neighbors = neighbor_of_node(adj, i)
        if len(neighbors) == 0:
            neighbors = [i]
        # print(neighbors)
        # neighbors = neighbors[0]
        v_deg= hg.D_v
        if len(neighbors) != 1:
            neighbors = torch.argsort(v_deg.values()[neighbors], descending=True)[:math.floor(len(neighbors)/2)]
        x = features[neighbors]
        x = x.numpy().reshape(x.shape[0],x.shape[1])
        c = np.tile(features[i], (x.shape[0], 1))
        # print(x.shape, c.shape)
        x_list.append(x)
        c_list.append(c)
    
    features_x = np.vstack(x_list)
    features_c = np.vstack(c_list)
    
    del x_list
    del c_list
    gc.collect()

    features_x = torch.tensor(features_x, dtype=torch.float32)
    features_c = torch.tensor(features_c, dtype=torch.float32)

    cvae_features = torch.tensor(features, dtype=torch.float32)
    
    cvae_dataset = TensorDataset(features_x, features_c)
    
    cvae_dataset_sampler = RandomSampler(cvae_dataset)
    cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=32)

    # print('\n')
    # print(len(cvae_dataset_dataloader))
    # print('\n')

    hidden = 64
    dropout = 0.5
    lr = 0.01
    weight_decay = 5e-4
    epochs = 400
    
    model = HGNN(in_channels=features.shape[1], hid_channels=hidden, num_classes=labels.max().item()+1, use_bn=False, drop_rate=dropout)
    model_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    features_normalized = features_normalized.to(device)
    hg = hg.to(device)
    cvae_features = cvae_features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)

    for _ in range(int(epochs / 2)):
        model.train()
        model_optimizer.zero_grad()
        output = model(features_normalized, hg)
        output = torch.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        model_optimizer.step()

    
    # pretrain
    cvae = CVAE(features.shape[1], 256, args.latent_size, True, features.shape[1])
    # print(cvae)
    # cvae = CVAE(features.shape[1], 256, 64, False, 0)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)
    cvae.to(device)

    t = 0
    best_augmented_features = None
    cvae_model = CVAE(features.shape[1], 256, args.latent_size, True, features.shape[1])
    best_score = -float("inf")
    for epoch in trange(args.pretrain_epochs, desc='Run CVAE Train'): # 遍历预训练的epoch数
        for _, (x, c) in enumerate(tqdm(cvae_dataset_dataloader)): # 遍历CVAE的数据加载器
            # print(x.shape, c.shape)
            cvae.train()
            # x, c, H = x.to(device), c.to(device), H.to(device)
            # num_v = H.size(0)
            # e_list = []
            # for i in range(H.size(1)):
            #     node_idx = torch.nonzero(H[:,i]).squeeze(1)
            #     e_list.append(node_idx)
            # hg = Hypergraph(num_v, e_list)
        # cvae.train()
        # x, c = features_x.to(device), features_c.to(device)
            recon_x, mean, log_var, _ = cvae(x, c)
            cvae_loss = loss_fn(recon_x, x, mean, log_var)

            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()

            z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
            augmented_feats = cvae.inference(z, cvae_features)
            augmented_feats = feature_tensor_normalize(augmented_feats)
            print('='*50)
            print(augmented_feats, augmented_feats.shape)

            total_logits = 0
            cross_entropy = 0
            for i in range(args.num_models):
                logits = model(augmented_feats, hg)
                total_logits += F.softmax(logits, dim=1)
                output = F.log_softmax(logits, dim=1)
                cross_entropy += F.nll_loss(output[idx_train], labels[idx_train])
            output = torch.log(total_logits / args.num_models)
            U_score = F.nll_loss(output[idx_train], labels[idx_train]) - cross_entropy / args.num_models # 计算HGNN模型在增强特征上的损失
            t += 1
            print("Epoch: ", epoch, "U Score: ", U_score, " Best Score: ", best_score)
            if U_score > best_score: 
                best_score = U_score # 更新最新best_score和cvae_model
                if t > args.warmup: # 达到一定预热期，开始更新HGNN模型 early-stopping
                    cvae_model = copy.deepcopy(cvae)
                    print("U_score: ", U_score, " t: ", t)
                    best_augmented_features = augmented_feats.clone().detach().requires_grad_(True)
                    # best_augmented_features = augmented_feats
                    for i in range(args.update_epochs):
                        model.train()
                        model_optimizer.zero_grad()
                        output = model(best_augmented_features, hg)
                        output = torch.log_softmax(output, dim=1)
                        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                        loss_train.backward()
                        model_optimizer.step()
                    print('*'* 50)
                    print(best_augmented_features)
                    # best_augmented_features = torch.tensor(best_augmented_features)

    torch.save(cvae_model.state_dict(), "cvae_model_pre.pth") # 整个训练过程结束后，保存与训练得到的CVAE模型和最佳增强特征
    torch.save(best_augmented_features,'cvae_model_best_features.pt')

    return best_augmented_features, cvae_model


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # Cocitation Cora Data
# data = Cooking200()
# hg = Hypergraph(data["num_vertices"], data["edge_list"])
# print(hg)
# train_mask = data["train_mask"]
# val_mask = data["val_mask"]
# test_mask = data["test_mask"]

# idx_train = np.where(train_mask)[0]
# idx_val = np.where(val_mask)[0]
# idx_test = np.where(test_mask)[0]

# # for cooking200
# v_deg= hg.D_v
# X = v_deg.to_dense()/torch.max(v_deg.to_dense())
# # X = data["features"]
# # Normalize adj and features
# features = X.numpy()
# features_normalized = normalize_features(features)
# labels = data["labels"]

# features_normalized = torch.FloatTensor(features_normalized)
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

# # input_dim = X.shape[1]
# # hidden_dim = 256
# # latent_dim = 128
# # out_channels = (torch.max(labels)+1).item()

# cvae_model = CVAE(features.shape[1], 256, 8, True, features.shape[1])
# augmenteed_feats, _ = get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)
# cvae_model = torch.load("cooking200_cocitation.pth")

# all_val = []
# all_test = []
# for i in trange(5, desc='Run Train'):
#     # Model and optimizer
#     model = LAHGCN(concat=5,
#                   in_channels=features.shape[1],
#                   hid_channels=8,
#                   num_classes=labels.max().item() + 1,
#                   dropout=0.5)
#     optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#     model.to(device)

#     # train model
#     best = 99999999
#     best_model = None
#     best_X_list = None
#     for epoch in range(200):
#         model.train()
#         optimizer.zero_grad()

#         output_list = []
#         for k in range(100):
#             X_list = aug_features_concat(5, features, cvae_model)
#             # print(len(X_list), features_normalized.shape)
#             output_list.append(torch.log_softmax(model(X_list + [features_normalized], hg), dim=1))

#         loss_train = 0.
#         for k in range(len(output_list)):
#             loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])

#         loss_train = loss_train / len(output_list)
#         loss_consis = consis_loss(output_list)
#         loss_train = loss_train + loss_consis

#         loss_train.backward()
#         optimizer.step()

#         model.eval()
#         val_X_list = aug_features_concat(5, features, cvae_model)
#         output = model(val_X_list, hg)
#         output = torch.log_softmax(output, dim=1)
#         loss_val = F.nll_loss(output[idx_val], labels[idx_val])

#         if loss_val < best:
#             best = loss_val
#             best_model = copy.deepcopy(model)
#             best_X_list = copy.deepcopy(val_X_list)

#     # validate and test
#     best_model.eval()
#     output = best_model(best_X_list+[features_normalized], hg)
#     output = torch.log_softmax(output, dim=1)
#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     acc_test = accuracy(output[idx_test], labels[idx_test])

#     all_val.append(acc_val.item())
#     all_test.append(acc_test.item())

# print(np.mean(all_val), np.std(all_val), np.mean(all_test), np.std(all_test))

