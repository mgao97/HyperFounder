import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange

from hgcn import *
from scipy.sparse import csr_matrix, diags
import scipy.sparse as sp
from hgcn.models import HGNN
from cvae_models import VAE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import copy
import dhg


# Training settings
exc_path = sys.path[0]


def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)


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

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def neighbor_of_node(adj_matrix, node):
    # 找到邻接矩阵中节点i对应的行
    node_row = adj_matrix[node, :].toarray().flatten()

    # 找到非零元素对应的列索引，即邻居节点
    neighbors = np.nonzero(node_row)[0]

    return neighbors

# adj_normalized = normalize_adj(adj_matrix)

# # -----------------------超图归一化-----------------------
# num_nodes = adj_normalized.shape[0]
# edges = []

# for i in range(num_nodes):
#     neighbors = np.nonzero(adj_matrix[i])[0].tolist()
#     edges.append(neighbors)

# hg = dhg.Hypergraph(num_nodes=num_nodes, edges=edges)
# # -----------------------超图归一化-----------------------

def generated_generator(args, device, hg, features, labels, features_normalized, idx_train):
    
    x_list, c_list = [], []

    adj_matrix = adjacency_matrix(hg, s=1, weight=False)
    

    for i in hg.v:
        # 获取节点关联的邻居节点
        neighbors = neighbor_of_node(adj_matrix, i)

        # 如果节点没有邻居，则将自己添加到邻居中，并使用自身的特征
        if len(neighbors) == 0:
            neighbors = [i]

        # 获取节点的特征
        x = torch.index_select(features, 0, torch.LongTensor(neighbors))
        
        # 重复节点的特征以匹配邻居数量
        # c = features[i].view(1, -1).expand(len(neighbors), -1)
        c = torch.cat([features[i].view(1, -1)] * len(neighbors), dim=0)

        x_list.append(x)
        c_list.append(c)
        
        
    features_x = torch.cat(x_list, dim=0)
    features_c = torch.cat(c_list, dim=0)
    del x_list
    del c_list
    gc.collect()

    features_x = features_x.to(device, dtype=torch.float32)
    features_c = features_c.to(device, dtype=torch.float32)

    cvae_features = features.to(device, dtype=torch.float32)

    cvae_dataset = TensorDataset(features_x, features_c)
    cvae_dataset_sampler = RandomSampler(cvae_dataset)
    cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=args.batch_size)

    hidden = 8
    dropout = 0.5
    lr = 0.001
    weight_decay = 5e-4
    epochs = 200

    model = HGNN(in_channels=features.shape[1], hid_channels=hidden, num_classes=labels.max().item()+1, use_bn=False, drop_rate=dropout)
    model_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.cuda:
        model.to(device)
        features_normalized = features_normalized.to(device)
        # adj_normalized = adj_normalized.to(device)
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

    # pretain
    cvae = VAE(encoder_layer_sizes=[features.shape[1], 256],
               latent_size=args.latent_size,
               decoder_layer_sizes=[256, features.shape[1]],
               conditional=args.conditional, 
               conditional_size=features.shape[1])
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

    if args.cuda:
        cvae.to(device)

    # Pretrain
    t = 0
    best_augmented_features  = None
    cvae_model = None
    best_score = -float("inf")
    for _ in trange(args.pretrain_epochs, desc='Run CVAE Train'):
        for _, (x, c) in enumerate(tqdm(cvae_dataset_dataloader)):
            cvae.train()
            x, c = x.to(device), c.to(device)
            if args.conditional:
                recon_x, mean, log_var, _ = cvae(x, c)
            else:
                recon_x, mean, log_var, _ = cvae(x)
            cvae_loss = loss_fn(recon_x, x, mean, log_var)

            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()

            
            z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
            
            augmented_features = cvae.inference(z, cvae_features)
            augmented_features = feature_tensor_normalize(augmented_features).detach()
            
            total_logits = 0
            cross_entropy = 0
            for i in range(args.num_models):
                logits = model(augmented_features, hg)
                total_logits += F.softmax(logits, dim=1)
                output = F.log_softmax(logits, dim=1)
                cross_entropy += F.nll_loss(output[idx_train], labels[idx_train])
            output = torch.log(total_logits / args.num_models)
            U_score = F.nll_loss(output[idx_train], labels[idx_train]) - cross_entropy / args.num_models
            t += 1
            print("U Score: ", U_score, " Best Score: ", best_score)
            if U_score > best_score:
                best_score = U_score
                if t > args.warmup:
                    cvae_model = copy.deepcopy(cvae)
                    print("U_score: ", U_score, " t: ", t)
                    best_augmented_features = copy.deepcopy(augmented_features)
                    for i in range(args.update_epochs):
                        model.train()
                        model_optimizer.zero_grad()
                        output = model(best_augmented_features, hg)
                        output = torch.log_softmax(output, dim=1)
                        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                        loss_train.backward()
                        model_optimizer.step() 


    return best_augmented_features, cvae_model

