# from __future__ import division
# from __future__ import print_function
# import argparse
# import numpy as np
# import scipy.sparse as sp
# import torch
# import sys
# import copy
# import random
# import torch.nn.functional as F
# import torch.optim as optim
# import hgnn_cvae_pretrain_new_cora

# import time
# from copy import deepcopy
# # from config import config
# import torch
# import torch.optim as optim
# import torch.nn.functional as F

# from dhg import Hypergraph
# from dhg.data import CocitationCora, Cooking200
# # from data_load_utils import *
# # from dhg.models import HGNN, LAHGCN
# # from dhg.random import set_seed
# from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

# from utils import accuracy, normalize_features, micro_f1, macro_f1, sparse_mx_to_torch_sparse_tensor, normalize_adj
# from hgcn import *
# from hgcn.models import HGNN, LAHGCN
# from tqdm import trange
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler
# exc_path = sys.path[0]
# import os, torch, numpy as np
# import hgnn_cvae_pretrain_new_cora
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import warnings
# warnings.filterwarnings("ignore")

# parser = argparse.ArgumentParser()
# parser.add_argument("--samples", type=int, default=4)
# parser.add_argument("--concat", type=int, default=10)
# parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

# parser.add_argument("--latent_size", type=int, default=10)
# parser.add_argument('--dataset', default='cora', help='Dataset string.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
# parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
# parser.add_argument('--lam', type=float, default=1., help='Lamda')
# parser.add_argument("--pretrain_epochs", type=int, default=15)
# parser.add_argument("--pretrain_lr", type=float, default=0.05)
# parser.add_argument("--conditional", action='store_true', default=True)
# parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
# parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
# parser.add_argument('--warmup', type=int, default=200, help='Warmup')
# # parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

# args = parser.parse_args()

# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# args.cuda = torch.cuda.is_available()

# print('args\n', args)

# # Adapted from GRAND: https://github.com/THUDM/GRAND
# def consis_loss(logps, temp=args.tem):
#     ps = [torch.exp(p) for p in logps]
#     sum_p = 0.
#     for p in ps:
#         sum_p = sum_p + p
#     avg_p = sum_p/len(ps)
#     #p2 = torch.exp(logp2)
    
#     sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
#     loss = 0.
#     for p in ps:
#         loss += torch.mean((p-sharp_p).pow(2).sum(1))
#     loss = loss/len(ps)
#     return args.lam * loss


# # Load data
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

# # args = config.parse()



# # seed
# # torch.manual_seed(args.seed)
# # np.random.seed(args.seed)



# # gpu, seed
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
# # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# os.environ['PYTHONHASHSEED'] = str(args.seed)



# # load data
# data = CocitationCora()
# print(data)

# hg = Hypergraph(data["num_vertices"], data["edge_list"])
# # train_mask = data["train_mask"]
# # val_mask = data["val_mask"]
# # test_mask = data["test_mask"]

# # 设置随机种子，以确保结果可复现
# random_seed = 42

# node_idx = [i for i in range(data['num_vertices'])]
# # 将idx_test划分为训练（60%）、验证（20%）和测试（20%）集
# idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
# idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# # 确保划分后的集合没有重叠
# assert len(set(idx_train) & set(idx_val)) == 0
# assert len(set(idx_train) & set(idx_test)) == 0
# assert len(set(idx_val) & set(idx_test)) == 0

# # idx_train = np.where(train_mask)[0]
# # idx_val = np.where(val_mask)[0]
# # idx_test = np.where(test_mask)[0]

# # v_deg= hg.D_v
# # X = v_deg.to_dense()/torch.max(v_deg.to_dense())

# X = data["features"]

# # Normalize adj and features
# features = data["features"].numpy()
# features = X.numpy()
# features_normalized = normalize_features(features)
# # nor_hg = normalize_adj(hg)

# # To PyTorch Tensor
# # labels = torch.LongTensor(labels)
# # labels = torch.max(labels, dim=1)[1]
# labels = data["labels"]
# features_normalized = torch.FloatTensor(features_normalized)

# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)


# train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# train_mask[idx_train] = True
# val_mask[idx_val] = True
# test_mask[idx_test] = True

# cvae_model = torch.load("{}/model/{}_1208.pkl".format(exc_path, args.dataset))

# # best_augmented_features, cvae_model = hgnn_cvae_pretrain_new_cora.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)

# def get_augmented_features(concat):
#     X_list = []
#     cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
#     for _ in range(concat):
#         z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
#         augmented_features = cvae_model.inference(z, cvae_features)
#         augmented_features = hgnn_cvae_pretrain_new_cora.feature_tensor_normalize(augmented_features).detach()
#         if args.cuda:
#             X_list.append(augmented_features.to(device))
#         else:
#             X_list.append(augmented_features)
#     return X_list


# if args.cuda:
#     hg = hg.to(device)
#     labels = labels.to(device)
#     idx_train = idx_train.to(device)
#     idx_val = idx_val.to(device)
#     idx_test = idx_test.to(device)
#     features_normalized = features_normalized.to(device)

# all_val = []
# all_test = []

# all_test_microf1, all_test_macrof1 = [], []


# for i in trange(args.runs, desc='Run Train'):

#     # Model and optimizer
#     model = LAHGCN(concat=args.concat+1,
#                   in_channels=features.shape[1],
#                   hid_channels=args.hidden,
#                   num_classes=labels.max().item() + 1,
#                   use_bn=False,
#                   drop_rate=args.dropout)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     if args.cuda:
#         model.to(device)

#     # Train model
#     best = 999999999
#     best_model = None
#     best_X_list = None
#     for epoch in range(args.epochs):

#         model.train()
#         optimizer.zero_grad()

#         output_list = []
#         for k in range(args.samples):
#             X_list = get_augmented_features(args.concat)
#             output_list.append(torch.log_softmax(model(X_list+[features_normalized], hg), dim=-1))

#         loss_train = 0.
#         for k in range(len(output_list)):
#             loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
#         loss_train = loss_train/len(output_list)

#         loss_consis = consis_loss(output_list)
#         loss_train = loss_train + loss_consis

#         loss_train.backward()
#         optimizer.step()

#         model.eval()
#         val_X_list = get_augmented_features(args.concat)
#         output = model(val_X_list+[features_normalized],hg)
#         output = torch.log_softmax(output, dim=1)
#         loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        

#         print('Run:{:02d}'.format(i+1),
#               'Epoch: {:04d}'.format(epoch+1),
#               'loss_train: {:.4f}'.format(loss_train.item()),
#               'loss_val: {:.4f}'.format(loss_val.item()))
                
#         if loss_val < best:
#             best = loss_val
#             best_model = copy.deepcopy(model)
#             best_X_list = copy.deepcopy(val_X_list)
#             torch.save(best_model.state_dict(), 'model/lahgnn_cocitationciteseer_best_model_1224.pth')

#     #Validate and Test
#     best_model.eval()
#     output = best_model(best_X_list+[features_normalized], hg)
    

#     outs, lbl = output[idx_test], labels[idx_test]

#     output = torch.log_softmax(output, dim=1)

#     # Calculate accuracy
#     _, predicted = torch.max(outs, 1)
#     # 将predicted结果转换为numpy数组
#     predicted_array = predicted.cpu().numpy()

#     # 保存到文件
#     np.savetxt('res/lahgnn_predicted_cocitationcora.txt', predicted_array, fmt='%d')

#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     acc_test = accuracy(output[idx_test], labels[idx_test])

#     # micro_f1_val = micro_f1(output[idx_val], labels[idx_val])
#     # macro_f1_val = macro_f1(output[idx_val], labels[idx_val])
#     micro_f1_test = micro_f1(output[idx_test], labels[idx_test])
#     macro_f1_test = macro_f1(output[idx_test], labels[idx_test])

#     all_val.append(acc_val.item())
#     all_test.append(acc_test.item())
#     all_test_microf1.append(micro_f1_test.item())
#     all_test_macrof1.append(macro_f1_test.item())

# # print('val acc:', np.mean(all_val), 'val acc std:', np.std(all_val))
# # print('\n')
# print('test acc:', np.mean(all_test), 'test acc std', np.std(all_test))
# print('\n')
# print('test micro f1:', np.mean(all_test_microf1), 'test macro f1', np.mean(all_test_macrof1))


# def get_augmented_features(concat):
#     X_list = []
#     cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
#     for _ in range(concat):
#         z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
#         augmented_features = cvae_model.inference(z, cvae_features)
#         augmented_features = hgnn_cvae_pretrain_new_cora.feature_tensor_normalize(augmented_features).detach()
#         if args.cuda:
#             X_list.append(augmented_features.to(device))
#         else:
#             X_list.append(augmented_features)
#     return X_list

# X_list = get_augmented_features(args.concat)
# lahgnn_emb = model(X_list+[features_normalized],hg)


# tsne = TSNE(n_components=2, verbose=1, random_state=0)
# z = tsne.fit_transform(lahgnn_emb.detach().numpy())
# z_data = np.vstack((z.T, lbls)).T
# df_tsne = pd.DataFrame(z_data, columns=['Dimension 1', 'Dimension 2', 'Class'])
# df_tsne['Class'] = df_tsne['Class'].astype(int)
# plt.figure(figsize=(8, 8))
# sns.set(font_scale=1.5)
# plt.legend(loc='upper right')
# sns.scatterplot(data=df_tsne, hue='Class', x='Dimension 1', y='Dimension 2', palette=['green','orange','brown','red', 'blue','black','purple'])
# plt.savefig("figs/hgnn_cocitationcora_emb_1224.pdf", bbox_inches="tight") # save embeddings if needed
# plt.savefig("figs/hgnn_cocitationcora_emb_1224.png", bbox_inches="tight")
# plt.show()

from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import hgnn_cvae_pretrain_new_cora

import time
from copy import deepcopy
# from config import config
import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import CocitationCora, Cooking200
# from data_load_utils import *
# from dhg.models import HGNN, LAHGCN
# from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from utils import accuracy, normalize_features, micro_f1, macro_f1, sparse_mx_to_torch_sparse_tensor, normalize_adj
from hgcn import *
from hgcn.models import HGNN, LAHGCN
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
exc_path = sys.path[0]
import os, torch, numpy as np
import hgnn_cvae_pretrain_new_cora
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--concat", type=int, default=10)
parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument("--pretrain_epochs", type=int, default=15)
parser.add_argument("--pretrain_lr", type=float, default=0.05)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
# parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

print('args\n', args)

# Adapted from GRAND: https://github.com/THUDM/GRAND
def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss


# Load data
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

# args = config.parse()



# seed
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)



# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)



# load data
data = CocitationCora()
print(data)

hg = Hypergraph(data["num_vertices"], data["edge_list"])
# train_mask = data["train_mask"]
# val_mask = data["val_mask"]
# test_mask = data["test_mask"]

# 设置随机种子，以确保结果可复现
random_seed = 42

node_idx = [i for i in range(data['num_vertices'])]
# 将idx_test划分为训练（60%）、验证（20%）和测试（20%）集
idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# 确保划分后的集合没有重叠
assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0

# idx_train = np.where(train_mask)[0]
# idx_val = np.where(val_mask)[0]
# idx_test = np.where(test_mask)[0]

# v_deg= hg.D_v
# X = v_deg.to_dense()/torch.max(v_deg.to_dense())

X = data["features"]

# Normalize adj and features
features = data["features"].numpy()
features = X.numpy()
features_normalized = normalize_features(features)
# nor_hg = normalize_adj(hg)

# To PyTorch Tensor
# labels = torch.LongTensor(labels)
# labels = torch.max(labels, dim=1)[1]
labels = data["labels"]
features_normalized = torch.FloatTensor(features_normalized)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

cvae_model = torch.load("{}/model/{}_1208.pkl".format(exc_path, args.dataset))

# best_augmented_features, cvae_model = hgnn_cvae_pretrain_new_cora.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = hgnn_cvae_pretrain_new_cora.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list


if args.cuda:
    hg = hg.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

all_val = []
all_test = []

all_test_microf1, all_test_macrof1 = [], []


for i in trange(args.runs, desc='Run Train'):

    # Model and optimizer
    model = LAHGCN(concat=args.concat+1,
                  in_channels=features.shape[1],
                  hid_channels=args.hidden,
                  num_classes=labels.max().item() + 1,
                  use_bn=False,
                  drop_rate=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)

    # Train model
    best = 999999999
    best_model = None
    best_X_list = None
    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        output_list = []
        for k in range(args.samples):
            X_list = get_augmented_features(args.concat)
            output_list.append(torch.log_softmax(model(X_list+[features_normalized], hg), dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(output_list)

        loss_consis = consis_loss(output_list)
        loss_train = loss_train + loss_consis

        loss_train.backward()
        optimizer.step()

        model.eval()
        val_X_list = get_augmented_features(args.concat)
        output = model(val_X_list+[features_normalized],hg)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        

        print('Run:{:02d}'.format(i+1),
              'Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()))
                
        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            best_X_list = copy.deepcopy(val_X_list)

    #Validate and Test
    best_model.eval()
    output = best_model(best_X_list+[features_normalized], hg)

    outs, lbl = output[idx_test], labels[idx_test]

    output = torch.log_softmax(output, dim=1)

    # Calculate accuracy
    _, predicted = torch.max(outs, 1)
    # 将predicted结果转换为numpy数组
    predicted_array = predicted.cpu().numpy()

    # 保存到文件
    np.savetxt('res/lahgnn_predicted_cocitationcora.txt', predicted_array, fmt='%d')

    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # micro_f1_val = micro_f1(output[idx_val], labels[idx_val])
    # macro_f1_val = macro_f1(output[idx_val], labels[idx_val])
    micro_f1_test = micro_f1(output[idx_test], labels[idx_test])
    macro_f1_test = macro_f1(output[idx_test], labels[idx_test])

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())
    all_test_microf1.append(micro_f1_test.item())
    all_test_macrof1.append(macro_f1_test.item())

# print('val acc:', np.mean(all_val), 'val acc std:', np.std(all_val))
# print('\n')
print('test acc:', np.mean(all_test), 'test acc std', np.std(all_test))
print('\n')
print('test micro f1:', np.mean(all_test_microf1), 'test macro f1', np.mean(all_test_macrof1))



import numpy as np
import matplotlib.pyplot as plt

import numpy as np


A = hg.H @ hg.H.T
# print(A)

# 将稀疏张量转换为标准的邻接矩阵表示
adj_matrix = torch.sparse_coo_tensor(A.indices(), A.values(), A.size())

# 计算每个节点的度
degree_list = adj_matrix.to_dense().sum(dim=1)
degree_list = degree_list.cpu().numpy().tolist()


# 从文件中读取预测结果
predicted_array = np.loadtxt('res/lahgnn_predicted_cocitationcora.txt', dtype=int)
# 将numpy数组转换为列表
predicted_labels = predicted_array.tolist()
true_labels = labels[test_mask].cpu().numpy().tolist()


predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)
# 计算每个节点的ACC
trues = (predicted_labels == true_labels).astype(int)
errors = [1 if value == 0 else 0 for value in trues]

degree_counts = {}
filtered_data = [(degree, value1, value2) for degree, value1, value2 in zip(degree_list, trues, errors) if 0 <= degree <= 5]

# Populate the dictionary
for degree, value1, value2 in filtered_data:
    if degree not in degree_counts:
        degree_counts[degree] = {'ones': 0, 'zeros': 0, 'total': 0}
    degree_counts[degree]['ones'] += value1
    degree_counts[degree]['zeros'] += value2
    degree_counts[degree]['total'] += 1

# Calculate the probabilities
degree_probabilities_list1 = [degree_counts.get(degree, {'ones': 0})['ones'] / degree_counts.get(degree, {'total': 1})['total'] for degree in range(1,6)]
degree_probabilities_list2 = [degree_counts.get(degree, {'zeros': 0})['zeros'] / degree_counts.get(degree, {'total': 1})['total'] for degree in range(1,6)]

# Plot the results
width = 0.35
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(np.arange(1,6), degree_probabilities_list1, width, label='Ratio of True Predictions', color='#FFA2A5', edgecolor='#FF0000',alpha=0.7)
ax.bar(np.arange(1,6), degree_probabilities_list2, width, label='Ratio of False Predictions', color='#A7A7FF', edgecolor='#0000FF',alpha=0.7, bottom=degree_probabilities_list1)
ax.set_xlabel('Number of neighbor nodes (1-6)')
ax.set_ylabel('Probability')
ax.grid(True, linestyle='--', alpha=0.5)
# ax.set_title('Ratio distribution of True and False predictions')
ax.legend(fontsize=10)
plt.savefig('figs/lahgnn_deg_acc_cocitationcora_1225.pdf')