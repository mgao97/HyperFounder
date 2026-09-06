from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import sys
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import hgnn_cvae_pretrain_new_cora
import matplotlib.pyplot as plt
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

import time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, f1_score
from dhg import Hypergraph
from dhg.data import *
from dhg.models import *
from dhg.random import set_seed
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
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


data = CocitationCora()
G = Hypergraph(data["num_vertices"], data["edge_list"])
print(G)
# train_mask = data["train_mask"]
# val_mask = data["val_mask"]
# test_mask = data["test_mask"]

# # 设置随机种子，以确保结果可复现
random_seed = 42

node_idx = [i for i in range(data['num_vertices'])]
# 将idx_test划分为训练（50%）、验证（25%）和测试（25%）集
idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# 确保划分后的集合没有重叠
assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0

train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

# idx_train = np.where(train_mask)[0]
# idx_val = np.where(val_mask)[0]
# idx_test = np.where(test_mask)[0]

# v_deg= G.D_v
# X = v_deg.to_dense()/torch.max(v_deg.to_dense())
X = data["features"]
lbls = data["labels"]
print('X dim:', X.shape)
print('labels:', len(torch.unique(lbls)))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])


num_epochs = 200


# X, lbls = X.to(device), lbls.to(device)
# G = G.to(device)


# best_state = None
# best_epoch, best_val = 0, 0

# all_acc, all_microf1, all_macrof1 = [],[],[]
# for run in range(5):
#     net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
#     optimizer = optim.Adam(net.parameters(), lr=0.5, weight_decay=5e-4)
#     scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
#     net = net.to(device)

#     print(f'net:{net}')

#     train_losses = []  # 新增：用于存储每个epoch的train_loss
#     val_losses = []  # 新增：用于存储每个epoch的val_loss
#     for epoch in range(num_epochs):
#         # train
#         net.train()
#         optimizer.zero_grad()
#         outs = net(X,G)
#         outs, lbl = outs[idx_train], lbls[idx_train]
#         loss = F.cross_entropy(outs, lbl)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())

#         # validation
#         net.eval()
#         with torch.no_grad():
#             outs = net(X,G)
#             outs, lbl = outs[idx_val], lbls[idx_val]
#             val_loss = F.cross_entropy(outs, lbl)
#             val_losses.append(val_loss)  # 新增：记录val_loss

#             _, predicted = torch.max(outs, 1)
#             correct = (predicted == lbl).sum().item()
#             total = lbl.size(0)
#             val_acc = correct / total

#             if epoch % 5 == 0:
#                 print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

#             # Save the model if it has the best validation accuracy
#             if val_acc > best_val:
#                 print(f"update best: {val_acc:.5f}")
#                 best_val = val_acc
#                 best_state = deepcopy(net.state_dict())
#                 torch.save(net.state_dict(), 'model/hgnn_cocitationcora_best_model.pth')
#         scheduler.step()
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")

#     # # 绘制曲线图
#     # plt.plot(range(num_epochs), train_losses, label='Train Loss')
#     # plt.plot(range(num_epochs), val_losses, label='Validation Loss')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Loss')
#     # plt.legend()
#     # plt.show()

#     # test
#     print("test...")
#     net.load_state_dict(best_state)

#     net.eval()
#     with torch.no_grad():
#         outs = net(X, G)
#         outs, lbl = outs[idx_test], lbls[idx_test]
        

#         # Calculate accuracy
#         _, predicted = torch.max(outs, 1)
#         correct = (predicted == lbl).sum().item()
#         total = lbl.size(0)
#         test_acc = correct / total
#         print(f'Test Accuracy: {test_acc}')

#         # Calculate micro F1
#         micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
#         print(f'Micro F1: {micro_f1}')

#         # Calculate macro F1
#         macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
#         print(f'Macro F1: {macro_f1}')

#     all_acc.append(test_acc)
#     all_microf1.append(micro_f1)
#     all_macrof1.append(macro_f1)

# # avg of 5 times
# print('Model HGNN Results:\n')
# print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# print('\n')
# print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))


net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
net.load_state_dict(torch.load('model/hgnn_cocitationcora_best_model.pth'))
X, lbls = X.to(device), lbls.to(device)
G = G.to(device)
net = net.to(device)

hgnn_emb = net(X,G)

features = data["features"].numpy()
features = X.numpy()
features_normalized = normalize_features(features)
# nor_hg = normalize_adj(hg)

# To PyTorch Tensor
# labels = torch.LongTensor(labels)
# labels = torch.max(labels, dim=1)[1]
labels = data["labels"]
features_normalized = torch.FloatTensor(features_normalized)



model = LAHGCN(concat=args.concat+1,
                  in_channels=features.shape[1],
                  hid_channels=args.hidden,
                  num_classes=lbls.max().item() + 1,
                  use_bn=False,
                  drop_rate=args.dropout)
model.load_state_dict(torch.load('model/lahgnn_cocitationcora_best_model.pth'))
X, lbls = X.to(device), lbls.to(device)
hg = G.to(device)
net = model.to(device)



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

X_list = get_augmented_features(args.concat)
lahgnn_emb = model(X_list+[features_normalized],hg)

import torch.nn.functional as F

def column_cosine_similarity(tensor1, tensor2):
    # 按列归一化
    norm_tensor1 = F.normalize(tensor1, dim=0)
    norm_tensor2 = F.normalize(tensor2, dim=0)
    
    # 计算余弦相似性
    similarity = F.cosine_similarity(norm_tensor1, norm_tensor2, dim=0)
    
    return similarity

# def column_euclidean_distance(tensor1, tensor2):
#     # 计算欧几里得距离
#     distance = torch.norm(tensor1 - tensor2, dim=0)
    
#     return distance


cos_sim = column_cosine_similarity(hgnn_emb, lahgnn_emb)
print('cos_sim:', cos_sim,'avg of cos_sim:', torch.mean(cos_sim).item())

#MSE 
# 计算差值
diff = hgnn_emb-lahgnn_emb

# 计算平方
squared_diff = diff**2

# 计算均值
mse = torch.mean(squared_diff)

# 输出 MSE
print(mse.item())
# euc_sim = column_euclidean_distance(hgnn_emb, lahgnn_emb)
# print('cos_sim:', cos_sim, 'euc_sim:',euc_sim)



# print(f'hgnn_emb: {hgnn_emb}', f'hgnn_emb size: {hgnn_emb.shape}')

# tsne = TSNE(n_components=2, verbose=1, random_state=0)
# z = tsne.fit_transform(hgnn_emb.detach().numpy())
# z_data = np.vstack((z.T, lbls)).T
# df_tsne = pd.DataFrame(z_data, columns=['Dimension 1', 'Dimension 2', 'Class'])
# df_tsne['Class'] = df_tsne['Class'].astype(int)
# plt.figure(figsize=(8, 8))
# sns.set(font_scale=1.5)
# plt.legend(loc='upper right')
# sns.scatterplot(data=df_tsne, hue='Class', x='Dimension 1', y='Dimension 2', palette=['green','orange','brown','red', 'blue','black','purple'])
# plt.savefig("emb_figs/hgnn_cocitationcora.pdf", bbox_inches="tight") # save embeddings if needed
# plt.savefig("emb_figs/hgnn_cocitationcora.png", bbox_inches="tight")
# plt.show()

# print('='*100)

# # min_value, min_index = torch.min(G.D_e.values(), dim=0)
# # print("hyperedges度值最小值:", min_value)
# # print("hyperedges度值最小值对应的下标（也就是仅包含孤立点的超边的index，需要remove）:", min_index)
# #------
# # hyperedges度值最小值: tensor(1.)
# # hyperedges度值最小值对应的下标（也就是仅包含孤立点的超边的index，需要remove）: tensor(27)
# #------
# # min_value, min_index = torch.min(G.D_v.values(), dim=0)
# # print("节点度值最小值:", min_value)
# # print("节点度值最小值对应的下标（也就是孤立点的index，需要remove）:", min_index)
# #------
# # hyperedges度值最小值: tensor(1.)
# # hyperedges度值最小值对应的下标（也就是仅包含孤立点的超边的index，需要remove）: tensor(27)
# # DHG实现的HyperGCN不支持孤立点（即一条超边包含一个节点），因此额外补充了一个节点1291和超边中的孤立点一起构成超边27



# # he = G.e[0]
# # G.e[0][27] = (447, data['num_vertices'])
# # print(G.e[0][27])
# # new_G = Hypergraph(1291, he)
# # lbls = data["labels"]
# # # print(lbls.shape, lbls[447])
# # # print(torch.Size([1290]), tensor(2))

# # # # 设置随机种子，以确保结果可复现
# # random_seed = 42

# # node_idx = [i for i in range(new_G.num_v)]
# # # 将idx_test划分为训练（50%）、验证（25%）和测试（25%）集
# # idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
# # idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# # # 确保划分后的集合没有重叠
# # assert len(set(idx_train) & set(idx_val)) == 0
# # assert len(set(idx_train) & set(idx_test)) == 0
# # assert len(set(idx_val) & set(idx_test)) == 0

# # train_mask = torch.zeros(new_G.num_v, dtype=torch.bool)
# # val_mask = torch.zeros(new_G.num_v, dtype=torch.bool)
# # test_mask = torch.zeros(new_G.num_v, dtype=torch.bool)
# # train_mask[idx_train] = True
# # val_mask[idx_val] = True
# # test_mask[idx_test] = True

# # idx_train = np.where(train_mask)[0]
# # idx_val = np.where(val_mask)[0]
# # idx_test = np.where(test_mask)[0]

# # new_v_deg= new_G.D_v
# # new_X = new_v_deg.to_dense()/torch.max(new_v_deg.to_dense())


# # new_lbls = torch.cat((lbls, torch.LongTensor([2])), dim=0)
# # print('new_X dim:', new_X.shape)
# # print('new labels:', new_lbls.shape)


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, f1_score

# set_seed(42)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])


# num_epochs = 200


# # new_X, new_lbls = new_X.to(device), new_lbls.to(device)
# # new_G = new_G.to(device)


# best_state = None
# best_epoch, best_val = 0, 0

# all_acc, all_microf1, all_macrof1 = [],[],[]
# for run in range(5):
#     net = HyperGCN(X.shape[1], 256, data["num_classes"])
#     optimizer = optim.Adam(net.parameters(), lr=0.05, weight_decay=5e-4)
#     scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.01)
#     net = net.to(device)

#     print(f'net:\n')
#     print(net)

#     train_losses = []  # 新增：用于存储每个epoch的train_loss
#     val_losses = []  # 新增：用于存储每个epoch的val_loss
#     for epoch in range(num_epochs):
#         # train
#         net.train()
#         optimizer.zero_grad()
#         outs = net(X,G)
#         outs, lbl = outs[idx_train], lbls[idx_train]
#         loss = F.cross_entropy(outs, lbl)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())

#         # validation
#         net.eval()
#         with torch.no_grad():
#             outs = net(X,G)
#             outs, lbl = outs[idx_val], lbls[idx_val]
#             val_loss = F.cross_entropy(outs, lbl)
#             val_losses.append(val_loss)  # 新增：记录val_loss

#             _, predicted = torch.max(outs, 1)
#             correct = (predicted == lbl).sum().item()
#             total = lbl.size(0)
#             val_acc = correct / total

#             if epoch % 5 == 0:
#                 print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

#             # Save the model if it has the best validation accuracy
#             if val_acc > best_val:
#                 print(f"update best: {val_acc:.5f}")
#                 best_val = val_acc
#                 best_state = deepcopy(net.state_dict())
#                 torch.save(net.state_dict(), 'model/hypergcn_cocitationcora_best_model.pth')
#         scheduler.step()
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")

#     # 绘制曲线图
#     # plt.plot(range(num_epochs), train_losses, label='Train Loss')
#     # plt.plot(range(num_epochs), val_losses, label='Validation Loss')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Loss')
#     # plt.legend()
#     # plt.show()

#     # test
#     print("test...")
#     net.load_state_dict(best_state)

#     net.eval()
#     with torch.no_grad():
#         outs = net(X, G)
#         outs, lbl = outs[idx_test], lbls[idx_test]

#         # Calculate accuracy
#         _, predicted = torch.max(outs, 1)
#         correct = (predicted == lbl).sum().item()
#         total = lbl.size(0)
#         test_acc = correct / total
#         print(f'Test Accuracy: {test_acc}')

#         # Calculate micro F1
#         micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
#         print(f'Micro F1: {micro_f1}')

#         # Calculate macro F1
#         macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
#         print(f'Macro F1: {macro_f1}')

#     all_acc.append(test_acc)
#     all_microf1.append(micro_f1)
#     all_macrof1.append(macro_f1)

# # avg of 5 times
# print('Model HyperGCN Results:\n')
# print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# print('\n')
# print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))


# # print('='*150)

# # # # 设置随机种子，以确保结果可复现
# # random_seed = 42

# # node_idx = [i for i in range(data['num_vertices'])]
# # # 将idx_test划分为训练（50%）、验证（25%）和测试（25%）集
# # idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
# # idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# # # 确保划分后的集合没有重叠
# # assert len(set(idx_train) & set(idx_val)) == 0
# # assert len(set(idx_train) & set(idx_test)) == 0
# # assert len(set(idx_val) & set(idx_test)) == 0

# # train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# # val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# # test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# # train_mask[idx_train] = True
# # val_mask[idx_val] = True
# # test_mask[idx_test] = True



# # # set_seed(42)
# # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # # evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])



# # # v_deg= G.D_v
# # # X = v_deg.to_dense()/torch.max(v_deg.to_dense())
# # X = data['features']

# # X, lbls = X.to(device), lbls.to(device)
# # G = G.to(device)


# # best_state = None
# # best_epoch, best_val = 0, 0
# # num_epochs = 200
# # all_acc, all_microf1, all_macrof1 = [],[],[]
# # for run in range(5):

# #     model_unigin = UniGIN(X.shape[1], 256, data["num_classes"], use_bn=True)
# #     optimizer = optim.Adam(model_unigin.parameters(), lr=0.5, weight_decay=5e-4)
# #     scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
# #     model_unigin = model_unigin.to(device)
# #     print(f'model: {model_unigin}')

# #     train_losses = []  # 新增：用于存储每个epoch的train_loss
# #     val_losses = []  # 新增：用于存储每个epoch的val_loss
# #     for epoch in range(num_epochs):
# #         # train
# #         model_unigin.train()
# #         optimizer.zero_grad()
# #         outs = model_unigin(X,G)
# #         outs, lbl = outs[idx_train], lbls[idx_train]
# #         loss = F.cross_entropy(outs, lbl)
# #         loss.backward()
# #         optimizer.step()
# #         train_losses.append(loss.item())

# #         # validation
# #         model_unigin.eval()
# #         with torch.no_grad():
# #             outs = model_unigin(X,G)
# #             outs, lbl = outs[idx_val], lbls[idx_val]
# #             val_loss = F.cross_entropy(outs, lbl)
# #             val_losses.append(val_loss)  # 新增：记录val_loss

# #             _, predicted = torch.max(outs, 1)
# #             correct = (predicted == lbl).sum().item()
# #             total = lbl.size(0)
# #             val_acc = correct / total
# #             print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

# #             # Save the model if it has the best validation accuracy
# #             if val_acc > best_val:
# #                 print(f"update best: {val_acc:.5f}")
# #                 best_val = val_acc
# #                 best_state = deepcopy(model_unigin.state_dict())
# #                 torch.save(model_unigin.state_dict(), 'unigin_cocitationcora_best_model.pth')

# #     print("\ntrain finished!")
# #     print(f"best val: {best_val:.5f}")

# #     # 绘制曲线图
# #     # plt.plot(range(num_epochs), train_losses, label='Train Loss')
# #     # plt.plot(range(num_epochs), val_losses, label='Validation Loss')
# #     # plt.xlabel('Epoch')
# #     # plt.ylabel('Loss')
# #     # plt.legend()
# #     # plt.show()

# #     # test
# #     print("test...")
# #     model_unigin.load_state_dict(best_state)

# #     model_unigin.eval()
# #     with torch.no_grad():
# #         outs = model_unigin(X, G)
# #         outs, lbl = outs[idx_test], lbls[idx_test]

# #         # Calculate accuracy
# #         _, predicted = torch.max(outs, 1)
# #         correct = (predicted == lbl).sum().item()
# #         total = lbl.size(0)
# #         test_acc = correct / total
# #         print(f'Test Accuracy: {test_acc}')

# #         # Calculate micro F1
# #         micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
# #         print(f'Micro F1: {micro_f1}')

# #         # Calculate macro F1
# #         macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
# #         print(f'Macro F1: {macro_f1}')

# #     all_acc.append(test_acc)
# #     all_microf1.append(micro_f1)
# #     all_macrof1.append(macro_f1)

# # # avg of 5 times
# # print('Model UniGIN Results:\n')
# # print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# # print('\n')
# # print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))


# # print('='*200)


# # best_state = None
# # best_epoch, best_val = 0, 0
# # num_epochs = 200
# # all_acc, all_microf1, all_macrof1 = [],[],[]
# # for run in range(5):

# #     model_unisage = UniSAGE(X.shape[1], 256, data["num_classes"], use_bn=True)
# #     optimizer = optim.Adam(model_unisage.parameters(), lr=0.5, weight_decay=5e-4)
# #     scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
# #     model_unisage = model_unisage.to(device)
# #     print(f'model: {model_unisage}')

# #     train_losses = []  # 新增：用于存储每个epoch的train_loss
# #     val_losses = []  # 新增：用于存储每个epoch的val_loss
# #     for epoch in range(num_epochs):
# #         # train
# #         model_unisage.train()
# #         optimizer.zero_grad()
# #         outs = model_unisage(X,G)
# #         outs, lbl = outs[idx_train], lbls[idx_train]
# #         loss = F.cross_entropy(outs, lbl)
# #         loss.backward()
# #         optimizer.step()
# #         train_losses.append(loss.item())

# #         # validation
# #         model_unisage.eval()
# #         with torch.no_grad():
# #             outs = model_unisage(X,G)
# #             outs, lbl = outs[idx_val], lbls[idx_val]
# #             val_loss = F.cross_entropy(outs, lbl)
# #             val_losses.append(val_loss)  # 新增：记录val_loss

# #             _, predicted = torch.max(outs, 1)
# #             correct = (predicted == lbl).sum().item()
# #             total = lbl.size(0)
# #             val_acc = correct / total
# #             print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

# #             # Save the model if it has the best validation accuracy
# #             if val_acc > best_val:
# #                 print(f"update best: {val_acc:.5f}")
# #                 best_val = val_acc
# #                 best_state = deepcopy(model_unisage.state_dict())
# #                 torch.save(model_unisage.state_dict(), 'unisage_cocitationcora_best_model.pth')

# #     print("\ntrain finished!")
# #     print(f"best val: {best_val:.5f}")

# #     # 绘制曲线图
# #     # plt.plot(range(num_epochs), train_losses, label='Train Loss')
# #     # plt.plot(range(num_epochs), val_losses, label='Validation Loss')
# #     # plt.xlabel('Epoch')
# #     # plt.ylabel('Loss')
# #     # plt.legend()
# #     # plt.show()

# #     # test
# #     print("test...")
# #     model_unisage.load_state_dict(best_state)

# #     model_unisage.eval()
# #     with torch.no_grad():
# #         outs = model_unisage(X, G)
# #         outs, lbl = outs[idx_test], lbls[idx_test]

# #         # Calculate accuracy
# #         _, predicted = torch.max(outs, 1)
# #         correct = (predicted == lbl).sum().item()
# #         total = lbl.size(0)
# #         test_acc = correct / total
# #         print(f'Test Accuracy: {test_acc}')

# #         # Calculate micro F1
# #         micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
# #         print(f'Micro F1: {micro_f1}')

# #         # Calculate macro F1
# #         macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
# #         print(f'Macro F1: {macro_f1}')

# #     all_acc.append(test_acc)
# #     all_microf1.append(micro_f1)
# #     all_macrof1.append(macro_f1)

# # # avg of 5 times
# # print('Model UniSAGE Results:\n')
# # print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# # print('\n')
# # print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))