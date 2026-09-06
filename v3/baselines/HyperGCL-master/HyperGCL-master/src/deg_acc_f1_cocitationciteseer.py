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

data = CocitationCiteseer()
G = Hypergraph(data["num_vertices"], data["edge_list"])
print(G)


A = G.H @ G.H.T
# print(A)

# 将稀疏张量转换为标准的邻接矩阵表示
adj_matrix = torch.sparse_coo_tensor(A.indices(), A.values(), A.size())

# 计算每个节点的度
degree_list = adj_matrix.to_dense().sum(dim=1)
degree_list = degree_list.cpu().numpy().tolist()


# print(degree_list[:10], len(degree_list))

# train_mask = data["train_mask"]
# val_mask = data["val_mask"]
# test_mask = data["test_mask"]

# train_ratio = train_mask.sum() / len(train_mask)
# val_ratio = val_mask.sum() / len(val_mask)
# test_ratio = test_mask.sum() / len(test_mask)

# print("Train set ratio:", train_ratio)
# print("Validation set ratio:", val_ratio)
# print("Test set ratio:", test_ratio)


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

v_deg= G.D_v

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

net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=5e-4)

X, lbls = X.to(device), lbls.to(device)
G = G.to(device)
net = net.to(device)

# best_state = None
# best_epoch, best_val = 0, 0
# num_epochs = 200
# all_acc, all_microf1, all_macrof1 = [],[],[]
# for run in range(5):

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
#             print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

#             # Save the model if it has the best validation accuracy
#             if val_acc > best_val:
#                 print(f"update best: {val_acc:.5f}")
#                 best_val = val_acc
#                 best_state = deepcopy(net.state_dict())
#                 # torch.save(net.state_dict(), 'hgnn_cocitationcora_best_model.pth')

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

#         # 将predicted结果转换为numpy数组
#         predicted_array = predicted.cpu().numpy()

#         # 保存到文件
#         np.savetxt('res/predicted_cocitationciteseer.txt', predicted_array, fmt='%d')
    
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
# print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# print('\n')
# print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))


import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# 从文件中读取预测结果
predicted_array = np.loadtxt('res/predicted_cocitationciteseer.txt', dtype=int)
# 将numpy数组转换为列表
predicted_labels = predicted_array.tolist()
true_labels = lbls[test_mask].cpu().numpy().tolist()


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
ax.grid(True, linestyle='--', alpha=0.5)  # Add grid lines with dashed linestyle and alpha (transparency)
# ax.set_title('Ratio distribution of True and False predictions')
ax.legend(fontsize=12)
plt.savefig('figs/hgnn_deg_acc_cocitationciteseer.pdf')
