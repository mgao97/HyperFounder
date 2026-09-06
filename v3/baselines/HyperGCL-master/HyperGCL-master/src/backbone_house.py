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

data = HouseCommittees()
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

v_deg= G.D_v
X = v_deg.to_dense()/torch.max(v_deg.to_dense())
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


X, lbls = X.to(device), lbls.to(device)
G = G.to(device)


best_state = None
best_epoch, best_val = 0, 0

all_acc, all_microf1, all_macrof1 = [],[],[]
for run in range(5):
    net = HGNN(X.shape[1], 20, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
    net = net.to(device)

    print(f'net:{net}')

    train_losses = []  # 新增：用于存储每个epoch的train_loss
    val_losses = []  # 新增：用于存储每个epoch的val_loss
    for epoch in range(num_epochs):
        # train
        net.train()
        optimizer.zero_grad()
        outs = net(X,G)
        outs, lbl = outs[idx_train], lbls[idx_train]
        loss = F.cross_entropy(outs, lbl)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # validation
        net.eval()
        with torch.no_grad():
            outs = net(X,G)
            outs, lbl = outs[idx_val], lbls[idx_val]
            val_loss = F.cross_entropy(outs, lbl)
            val_losses.append(val_loss)  # 新增：记录val_loss

            _, predicted = torch.max(outs, 1)
            correct = (predicted == lbl).sum().item()
            total = lbl.size(0)
            val_acc = correct / total

            if epoch % 5 == 0:
                print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

            # Save the model if it has the best validation accuracy
            if val_acc > best_val:
                print(f"update best: {val_acc:.5f}")
                best_val = val_acc
                best_state = deepcopy(net.state_dict())
                # torch.save(net.state_dict(), 'model/hgnn_house_best_model.pth')
        scheduler.step()
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")

    # 绘制曲线图
    # plt.plot(range(num_epochs), train_losses, label='Train Loss')
    # plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # test
    print("test...")
    net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        outs = net(X, G)
        outs, lbl = outs[idx_test], lbls[idx_test]

        # Calculate accuracy
        _, predicted = torch.max(outs, 1)
        correct = (predicted == lbl).sum().item()
        total = lbl.size(0)
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc}')

        # Calculate micro F1
        micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
        print(f'Micro F1: {micro_f1}')

        # Calculate macro F1
        macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
        print(f'Macro F1: {macro_f1}')

    all_acc.append(test_acc)
    all_microf1.append(micro_f1)
    all_macrof1.append(macro_f1)

# avg of 5 times
print('Model HGNN Results:\n')
print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
print('\n')
print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))

print('='*100)

# min_value, min_index = torch.min(G.D_e.values(), dim=0)
# print("hyperedges度值最小值:", min_value)
# print("hyperedges度值最小值对应的下标（也就是仅包含孤立点的超边的index，需要remove）:", min_index)
#------
# hyperedges度值最小值: tensor(1.)
# hyperedges度值最小值对应的下标（也就是仅包含孤立点的超边的index，需要remove）: tensor(27)
#------
# min_value, min_index = torch.min(G.D_v.values(), dim=0)
# print("节点度值最小值:", min_value)
# print("节点度值最小值对应的下标（也就是孤立点的index，需要remove）:", min_index)
#------
# hyperedges度值最小值: tensor(1.)
# hyperedges度值最小值对应的下标（也就是仅包含孤立点的超边的index，需要remove）: tensor(27)
# DHG实现的HyperGCN不支持孤立点（即一条超边包含一个节点），因此额外补充了一个节点1291和超边中的孤立点一起构成超边27



he = G.e[0]
G.e[0][27] = (447, data['num_vertices'])
print(G.e[0][27])
new_G = Hypergraph(1291, he)
lbls = data["labels"]
# print(lbls.shape, lbls[447])
# print(torch.Size([1290]), tensor(2))

# # 设置随机种子，以确保结果可复现
random_seed = 42

node_idx = [i for i in range(new_G.num_v)]
# 将idx_test划分为训练（50%）、验证（25%）和测试（25%）集
idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# 确保划分后的集合没有重叠
assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0

train_mask = torch.zeros(new_G.num_v, dtype=torch.bool)
val_mask = torch.zeros(new_G.num_v, dtype=torch.bool)
test_mask = torch.zeros(new_G.num_v, dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

# idx_train = np.where(train_mask)[0]
# idx_val = np.where(val_mask)[0]
# idx_test = np.where(test_mask)[0]

new_v_deg= new_G.D_v
new_X = new_v_deg.to_dense()/torch.max(new_v_deg.to_dense())


new_lbls = torch.cat((lbls, torch.LongTensor([2])), dim=0)
print('new_X dim:', new_X.shape)
print('new labels:', new_lbls.shape)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])


num_epochs = 200


new_X, new_lbls = new_X.to(device), new_lbls.to(device)
new_G = new_G.to(device)


best_state = None
best_epoch, best_val = 0, 0

all_acc, all_microf1, all_macrof1 = [],[],[]
for run in range(5):
    net = HyperGCN(new_X.shape[1], 20, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.01)
    net = net.to(device)

    print(f'net:\n')
    print(net)

    train_losses = []  # 新增：用于存储每个epoch的train_loss
    val_losses = []  # 新增：用于存储每个epoch的val_loss
    for epoch in range(num_epochs):
        # train
        net.train()
        optimizer.zero_grad()
        outs = net(new_X,new_G)
        outs, lbl = outs[idx_train], new_lbls[idx_train]
        loss = F.cross_entropy(outs, lbl)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # validation
        net.eval()
        with torch.no_grad():
            outs = net(new_X,new_G)
            outs, lbl = outs[idx_val], new_lbls[idx_val]
            val_loss = F.cross_entropy(outs, lbl)
            val_losses.append(val_loss)  # 新增：记录val_loss

            _, predicted = torch.max(outs, 1)
            correct = (predicted == lbl).sum().item()
            total = lbl.size(0)
            val_acc = correct / total

            if epoch % 5 == 0:
                print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

            # Save the model if it has the best validation accuracy
            if val_acc > best_val:
                print(f"update best: {val_acc:.5f}")
                best_val = val_acc
                best_state = deepcopy(net.state_dict())
                torch.save(net.state_dict(), 'model/hypergcn_house_best_model_1216.pth')
        scheduler.step()
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")

    # 绘制曲线图
    # plt.plot(range(num_epochs), train_losses, label='Train Loss')
    # plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # test
    print("test...")
    net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        outs = net(new_X, new_G)
        outs, lbl = outs[idx_test], new_lbls[idx_test]

        # Calculate accuracy
        _, predicted = torch.max(outs, 1)
        correct = (predicted == lbl).sum().item()
        total = lbl.size(0)
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc}')

        # Calculate micro F1
        micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
        print(f'Micro F1: {micro_f1}')

        # Calculate macro F1
        macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
        print(f'Macro F1: {macro_f1}')

    all_acc.append(test_acc)
    all_microf1.append(micro_f1)
    all_macrof1.append(macro_f1)

# avg of 5 times
print('Model HyperGCN Results:\n')
print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
print('\n')
print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))


print('='*150)

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



# set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])



v_deg= G.D_v
X = v_deg.to_dense()/torch.max(v_deg.to_dense())

X, lbls = X.to(device), lbls.to(device)
G = G.to(device)


best_state = None
best_epoch, best_val = 0, 0
num_epochs = 200
all_acc, all_microf1, all_macrof1 = [],[],[]
for run in range(5):

    model_unigin = UniGIN(X.shape[1], 20, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(model_unigin.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
    model_unigin = model_unigin.to(device)
    print(f'model: {model_unigin}')

    train_losses = []  # 新增：用于存储每个epoch的train_loss
    val_losses = []  # 新增：用于存储每个epoch的val_loss
    for epoch in range(num_epochs):
        # train
        model_unigin.train()
        optimizer.zero_grad()
        outs = model_unigin(X,G)
        outs, lbl = outs[idx_train], lbls[idx_train]
        loss = F.cross_entropy(outs, lbl)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # validation
        model_unigin.eval()
        with torch.no_grad():
            outs = model_unigin(X,G)
            outs, lbl = outs[idx_val], lbls[idx_val]
            val_loss = F.cross_entropy(outs, lbl)
            val_losses.append(val_loss)  # 新增：记录val_loss

            _, predicted = torch.max(outs, 1)
            correct = (predicted == lbl).sum().item()
            total = lbl.size(0)
            val_acc = correct / total
            print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

            # Save the model if it has the best validation accuracy
            if val_acc > best_val:
                print(f"update best: {val_acc:.5f}")
                best_val = val_acc
                best_state = deepcopy(model_unigin.state_dict())
                torch.save(model_unigin.state_dict(), 'unigin_house_best_model_1216.pth')

    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")

    # 绘制曲线图
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # test
    print("test...")
    model_unigin.load_state_dict(best_state)

    model_unigin.eval()
    with torch.no_grad():
        outs = model_unigin(X, G)
        outs, lbl = outs[idx_test], lbls[idx_test]

        # Calculate accuracy
        _, predicted = torch.max(outs, 1)
        correct = (predicted == lbl).sum().item()
        total = lbl.size(0)
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc}')

        # Calculate micro F1
        micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
        print(f'Micro F1: {micro_f1}')

        # Calculate macro F1
        macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
        print(f'Macro F1: {macro_f1}')

    all_acc.append(test_acc)
    all_microf1.append(micro_f1)
    all_macrof1.append(macro_f1)

# avg of 5 times
print('Model UniGIN Results:\n')
print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
print('\n')
print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))


print('='*200)


best_state = None
best_epoch, best_val = 0, 0
num_epochs = 200
all_acc, all_microf1, all_macrof1 = [],[],[]
for run in range(5):

    model_unisage = UniSAGE(X.shape[1], 20, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(model_unisage.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
    model_unisage = model_unisage.to(device)
    print(f'model: {model_unisage}')

    train_losses = []  # 新增：用于存储每个epoch的train_loss
    val_losses = []  # 新增：用于存储每个epoch的val_loss
    for epoch in range(num_epochs):
        # train
        model_unisage.train()
        optimizer.zero_grad()
        outs = model_unisage(X,G)
        outs, lbl = outs[idx_train], lbls[idx_train]
        loss = F.cross_entropy(outs, lbl)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # validation
        model_unisage.eval()
        with torch.no_grad():
            outs = model_unisage(X,G)
            outs, lbl = outs[idx_val], lbls[idx_val]
            val_loss = F.cross_entropy(outs, lbl)
            val_losses.append(val_loss)  # 新增：记录val_loss

            _, predicted = torch.max(outs, 1)
            correct = (predicted == lbl).sum().item()
            total = lbl.size(0)
            val_acc = correct / total
            print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

            # Save the model if it has the best validation accuracy
            if val_acc > best_val:
                print(f"update best: {val_acc:.5f}")
                best_val = val_acc
                best_state = deepcopy(model_unisage.state_dict())
                torch.save(model_unisage.state_dict(), 'unisage_house_best_model_1216.pth')

    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")

    # 绘制曲线图
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # test
    print("test...")
    model_unisage.load_state_dict(best_state)

    model_unisage.eval()
    with torch.no_grad():
        outs = model_unisage(X, G)
        outs, lbl = outs[idx_test], lbls[idx_test]

        # Calculate accuracy
        _, predicted = torch.max(outs, 1)
        correct = (predicted == lbl).sum().item()
        total = lbl.size(0)
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc}')

        # Calculate micro F1
        micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
        print(f'Micro F1: {micro_f1}')

        # Calculate macro F1
        macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
        print(f'Macro F1: {macro_f1}')

    all_acc.append(test_acc)
    all_microf1.append(micro_f1)
    all_macrof1.append(macro_f1)

# avg of 5 times
print('Model UniSAGE Results:\n')
print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
print('\n')
print('test microf1:', np.mean(all_microf1), 'test macrof1:', np.mean(all_macrof1))