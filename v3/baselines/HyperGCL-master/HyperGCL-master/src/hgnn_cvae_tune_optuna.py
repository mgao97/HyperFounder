# import argparse
# import numpy as np
# import scipy.sparse as sp
# import torch
# import sys
# import random
# import torch.nn.functional as F
# import torch.optim as optim
# import hgnn_cvae

# from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
# # from gcn.models import GCN
# from hgnn_cvae import HGNN
# from tqdm import trange
# import dhg
# from dhg.data import CocitationCora, Cooking200
# from dhg import Hypergraph
# from dhg.nn import HGNNConv

# import optuna

# exc_path = sys.path[0]

# parser = argparse.ArgumentParser()
# parser.add_argument("--pretrain_epochs", type=int, default=20)
# parser.add_argument("--batch_size", type=int, default=64)
# parser.add_argument("--latent_size", type=int, default=10)
# parser.add_argument("--pretrain_lr", type=float, default=0.01)
# parser.add_argument("--conditional", action='store_true', default=True)
# parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
# parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
# parser.add_argument('--warmup', type=int, default=400, help='Warmup')
# parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')

# parser.add_argument('--dataset', default='cooking200',
#                     help='Dataset string.')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=1000,
#                     help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=8,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')

# args = parser.parse_args()

# def objective(trial):
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     args.cuda = torch.cuda.is_available()

#     # Load data
#     # adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     # Cocitation Cora/Cooking200 Data
#     data = Cooking200()
#     hg = Hypergraph(data["num_vertices"], data["edge_list"])
#     print(hg)
#     train_mask = data["train_mask"]
#     val_mask = data["val_mask"]
#     test_mask = data["test_mask"]

#     idx_train = np.where(train_mask)[0]
#     idx_val = np.where(val_mask)[0]
#     idx_test = np.where(test_mask)[0]
#     # for cooking200
#     v_deg= hg.D_v
#     X = v_deg.to_dense()/torch.max(v_deg.to_dense())
    
#     # X = data["features"]

#     # Normalize adj and features
#     features = X.numpy()
#     features_normalized = normalize_features(features)
#     labels = data["labels"]
#     features_normalized = torch.FloatTensor(features_normalized)
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)

#     # Pretrain
#     best_augmented_features = None

#     best_augmented_features, _ = hgnn_cvae.get_augmented_features(hg, X, labels, idx_train, features_normalized, device)
#     best_augmented_features = hgnn_cvae.feature_tensor_normalize(best_augmented_features).detach()

#     # Define hyperparameters to search
#     lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
#     weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
#     hidden_size = trial.suggest_int("hidden_size", 4, 32)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.9)

#     all_maxVal1Acc_Val2Acc = []
#     all_maxVal1Acc = []
#     for i in trange(args.runs, desc='Run Train'):
#         # Model and optimizer
#         idx_val1 = np.random.choice(list(idx_val), size=int(len(idx_val) * 0.5), replace=False)
#         idx_val2 = list(set(idx_val) - set(idx_val1))
#         idx_val1 = torch.LongTensor(idx_val1)
#         idx_val2 = torch.LongTensor(idx_val2)

#         model = HGNN(in_channels=features.shape[1],
#                     hid_channels=hidden_size,
#                     num_classes=labels.max().item() + 1,
#                     use_bn=True,
#                     drop_rate=dropout_rate) 
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#         if args.cuda:
#             model.to(device)
#             # adj_normalized = adj_normalized.to(device)
#             hg = hg.to(device)
#             features_normalized = features_normalized.to(device)
#             labels = labels.to(device)
#             idx_train = idx_train.to(device)
#             idx_val1 = idx_val1.to(device)
#             idx_val2 = idx_val2.to(device)
#             best_augmented_features = best_augmented_features.to(device)

#         # Train model
#         maxVal1Acc = 0
#         maxVal1Acc_Val2Acc = 0
#         for epoch in range(args.epochs):
#             model.train()
#             optimizer.zero_grad()
#             output = model(best_augmented_features, hg)
#             output = torch.log_softmax(output, dim=1)
#             loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#             acc_train = accuracy(output[idx_train], labels[idx_train])
#             loss_train.backward()
#             optimizer.step()

#             model.eval()
#             output = model(best_augmented_features, hg)
#             output = torch.log_softmax(output, dim=1)
#             loss_val1 = F.nll_loss(output[idx_val1], labels[idx_val1])
#             acc_val1 = accuracy(output[idx_val1], labels[idx_val1])

#             loss_val2 = F.nll_loss(output[idx_val2], labels[idx_val2])
#             acc_val2 = accuracy(output[idx_val2], labels[idx_val2])

#             if acc_val1 > maxVal1Acc:
#                 maxVal1Acc = acc_val1
#                 maxVal1Acc_Val2Acc = acc_val2


#         all_maxVal1Acc_Val2Acc.append(maxVal1Acc_Val2Acc.item())
#         all_maxVal1Acc.append(maxVal1Acc.item())

#     return np.mean(all_maxVal1Acc_Val2Acc)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# print("Number of finished trials: ", len(study.trials))
# print("Best trial:")
# trial = study.best_trial

# print("  Value: ", trial.value)
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import hgnn_cvae_pretrain

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
# from gcn.models import GCN
from hgnn_cvae_pretrain import HGNN
from tqdm import trange
import dhg
from dhg.data import CocitationCora, Cooking200
from dhg import Hypergraph
from dhg.nn import HGNNConv

import optuna

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=50, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=100, help='Warmup')
parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

parser.add_argument('--dataset', default='cooking200',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

def objective(trial):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.cuda = torch.cuda.is_available()

    # Load data
    # adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Cocitation Cora/Cooking200 Data
    data = Cooking200()
    hg = Hypergraph(data["num_vertices"], data["edge_list"])
    print(hg)
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]
    # for cooking200
    v_deg= hg.D_v
    X = v_deg.to_dense()/torch.max(v_deg.to_dense())
    
    # X = data["features"]

    # Normalize adj and features
    features = X.numpy()
    features_normalized = normalize_features(features)
    labels = data["labels"]
    features_normalized = torch.FloatTensor(features_normalized)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Pretrain
    # best_augmented_features = None

    best_augmented_features, _ = hgnn_cvae_pretrain.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)
    # best_augmented_features = torch.load('best_features.pt')
    print(best_augmented_features, best_augmented_features.shape)
    best_augmented_features = hgnn_cvae_pretrain.feature_tensor_normalize(best_augmented_features).detach()

    # Define hyperparameters to search
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_size = trial.suggest_int("hidden_size", 4, 32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.9)

    all_maxVal1Acc_Val2Acc = []
    all_maxVal1Acc = []
    for i in trange(args.runs, desc='Run Train'):
        # Model and optimizer
        idx_val1 = np.random.choice(list(idx_val), size=int(len(idx_val) * 0.5), replace=False)
        idx_val2 = list(set(idx_val) - set(idx_val1))
        idx_val1 = torch.LongTensor(idx_val1)
        idx_val2 = torch.LongTensor(idx_val2)

        model = HGNN(in_channels=features.shape[1],
                    hid_channels=hidden_size,
                    num_classes=labels.max().item() + 1,
                    use_bn=True,
                    drop_rate=dropout_rate) 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if args.cuda:
            model.to(device)
            # adj_normalized = adj_normalized.to(device)
            hg = hg.to(device)
            features_normalized = features_normalized.to(device)
            labels = labels.to(device)
            idx_train = idx_train.to(device)
            idx_val1 = idx_val1.to(device)
            idx_val2 = idx_val2.to(device)
            best_augmented_features = best_augmented_features.to(device)

        # Train model
        maxVal1Acc = 0
        maxVal1Acc_Val2Acc = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(best_augmented_features, hg)
            output = torch.log_softmax(output, dim=1)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            model.eval()
            output = model(best_augmented_features, hg)
            output = torch.log_softmax(output, dim=1)
            loss_val1 = F.nll_loss(output[idx_val1], labels[idx_val1])
            acc_val1 = accuracy(output[idx_val1], labels[idx_val1])

            loss_val2 = F.nll_loss(output[idx_val2], labels[idx_val2])
            acc_val2 = accuracy(output[idx_val2], labels[idx_val2])

            if acc_val1 > maxVal1Acc:
                maxVal1Acc = acc_val1
                maxVal1Acc_Val2Acc = acc_val2


        all_maxVal1Acc_Val2Acc.append(maxVal1Acc_Val2Acc.item())
        all_maxVal1Acc.append(maxVal1Acc.item())

    return np.mean(all_maxVal1Acc_Val2Acc)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


