import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import baseline_cvae_pretrain_cocitationcora
import easygraph as eg
from dhg import Hypergraph
from dhg.data import *

from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from utils import accuracy, normalize_features, micro_f1, macro_f1, sparse_mx_to_torch_sparse_tensor, normalize_adj
from hgcn import *
from gcn import *
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from gcn.models import GCN
from tqdm import trange
import networkx as nx

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')

parser.add_argument('--dataset', default='cora',
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
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# load data
data = CocitationCora()
print(data)

hg = eg.Hypergraph(data["num_vertices"], data["edge_list"])
train_mask = data["train_mask"]
val_mask = data["val_mask"]
test_mask = data["test_mask"]

idx_train = np.where(train_mask)[0]
idx_val = np.where(val_mask)[0]
idx_test = np.where(test_mask)[0]

X = data["features"]
features = X.numpy()
features_normalized = normalize_features(features)
labels = data["labels"]
features_normalized = torch.FloatTensor(features_normalized)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

g = eg.get_clique_expansion(hg)
print(len(g.edges))

# 创建图
G = nx.Graph()
G.add_edges_from(g.edges)
print(G)

# 得到邻接矩阵
adj = nx.adjacency_matrix(G).toarray()


features = X
# Normalize adj and features
features = features.numpy()
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) 
features_normalized = normalize_features(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
# labels = torch.max(labels, dim=1)[1]
features_normalized = torch.FloatTensor(features_normalized)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

_, cvae_model = baseline_cvae_pretrain_cocitationcora.generated_generator(args, device, adj, features, labels, features_normalized, adj_normalized, idx_train)
#  We have supplied a pretrained model named "%s.pkl" for the %s dataset. If you wish to use your own pretrained model, please save it with the filename "model/%s_.pkl" to avoid overwriting our provided models.
torch.save(cvae_model,"model/%s_1222.pkl"%args.dataset)