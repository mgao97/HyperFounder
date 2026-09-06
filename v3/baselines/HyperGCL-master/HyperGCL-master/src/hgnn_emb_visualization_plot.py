import time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dhg import Hypergraph
from dhg.data import *
from dhg.models import *
from dhg.random import set_seed
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

# load dataset
data = CocitationCora()
G = Hypergraph(data["num_vertices"], data["edge_list"])
print(G)

# # 设置随机种子，以确保结果可复现
# random_seed = 42

# node_idx = [i for i in range(data['num_vertices'])]
# # 将idx_test划分为训练（50%）、验证（25%）和测试（25%）集
# idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
# idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# # 确保划分后的集合没有重叠
# assert len(set(idx_train) & set(idx_val)) == 0
# assert len(set(idx_train) & set(idx_test)) == 0
# assert len(set(idx_val) & set(idx_test)) == 0

# train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
# train_mask[idx_train] = True
# val_mask[idx_val] = True
# test_mask[idx_test] = True

# v_deg= G.D_v
# X = v_deg.to_dense()/torch.max(v_deg.to_dense())
X = data["features"]
lbls = data["labels"]
print('X dim:', X.shape)
print('labels:', len(torch.unique(lbls)))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


net = HGNN(X.shape[1], 256, data["num_classes"], use_bn=True)
net.load_state_dict(torch.load('model/hgnn_cocitationcora_best_model.pth'))
X, lbls = X.to(device), lbls.to(device)
G = G.to(device)
net = net.to(device)

hgnn_emb = net(X,G)
print(f'hgnn_emb: {hgnn_emb}', f'hgnn_emb size: {hgnn_emb.shape}')

tsne = TSNE(n_components=2, verbose=1, random_state=0)
z = tsne.fit_transform(hgnn_emb.detach().numpy())
z_data = np.vstack((z.T, lbls)).T
df_tsne = pd.DataFrame(z_data, columns=['Dimension 1', 'Dimension 2', 'Class'])
df_tsne['Class'] = df_tsne['Class'].astype(int)
plt.figure(figsize=(8, 8))
sns.set(font_scale=1.5)
plt.legend(loc='upper right')
sns.scatterplot(data=df_tsne, hue='Class', x='Dimension 1', y='Dimension 2', palette=['green','orange','brown','red', 'blue','black','purple'])
plt.savefig("emb_figs/hgnn_cocitationcora.pdf", bbox_inches="tight") # save embeddings if needed
plt.savefig("emb_figs/hgnn_cocitationcora.png", bbox_inches="tight")
plt.show()
