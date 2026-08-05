"""
Unified Benchmark Script for Hypergraph Neural Networks (HNN)
============================================================

This script evaluates various HNN baselines on hypergraph datasets.
All hyperparameters are from DHG-Bench for fair comparison.

Supported Models:
- MLP, CEGCN, CEGAT, HGNN (HCHA), HyperGCN, HCHA, LEGCN
- HyperND, PhenomNN, PhenomNNS, TF-HNN, HJRL, DPHGNN
- HyperGT, ED-HNN (EquivSetGNN), T-HyperGNN, UniGCNII
- AllSetTransformer, UniGNN, SheafHyperGNN, EHNN

Usage:
    python baselines/run_hnn_benchmark.py --model dphgnn --dataset cora
    python baselines/run_hnn_benchmark.py --model all --dataset citation
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dhg_datasets import load_dhg_sample
from utils.hypergraph import SimpleHypergraph

# Check torch_sparse availability
TORCH_SPARSE_AVAILABLE = False
try:
    import torch_sparse
    TORCH_SPARSE_AVAILABLE = True
except (ImportError, OSError):
    print("[WARN] torch_sparse not available. Some models will be skipped.")

# Models that require torch_geometric.nn (which may trigger torch_sparse import)
TORCH_GEOMETRIC_MODELS = {'hypergcn', 'legcn', 'unigcnii', 'allset', 'unignn', 'unigencoder', 
                          'cegcn', 'cegat', 'hgnn', 'hnhn', 'sheafhypergnn', 'ehnn', 'mlp'}

# =============================================================================
# Model Hyperparameters (from DHG-Bench)
# =============================================================================

MODEL_CONFIGS = {
    # MLP
    'mlp': {
        'default': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100},
        'cora': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100},
        'citeseer': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100},
        'pubmed': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100},
    },
    
    # CEGCN
    'cegcn': {
        'default': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
        'cora': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
        'citeseer': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
        'pubmed': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
    },
    
    # CEGAT
    'cegat': {
        'default': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 4},
        'cora': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 4},
    },
    
    # HGNN (HCHA)
    'hgnn': {
        'default': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
        'cora': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
        'citeseer': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
        'pubmed': {'hidden': 256, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200, 'heads': 1},
    },
    
    # HNHN
    'hnhn': {
        'default': {'hidden': 256, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': -1.5, 'beta': -0.5},
        'cora': {'hidden': 256, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': -1.5, 'beta': -0.5},
        'citeseer': {'hidden': 256, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': -1.5, 'beta': -0.5},
        'pubmed': {'hidden': 256, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': -1.5, 'beta': -0.5},
    },
    
    # HyperGCN
    'hypergcn': {
        'default': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200},
        'cora': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200},
        'citeseer': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200},
        'pubmed': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200},
    },
    
    # LEGCN
    'legcn': {
        'default': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'cora': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'citeseer': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'pubmed': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
    },
    
    # AllSet
    'allset': {
        'default': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 500},
        'cora': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 500},
        'citeseer': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 500},
        'pubmed': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 500},
    },
    
    # UniGNN
    'unignn': {
        'default': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': 0.5, 'beta': 0.5},
        'cora': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': 0.5, 'beta': 0.5},
        'citeseer': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': 0.5, 'beta': 0.5},
        'pubmed': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0005, 'epochs': 200, 'alpha': 0.5, 'beta': 0.5},
    },
    
    # SheafHyperGNN
    'sheafhypergnn': {
        'default': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'cora': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'citeseer': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'pubmed': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
    },
    
    # EHNN
    'ehnn': {
        'default': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'cora': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'citeseer': {'hidden': 64, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
        'pubmed': {'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200},
    },
    
    # ====== NEW MODELS ======
    
    # DPHGNN
    'dphgnn': {
        'default': {
            'hidden': 256, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0, 'epochs': 100,
            'expan_dim': 256, 'taa_spatial_dim': 64, 'taa_spectral_dim': 64, 'num_heads': 1,
            'chunk_size': -1, 'spectral_embed_dim': 64, 'fc_dim': 64, 'dff_MLP_hidden': 64,
            'dff_num_layers': 1, 'atten_neg_slope': 0.2,
            'cg_num_layer': 2, 'cg_MLP_hidden': 64, 'cg_dropout': 0.5,
            'hg_num_layer': 2, 'hg_MLP_hidden': 64, 'hg_dropout': 0.5,
            'sg_num_layer': 2, 'sg_MLP_hidden': 64, 'sg_dropout': 0.5,
        },
        'cora': {
            'hidden': 256, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0, 'epochs': 100,
            'expan_dim': 256, 'taa_spatial_dim': 64, 'taa_spectral_dim': 64, 'num_heads': 1,
            'chunk_size': -1, 'spectral_embed_dim': 64, 'fc_dim': 64, 'dff_MLP_hidden': 64,
            'dff_num_layers': 1, 'atten_neg_slope': 0.2,
            'cg_num_layer': 2, 'cg_MLP_hidden': 64, 'cg_dropout': 0.5,
            'hg_num_layer': 2, 'hg_MLP_hidden': 64, 'hg_dropout': 0.5,
            'sg_num_layer': 2, 'sg_MLP_hidden': 64, 'sg_dropout': 0.5,
        },
        'pubmed': {
            'hidden': 256, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0, 'epochs': 100,
            'expan_dim': 256, 'taa_spatial_dim': 64, 'taa_spectral_dim': 64, 'num_heads': 1,
            'chunk_size': -1, 'spectral_embed_dim': 64, 'fc_dim': 64, 'dff_MLP_hidden': 64,
            'dff_num_layers': 1, 'atten_neg_slope': 0.2,
            'cg_num_layer': 1, 'cg_MLP_hidden': 64, 'cg_dropout': 0.5,
            'hg_num_layer': 2, 'hg_MLP_hidden': 64, 'hg_dropout': 0.5,
            'sg_num_layer': 2, 'sg_MLP_hidden': 64, 'sg_dropout': 0.5,
        },
    },
    
    # HyperGT
    'hypergt': {
        'default': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100,
            'num_heads': 4, 'nb_random_features': 30, 'use_bn': True, 'use_gumbel': True,
            'use_residual': True, 'use_act': True, 'use_jk': True, 'nb_gumbel_sample': 10,
            'rb_order': 0, 'rb_trans': 'sigmoid', 'use_edge_loss': True, 'pe': 'HEPEHtEPE',
        },
        'cora': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.005, 'wd': 0.001, 'epochs': 100,
            'num_heads': 4, 'nb_random_features': 30, 'use_bn': True, 'use_gumbel': True,
            'use_residual': True, 'use_act': True, 'use_jk': True, 'nb_gumbel_sample': 10,
            'rb_order': 0, 'rb_trans': 'sigmoid', 'use_edge_loss': True, 'pe': 'HEPEHtEPE',
        },
        'pubmed': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.001, 'epochs': 100,
            'num_heads': 4, 'nb_random_features': 30, 'use_bn': True, 'use_gumbel': True,
            'use_residual': True, 'use_act': True, 'use_jk': True, 'nb_gumbel_sample': 10,
            'rb_order': 0, 'rb_trans': 'sigmoid', 'use_edge_loss': True, 'pe': 'HEPEHtEPE',
        },
    },
    
    # ED-HNN (EquivSetGNN)
    'edhnn': {
        'default': {
            'hidden': 128, 'layers': 1, 'dropout': 0.2, 'lr': 0.001, 'wd': 0.0, 'epochs': 200,
            'edconv_type': 'EquivSet', 'MLP_num_layers': 2, 'MLP2_num_layers': -1, 'MLP3_num_layers': -1,
            'decoder_hidden': 128, 'decoder_num_layer': 1, 'aggregate': 'mean', 'normalization': 'ln',
            'activation': 'prelu', 'AllSet_input_norm': True, 'alpha': 0.5,
        },
        'cora': {
            'hidden': 512, 'layers': 1, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200,
            'edconv_type': 'EquivSet', 'MLP_num_layers': 0, 'MLP2_num_layers': 1, 'MLP3_num_layers': 1,
            'decoder_hidden': 256, 'decoder_num_layer': 1, 'aggregate': 'mean', 'normalization': 'ln',
            'activation': 'prelu', 'AllSet_input_norm': True, 'alpha': 0.2,
        },
        'pubmed': {
            'hidden': 512, 'layers': 1, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100,
            'edconv_type': 'EquivSet', 'MLP_num_layers': 0, 'MLP2_num_layers': 1, 'MLP3_num_layers': 1,
            'decoder_hidden': 256, 'decoder_num_layer': 1, 'aggregate': 'mean', 'normalization': 'ln',
            'activation': 'prelu', 'AllSet_input_norm': True, 'alpha': 0.8,
        },
    },
    
    # HJRL
    'hjrl': {
        'default': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100,
            'activation': 'relu', 'neg_slope': 0.01, 'gamma': 0.1, 'sample_ratio': 0.1,
        },
        'cora': {
            'hidden': 256, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100,
            'activation': 'relu', 'neg_slope': 0.01, 'gamma': 0.01, 'sample_ratio': False,
        },
        'pubmed': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.0, 'epochs': 100,
            'activation': 'relu', 'neg_slope': 0.01, 'gamma': 0.1, 'sample_ratio': False,
        },
    },
    
    # TF-HNN
    'tfhnn': {
        'default': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100,
            'MLP_num_layers': 2, 'alpha': 0.5, 'normalization': 'ln',
        },
        'cora': {
            'hidden': 256, 'layers': 2, 'dropout': 0.7, 'lr': 0.001, 'wd': 0.0, 'epochs': 30,
            'MLP_num_layers': 3, 'alpha': 0.1, 'normalization': 'ln',
        },
        'pubmed': {
            'hidden': 512, 'layers': 2, 'dropout': 0.7, 'lr': 0.001, 'wd': 0.0, 'epochs': 200,
            'MLP_num_layers': 3, 'alpha': 0.05, 'normalization': 'ln',
        },
    },
    
    # PhenomNN
    'phenomnn': {
        'default': {
            'hidden': 128, 'layers': 1, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 100,
            'lam0': 10, 'lam1': 10, 'alpha': 0.1, 'prop_step': 16, 'normalization': 'ln',
            'encoder_num_layers': 1, 'decoder_hidden': 128, 'decoder_num_layers': 1,
        },
        'cora': {
            'hidden': 64, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.00001, 'epochs': 50,
            'lam0': 10, 'lam1': 10, 'alpha': 0.5, 'prop_step': 16, 'normalization': 'ln',
            'encoder_num_layers': 1, 'decoder_hidden': 128, 'decoder_num_layers': 1,
        },
        'pubmed': {
            'hidden': 64, 'layers': 1, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.00001, 'epochs': 300,
            'lam0': 10, 'lam1': 10, 'alpha': 0.2, 'prop_step': 16, 'normalization': 'ln',
            'encoder_num_layers': 1, 'decoder_hidden': 128, 'decoder_num_layers': 1,
        },
    },
    
    # HyperND
    'hypernd': {
        'default': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.001, 'wd': 0.0, 'epochs': 200,
            'HyperND_ord': 1.0, 'HyperND_tol': 1e-4, 'HyperND_steps': 100, 'restart_alpha': 0.5,
            'normalization': 'ln', 'aggregate': 'mean',
        },
        'cora': {
            'hidden': 128, 'layers': 2, 'dropout': 0.5, 'lr': 0.01, 'wd': 0.00001, 'epochs': 50,
            'HyperND_ord': 2.0, 'HyperND_tol': 1e-5, 'HyperND_steps': 100, 'restart_alpha': 0.5,
            'normalization': 'ln', 'aggregate': 'sum',
        },
        'pubmed': {
            'hidden': 512, 'layers': 2, 'dropout': 0.2, 'lr': 0.01, 'wd': 0.00001, 'epochs': 300,
            'HyperND_ord': 1.0, 'HyperND_tol': 1e-3, 'HyperND_steps': 100, 'restart_alpha': 0.2,
            'normalization': 'ln', 'aggregate': 'sum',
        },
    },
    
    # UniGCNII
    'unigcnii': {
        'default': {
            'hidden': 256, 'layers': 2, 'dropout': 0.8, 'lr': 0.001, 'wd': 0.0, 'epochs': 50,
            'input_drop': 0.5, 'activation': 'prelu', 'use_norm': False,
            'restart_alpha': 0.3, 'lamda': 0.8, 'reduce': 'mean',
        },
        'cora': {
            'hidden': 256, 'layers': 2, 'dropout': 0.8, 'lr': 0.001, 'wd': 0.0, 'epochs': 50,
            'input_drop': 0.5, 'activation': 'prelu', 'use_norm': False,
            'restart_alpha': 0.3, 'lamda': 0.8, 'reduce': 'mean',
        },
        'pubmed': {
            'hidden': 128, 'layers': 2, 'dropout': 0.0, 'lr': 0.001, 'wd': 0.0, 'epochs': 100,
            'input_drop': 0.3, 'activation': 'prelu', 'use_norm': False,
            'restart_alpha': 0.5, 'lamda': 0.5, 'reduce': 'mean',
        },
    },
}


# =============================================================================
# Data Conversion Utilities
# =============================================================================

def hypergraph_to_pyg_data(graph: SimpleHypergraph):
    """Convert SimpleHypergraph to a data object compatible with HNN models."""
    class HypergraphData:
        def __init__(self):
            object.__setattr__(self, '_dict', {})
        
        def __setattr__(self, key, value):
            self._dict[key] = value
        
        def __getattr__(self, key):
            if key == '_dict':
                return object.__getattribute__(self, '_dict')
            return self._dict.get(key)
        
        def to(self, device):
            for k, v in list(self._dict.items()):
                if isinstance(v, torch.Tensor):
                    self._dict[k] = v.to(device)
            return self
    
    data = HypergraphData()
    
    num_nodes = graph.num_nodes
    num_edges = len(graph.hyperedges)
    
    edge_indices = []
    for edge_idx, edge in enumerate(graph.hyperedges):
        if edge:
            for node_id in edge:
                if 0 <= node_id < num_nodes:
                    edge_indices.append((node_id, edge_idx))
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    x = graph.x.clone()
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    
    y = graph.node_labels.clone()
    
    data.x = x
    data.hyperedge_index = edge_index
    data.y = y
    data.num_nodes = num_nodes
    data.num_edges = num_edges
    data.edge_index = edge_index
    
    # Compute clique-expanded edge index for CEGCN/CEGAT models
    clique_edge_index = clique_expansion(edge_index, num_nodes)
    data.clique_edge_index = clique_edge_index
    
    if graph.node_train_mask is not None:
        data.train_mask = graph.node_train_mask.clone()
    if graph.node_val_mask is not None:
        data.val_mask = graph.node_val_mask.clone()
    if graph.node_test_mask is not None:
        data.test_mask = graph.node_test_mask.clone()
    
    return data


# =============================================================================
# Preprocessing Functions
# =============================================================================

def clique_expansion(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Create clique-expanded graph edges."""
    import scipy.sparse as sp
    
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    values = np.ones(len(row))
    
    H = sp.coo_matrix((values, (row, col)), shape=(num_nodes, edge_index[1].max().item() + 1)).astype(np.float32)
    clique_unorm = H.dot(H.transpose())
    clique_unorm.sum_duplicates()
    src_idx, dst_idx = clique_unorm.nonzero()
    clique_edge_index = torch.tensor(np.vstack((src_idx, dst_idx)), dtype=torch.long)
    return clique_edge_index


def hnhc_preprocessing(data) -> None:
    """HNHN preprocessing - compute degree normalization matrices."""
    import torch_scatter
    
    edge_index = data.edge_index.clone()
    num_edges = edge_index.shape[1]
    ones = torch.ones(num_edges, device=edge_index.device)
    
    DV = torch_scatter.scatter_add(ones, edge_index[0], dim=0)
    DE = torch_scatter.scatter_add(ones, edge_index[1], dim=0)
    
    alpha, beta = -0.5, -0.5
    
    D_e_alpha = torch.pow(DE, torch.tensor(alpha, device=DE.device))
    D_e_alpha = torch.where(torch.isinf(D_e_alpha), torch.zeros_like(D_e_alpha), D_e_alpha)
    
    D_v_alpha = torch.zeros(DV.shape[0], device=DV.device)
    for i in range(edge_index.shape[1]):
        D_v_alpha[edge_index[0, i]] += DE[edge_index[1, i]]
    
    D_v_beta = torch.pow(DV, torch.tensor(beta, device=DV.device))
    D_v_beta = torch.where(torch.isinf(D_v_beta), torch.zeros_like(D_v_beta), D_v_beta)
    
    D_e_beta = torch.zeros(DE.shape[0], device=DE.device)
    for i in range(edge_index.shape[1]):
        D_e_beta[edge_index[1, i]] += DV[edge_index[0, i]]
    
    D_v_alpha_inv = 1.0 / (D_v_alpha + 1e-10)
    D_v_alpha_inv = torch.where(torch.isinf(D_v_alpha_inv), torch.zeros_like(D_v_alpha_inv), D_v_alpha_inv)
    D_e_beta_inv = 1.0 / (D_e_beta + 1e-10)
    D_e_beta_inv = torch.where(torch.isinf(D_e_beta_inv), torch.zeros_like(D_e_beta_inv), D_e_beta_inv)
    
    data.D_e_alpha = D_e_alpha.float()
    data.D_v_alpha_inv = D_v_alpha_inv.float()
    data.D_v_beta = D_v_beta.float()
    data.D_e_beta_inv = D_e_beta_inv.float()
    data.DE = DE


# =============================================================================
# Model Wrappers
# =============================================================================

class ModelArgs:
    """Generic args class for model parameters."""
    def __init__(self, params: Dict):
        for k, v in params.items():
            setattr(self, k, v)


def create_model_wrapper(model_name: str, num_features: int, num_classes: int, params: Dict):
    """Create a model wrapper based on model name."""
    
    # Check if model requires torch_geometric and if it's available
    if model_name in TORCH_GEOMETRIC_MODELS and not TORCH_SPARSE_AVAILABLE:
        raise ImportError(f"Model {model_name} requires torch_geometric which is not available")
    
    class GenericWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = None
            self.model_name = model_name
            
            if model_name == 'mlp':
                from baselines.HNN.mlp import MLP
                args = ModelArgs({
                    'MLP_hidden': params['hidden'],
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'normalization': 'ln',
                    'InputNorm': False,
                })
                self.model = MLP(num_features, params['hidden'], num_classes, params['layers'],
                                dropout=params['dropout'], Normalization='ln', InputNorm=False)
                
            elif model_name == 'hgnn':
                from baselines.HNN.hgnn import HCHA
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'HCHA_symdegnorm': False,
                    'MLP_hidden': params['hidden'],
                    'heads': params.get('heads', 1),
                })
                self.model = HCHA(num_features, num_classes, args)
                
            elif model_name == 'hnhn':
                from baselines.HNN.hnhn import HNHN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'HNHN_alpha': params.get('alpha', -1.5),
                    'HNHN_beta': params.get('beta', -0.5),
                    'HNHN_nonlinear_inbetween': True,
                })
                self.model = HNHN(num_features, num_classes, args)
                
            elif model_name == 'hypergcn':
                from baselines.HNN.hypergcn import HyperGCN as HyperGCNModel
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'mediator': False,
                    'HyperGCN_fast': True,
                })
                self.model = HyperGCNModel(num_features, num_classes, args)
                
            elif model_name == 'legcn':
                from baselines.HNN.legcn import LEGCN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                })
                self.model = LEGCN(num_features, num_classes, args)
                
            elif model_name == 'allset':
                from baselines.HNN.allset import SetGNN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                })
                self.model = SetGNN(num_features, num_classes, args)
                
            elif model_name == 'unignn':
                from baselines.HNN.unignn import UniGNN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'alpha': params.get('alpha', 0.5),
                    'beta': params.get('beta', 0.5),
                })
                self.model = UniGNN(num_features, num_classes, args)
                
            elif model_name == 'ehnn':
                from baselines.HNN.ehnn import EHNN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                })
                self.model = EHNN(num_features, num_classes, args)
                
            elif model_name == 'cegcn' or model_name == 'cegat':
                from baselines.HNN.cegnn import CEGCN, CEGAT
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'heads': params.get('heads', 1),
                    'use_bn': False,
                    'concat': True,
                })
                if model_name == 'cegat':
                    self.model = CEGAT(num_features, num_classes, args)
                else:
                    self.model = CEGCN(num_features, num_classes, args)
                
            elif model_name == 'sheafhypergnn':
                from baselines.HNN.sheafhypergnn import SheafHyperGNN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                })
                self.model = SheafHyperGNN(num_features, num_classes, args)
                
            elif model_name == 'dphgnn':
                from baselines.HNN.dphgnn import DPHGNN
                args = ModelArgs({
                    'fc_dim': params.get('fc_dim', 64),
                    'dff_MLP_hidden': params.get('dff_MLP_hidden', 64),
                    'dff_num_layers': params.get('dff_num_layers', 1),
                    'atten_neg_slope': params.get('atten_neg_slope', 0.2),
                    'expan_dim': params.get('expan_dim', 256),
                    'taa_spatial_dim': params.get('taa_spatial_dim', 64),
                    'taa_spectral_dim': params.get('taa_spectral_dim', 64),
                    'num_heads': params.get('num_heads', 1),
                    'chunk_size': params.get('chunk_size', -1),
                    'spectral_embed_dim': params.get('spectral_embed_dim', 64),
                    'cg_num_layer': params.get('cg_num_layer', 2),
                    'cg_MLP_hidden': params.get('cg_MLP_hidden', 64),
                    'cg_dropout': params.get('cg_dropout', 0.5),
                    'hg_num_layer': params.get('hg_num_layer', 2),
                    'hg_MLP_hidden': params.get('hg_MLP_hidden', 64),
                    'hg_dropout': params.get('hg_dropout', 0.5),
                    'sg_num_layer': params.get('sg_num_layer', 2),
                    'sg_MLP_hidden': params.get('sg_MLP_hidden', 64),
                    'sg_dropout': params.get('sg_dropout', 0.5),
                    'device': torch.device('cpu'),
                })
                self.model = DPHGNN(num_features, num_classes, args)
                
            elif model_name == 'hypergt':
                from baselines.HNN.hypergt import HyperGT
                args = ModelArgs({
                    'hidden_channels': params['hidden'],
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'num_heads': params.get('num_heads', 4),
                    'nb_random_features': params.get('nb_random_features', 30),
                    'use_bn': params.get('use_bn', True),
                    'use_gumbel': params.get('use_gumbel', True),
                    'use_residual': params.get('use_residual', True),
                    'use_act': params.get('use_act', True),
                    'use_jk': params.get('use_jk', True),
                    'nb_gumbel_sample': params.get('nb_gumbel_sample', 10),
                    'rb_order': params.get('rb_order', 0),
                    'rb_trans': params.get('rb_trans', 'sigmoid'),
                    'use_edge_loss': params.get('use_edge_loss', True),
                    'pe': params.get('pe', 'HEPEHtEPE'),
                    'device': torch.device('cpu'),
                })
                self.model = HyperGT(num_features, num_classes, args)
                
            elif model_name == 'edhnn':
                from baselines.HNN.edgnn import EquivSetGNN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'edconv_type': params.get('edconv_type', 'EquivSet'),
                    'MLP_num_layers': params.get('MLP_num_layers', 2),
                    'MLP2_num_layers': params.get('MLP2_num_layers', -1),
                    'MLP3_num_layers': params.get('MLP3_num_layers', -1),
                    'decoder_hidden': params.get('decoder_hidden', 128),
                    'decoder_num_layer': params.get('decoder_num_layer', 1),
                    'aggregate': params.get('aggregate', 'mean'),
                    'normalization': params.get('normalization', 'ln'),
                    'activation': params.get('activation', 'prelu'),
                    'AllSet_input_norm': params.get('AllSet_input_norm', True),
                    'alpha': params.get('alpha', 0.5),
                })
                self.model = EquivSetGNN(num_features, num_classes, args)
                
            elif model_name == 'hjrl':
                from baselines.HNN.hjrl import HJRL
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'activation': params.get('activation', 'relu'),
                    'neg_slope': params.get('neg_slope', 0.01),
                })
                self.model = HJRL(num_features, num_classes, args)
                
            elif model_name == 'tfhnn':
                from baselines.HNN.tfhnn import TFHNN
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'MLP_num_layers': params.get('MLP_num_layers', 2),
                    'alpha': params.get('alpha', 0.5),
                    'normalization': params.get('normalization', 'ln'),
                    'device': torch.device('cpu'),
                })
                self.model = TFHNN(num_features, num_classes, args)
                
            elif model_name == 'phenomnn':
                from baselines.HNN.phenomnn import PhenomNN
                args = ModelArgs({
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'lam0': params.get('lam0', 10),
                    'lam1': params.get('lam1', 10),
                    'alpha': params.get('alpha', 0.1),
                    'prop_step': params.get('prop_step', 16),
                    'normalization': params.get('normalization', 'ln'),
                    'encoder_num_layers': params.get('encoder_num_layers', 1),
                    'decoder_hidden': params.get('decoder_hidden', 128),
                    'decoder_num_layers': params.get('decoder_num_layers', 1),
                })
                self.model = PhenomNN(num_features, num_classes, args)
                
            elif model_name == 'hypernd':
                from baselines.HNN.hypernd import HyperND
                args = ModelArgs({
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'MLP_num_layers': params['layers'],
                    'HyperND_ord': params.get('HyperND_ord', 1.0),
                    'HyperND_tol': params.get('HyperND_tol', 1e-4),
                    'HyperND_steps': params.get('HyperND_steps', 100),
                    'restart_alpha': params.get('restart_alpha', 0.5),
                    'normalization': params.get('normalization', 'ln'),
                    'aggregate': params.get('aggregate', 'mean'),
                })
                self.model = HyperND(num_features, num_classes, args)
                
            elif model_name == 'unigcnii':
                from baselines.HNN.unigcn2 import UniGCNII
                args = ModelArgs({
                    'All_num_layers': params['layers'],
                    'dropout': params['dropout'],
                    'MLP_hidden': params['hidden'],
                    'input_drop': params.get('input_drop', 0.5),
                    'activation': params.get('activation', 'prelu'),
                    'use_norm': params.get('use_norm', False),
                    'restart_alpha': params.get('restart_alpha', 0.3),
                    'lamda': params.get('lamda', 0.8),
                    'reduce': params.get('reduce', 'mean'),
                })
                self.model = UniGCNII(num_features, num_classes, args)
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        def forward(self, data):
            # MLP and MLP_inner expect tensor x, others expect data object
            if self.model_name in ['mlp', 'mlp_inner']:
                return self.model(data.x)
            else:
                return self.model(data)
    
    return GenericWrapper()


# =============================================================================
# Training and Evaluation
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def preprocess_data(data, model_name: str):
    """Apply model-specific preprocessing to data."""
    if model_name == 'hnhn':
        hnhc_preprocessing(data)
    elif model_name == 'hypergt':
        # Prepare hypergraph incidence matrix H for HyperGT
        _prepare_hypergt_data(data)
    elif model_name == 'hjrl':
        # Prepare HJRL required matrices
        _prepare_hjrl_data(data)
    elif model_name == 'phenomnn' or model_name == 'phenomnns':
        # Prepare PhenomNN required matrices
        _prepare_phenomnn_data(data)
    return data


def _prepare_hypergt_data(data):
    """Prepare data for HyperGT model."""
    import scipy.sparse as sp
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index[1].max().item() + 1
    
    # Create incidence matrix H (num_nodes x num_edges)
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    values = np.ones(len(row))
    H_coo = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_edges))
    H = torch.sparse_coo_tensor(
        indices=torch.from_numpy(np.vstack([H_coo.row, H_coo.col])),
        values=torch.from_numpy(H_coo.data),
        size=H_coo.shape
    ).float()
    
    # Create adjacency list (adjs) from edge_index
    # Convert edge_index to list of edges for each node
    adjs = [edge_index]  # Use edge_index as adjacency
    data.adjs = adjs
    data.H = H.to(edge_index.device)
    
    return data


def _prepare_hjrl_data(data):
    """Prepare data for HJRL model."""
    import scipy.sparse as sp
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index[1].max().item() + 1
    
    # Create incidence matrix H (num_nodes x num_edges)
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    values = np.ones(len(row))
    H_coo = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_edges))
    H = torch.from_numpy(H_coo.toarray()).float()
    HT = H.t()
    
    # Compute HHT and HTH
    HHT = torch.mm(H, HT)  # Node similarity (num_nodes x num_nodes)
    HTH = torch.mm(HT, H)  # Hyperedge similarity (num_edges x num_edges)
    
    # Degree matrices for normalization
    D_v = torch.diag(H.sum(dim=1))
    D_e = torch.diag(H.sum(dim=0))
    D_v_inv = torch.inverse(D_v + torch.eye(num_nodes) * 1e-10)
    D_e_inv = torch.inverse(D_e + torch.eye(num_edges) * 1e-10)
    
    # norm_X is the normalized feature matrix (num_nodes x num_features)
    # HJRL uses node features (data.x), not incidence matrix
    norm_X = data.x  # Use original features
    norm_E = data.x[:num_edges] if data.x.shape[0] >= num_edges else torch.zeros(num_edges, data.x.shape[1])
    
    data.norm_X = norm_X.to(edge_index.device)
    data.norm_E = norm_E.to(edge_index.device)
    data.HHT = HHT.to(edge_index.device)
    data.H = H.to(edge_index.device)
    data.HT = HT.to(edge_index.device)
    data.HTH = HTH.to(edge_index.device)
    
    return data


def _prepare_phenomnn_data(data):
    """Prepare data for PhenomNN model."""
    import scipy.sparse as sp
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index[1].max().item() + 1
    
    # Create incidence matrix H
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    values = np.ones(len(row))
    H_coo = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_edges))
    H = torch.from_numpy(H_coo.toarray()).float()
    HT = H.t()
    
    # Compute adjacency matrices
    # A_beta: node-to-node via hyperedges
    HHT = torch.mm(H, HT)
    HTH = torch.mm(HT, H)
    
    # D_beta for node normalization
    D_beta = torch.diag(HHT.sum(dim=1))
    I = torch.eye(num_nodes)
    
    # Normalized node adjacency
    D_beta_inv_sqrt = torch.inverse(torch.sqrt(D_beta + torch.eye(num_nodes) * 1e-10))
    A_beta = torch.mm(torch.mm(D_beta_inv_sqrt, HHT), D_beta_inv_sqrt)
    
    # PhenomNN uses A_beta for both adj entries (simplified version)
    # and D_beta for both D_beta and D_gamma in G
    data.adj = [A_beta, A_beta]  # Simplified: use same adj
    data.G = [D_beta, D_beta, I]
    data.adj = [a.to(edge_index.device) for a in data.adj]
    data.G = [g.to(edge_index.device) for g in data.G]
    
    return data


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    if isinstance(out, tuple):
        out = out[0]
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data)
    if isinstance(out, tuple):
        out = out[0]
    pred = out.argmax(dim=1)
    
    results = {}
    if hasattr(data, 'val_mask') and data.val_mask is not None:
        results['val_acc'] = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    if hasattr(data, 'test_mask') and data.test_mask is not None:
        results['test_acc'] = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return results


def run_single_experiment(model_name: str, data, num_classes, params, max_epochs, patience, seed):
    set_seed(seed)
    
    data = preprocess_data(data, model_name)
    num_features = data.x.size(1)
    
    model = create_model_wrapper(model_name, num_features, num_classes, params)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params.get('wd', 0.0))
    
    best_val_acc = 0
    best_test_acc = 0
    bad_epochs = 0
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, data, optimizer)
        metrics = evaluate(model, data)
        
        val_acc = metrics.get('val_acc', 0)
        test_acc = metrics.get('test_acc', 0)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        if bad_epochs >= patience:
            break
    
    return {
        'val_acc': best_val_acc,
        'test_acc': best_test_acc,
        'epochs_trained': epoch + 1,
        'final_train_loss': train_loss
    }


def run_benchmark(model_name: str, dataset_name: str, num_seeds: int = 3,
                  max_epochs: int = 500, patience: int = 50,
                  output_dir: str = "baselines/results") -> Dict:
    print(f"\n{'='*60}")
    print(f"Running Benchmark: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Get model config
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    dataset_key = dataset_name if dataset_name in config else 'default'
    params = copy.deepcopy(config.get(dataset_key, config['default']))
    params['epochs'] = min(params.get('epochs', 100), max_epochs)
    
    print(f"Hyperparameters: {params}")
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    try:
        graph = load_dhg_sample(
            dataset_name=dataset_name,
            target_dim=128,
            seed=42,
            require_node_splits=True
        )
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return {'error': str(e)}
    
    print(f"Converting to PyG format...")
    data = hypergraph_to_pyg_data(graph)
    
    num_classes = int(graph.metadata.get('num_node_classes', 0))
    print(f"  Nodes: {graph.num_nodes}, Edges: {len(graph.hyperedges)}")
    print(f"  Features: {data.x.size(1)}, Classes: {num_classes}")
    print(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    # Run experiments
    results = []
    for seed_idx in range(num_seeds):
        seed = 7 + seed_idx * 100
        print(f"\n  Seed {seed_idx + 1}/{num_seeds} (seed={seed})...")
        
        try:
            result = run_single_experiment(
                model_name=model_name,
                data=data,
                num_classes=num_classes,
                params=params,
                max_epochs=params['epochs'],
                patience=patience,
                seed=seed
            )
            results.append(result)
            print(f"    Val Acc: {result['val_acc']:.4f}, Test Acc: {result['test_acc']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        return {'error': f'All experiments failed for {model_name} on {dataset_name}'}
    
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    summary = {
        'model': model_name,
        'dataset': dataset_name,
        'num_seeds': len(results),
        'val_acc_mean': float(np.mean(val_accs)),
        'val_acc_std': float(np.std(val_accs)),
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'individual_results': results,
        'hyperparameters': params
    }
    
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Val Acc: {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f}")
    print(f"  Test Acc: {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"{'='*60}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{model_name}_{dataset_name}.json")
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {result_file}")
    
    return summary


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark HNN models on hypergraph datasets")
    parser.add_argument('--model', type=str, default='all',
                        help='Model to evaluate (or "all" for all models)')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'cora_cc', 'citeseer_cc', 'pubmed_cc', 'citation'],
                        help='Dataset to evaluate')
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='baselines/results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Determine datasets to run
    if args.dataset == 'citation':
        datasets = ['cora', 'citeseer', 'pubmed']
    else:
        datasets = [args.dataset]
    
    # Determine models to run
    if args.model == 'all':
        models = list(MODEL_CONFIGS.keys())
    elif args.model == 'new':
        # Only run newly implemented models that don't require torch_geometric
        models = ['dphgnn', 'hypergt', 'edhnn', 'hjrl', 'tfhnn', 'phenomnn', 'phenomnns', 'hypernd', 'unigcnii']
    elif args.model == 'available':
        # Only run models that should work in current environment
        models = ['dphgnn', 'hypergt', 'edhnn', 'hjrl', 'tfhnn', 'phenomnn', 'phenomnns', 'hypernd']
    else:
        models = [args.model]
    
    print(f"Models to run: {models}")
    print(f"Datasets to run: {datasets}")
    
    # Run benchmarks
    all_results = []
    for model_name in models:
        for dataset_name in datasets:
            try:
                result = run_benchmark(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    num_seeds=args.num_seeds,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    output_dir=args.output_dir
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error running {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save combined results
    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        combined_file = os.path.join(args.output_dir, 'combined_results.json')
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {combined_file}")
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
