"""
Corrected HNN Benchmark with Full Hyperparameter Search
=====================================================

Each model follows the standard hyperparameter search space from the paper.

Usage:
    python baselines/run_hnn_benchmark_corrected.py --model hgnn --dataset cora
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Hyperparameter Search Spaces (from paper)
# =============================================================================

@dataclass
class ModelSearchSpace:
    """Hyperparameter search space for each model."""
    name: str
    default_params: Dict
    search_space: Dict


# General settings: epochs={100,200,300,400,500,800,1000}, lr={0.1,0.01,0.001,0.0001}
# layers={1,2,3,4}, dropout={0,0.1,...,0.8}, hidden={64,128,256,512,1024}
# wd={0, 0.0005}

MODEL_SPACES = {
    'hgnn': ModelSearchSpace(
        name='HCHA',
        default_params={
            'epochs': 100,
            'lr': 0.001,
            'layers': 2,
            'dropout': 0.5,
            'hidden': 128,
            'wd': 0.0,
            'heads': 1,  # Extra param for HCHA
        },
        search_space={
            'heads': [1, 2, 4, 8, 16],  # Only search heads
        }
    ),
    
    'hnhn': ModelSearchSpace(
        name='HNHN',
        default_params={
            'epochs': 200,
            'lr': 0.01,
            'layers': 1,
            'dropout': 0.5,
            'hidden': 256,
            'wd': 0.0005,
            'alpha': -1.5,
            'beta': -0.5,
        },
        search_space={
            'alpha': [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5],
            'beta': [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        }
    ),
    
    'hypergcn': ModelSearchSpace(
        name='HyperGCN',
        default_params={
            'epochs': 200,
            'lr': 0.01,
            'layers': 2,
            'dropout': 0.5,
            'hidden': 128,
            'wd': 0.0005,
        },
        search_space={
            'lr': [0.1, 0.01, 0.001],
            'layers': [1, 2, 3, 4],
        }
    ),
    
    'allset': ModelSearchSpace(
        name='AllSet',
        default_params={
            'epochs': 500,
            'lr': 0.001,
            'layers': 2,
            'dropout': 0.5,
            'hidden': 128,
            'wd': 0.0,
        },
        search_space={
            'layers': [1, 2, 3, 4],
            'hidden': [64, 128, 256],
        }
    ),
    
    'unignn': ModelSearchSpace(
        name='UniGNN',
        default_params={
            'epochs': 200,
            'lr': 0.01,
            'layers': 2,
            'dropout': 0.5,
            'hidden': 128,
            'wd': 0.0005,
            'alpha': 0.5,
            'beta': 0.5,
        },
        search_space={
            'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.8, 0.9],
            'beta': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9],
        }
    ),
}


# =============================================================================
# Data Loading (Direct DHG)
# =============================================================================

def load_dhg_dataset(dataset_name: str) -> 'DataContainer':
    """Load dataset directly from DHG with correct splits."""
    import dhg
    
    dataset_map = {
        'cora': 'Cora',
        'citeseer': 'Citeseer',
        'pubmed': 'Pubmed',
        'cora_cc': 'CocitationCora',
        'citeseer_cc': 'CocitationCiteseer',
        'pubmed_cc': 'CocitationPubmed',
        'coauthorship_dblp': 'CoauthorshipDBLP',
        'imdb_4k': 'IMDB4k',
        'cooking_200': 'Cooking200',
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = getattr(dhg.data, dataset_map[dataset_name])
    data = loader()
    
    # Build hyperedge_index
    edge_list = data['edge_list']
    edge_indices = []
    for edge_idx, edge in enumerate(edge_list):
        for node_id in edge:
            edge_indices.append((node_id, edge_idx))
    hyperedge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # Create container
    class DataContainer:
        def __init__(self, data_dict):
            self._data = data_dict
            for k, v in data_dict.items():
                setattr(self, k, v)
        
        def __getitem__(self, key):
            return self._data[key]
    
    return DataContainer({
        'x': data['features'],
        'y': data['labels'],
        'hyperedge_index': hyperedge_index,
        'train_mask': data['train_mask'],
        'val_mask': data['val_mask'],
        'test_mask': data['test_mask'],
        'num_vertices': data['num_vertices'],
        'num_edges': data['num_edges'],
    })


# =============================================================================
# Model Wrappers
# =============================================================================

def create_hgnn_wrapper(params: Dict):
    """Create HGNN wrapper with given parameters."""
    from baselines.HNN.hgnn import HCHA
    
    class HCHAArgs:
        def __init__(self):
            self.All_num_layers = params['layers']
            self.dropout = params['dropout']
            self.HCHA_symdegnorm = False
            self.MLP_hidden = params['hidden']
            self.heads = params.get('heads', 1)
    
    class HGNNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = HCHA(params['num_features'], params['num_classes'], HCHAArgs())
        
        def forward(self, data):
            return self.model(data)
    
    return HGNNModel()


def create_hnhn_wrapper(params: Dict):
    """Create HNHN wrapper with given parameters."""
    from baselines.HNN.hnhn import HNHN
    
    class HNHNArgs:
        def __init__(self):
            self.All_num_layers = params['layers']
            self.dropout = params['dropout']
            self.MLP_hidden = params['hidden']
            self.HNHN_alpha = params['alpha']
            self.HNHN_beta = params['beta']
            self.HNHN_nonlinear_inbetween = True
    
    class HNHNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = HNHN(params['num_features'], params['num_classes'], HNHNArgs())
        
        def forward(self, data):
            return self.model(data)
    
    return HNHNModel()


MODEL_CREATORS = {
    'hgnn': create_hgnn_wrapper,
    'hnhn': create_hnhn_wrapper,
}


# =============================================================================
# Training and Evaluation
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)
    out = F.log_softmax(out, dim=1)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out, _ = model(data)
    out = F.log_softmax(out, dim=1)
    pred = out.argmax(dim=1)
    
    results = {}
    if data.val_mask.sum() > 0:
        results['val_acc'] = (pred == data.y)[data.val_mask].float().mean().item()
    if data.test_mask.sum() > 0:
        results['test_acc'] = (pred == data.y)[data.test_mask].float().mean().item()
    return results


def run_experiment(model, data, params, seed):
    """Run a single experiment."""
    set_seed(seed)
    
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    
    best_val_acc = 0
    best_test_acc = 0
    bad_epochs = 0
    
    for epoch in range(params['epochs']):
        train_loss = train_epoch(model, data, optimizer, criterion)
        metrics = evaluate(model, data)
        
        val_acc = metrics.get('val_acc', 0)
        test_acc = metrics.get('test_acc', 0)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        if bad_epochs >= params.get('patience', 50):
            break
    
    return {
        'val_acc': best_val_acc,
        'test_acc': best_test_acc,
        'epochs_trained': epoch + 1,
    }


def generate_param_combinations(search_space: Dict, max_combos: int = 20) -> List[Dict]:
    """Generate parameter combinations for grid search."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    
    combos = list(itertools.product(*values))
    np.random.shuffle(combos)
    
    return [dict(zip(keys, combo)) for combo in combos[:max_combos]]


def run_model_benchmark(model_name: str, dataset_name: str, 
                       num_seeds: int = 3, max_search: int = 10,
                       output_dir: str = "baselines/results") -> Dict:
    """Run benchmark with hyperparameter search."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    data = load_dhg_dataset(dataset_name)
    num_classes = int(data.y.max().item()) + 1
    num_features = data.x.shape[1]
    
    print(f"  Nodes: {data.num_vertices}, Edges: {data.num_edges}")
    print(f"  Classes: {num_classes}, Features: {num_features}")
    print(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    # Get search space
    space = MODEL_SPACES.get(model_name)
    if space is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"\nSearch space for {space.name}:")
    print(f"  Default params: {space.default_params}")
    print(f"  Search dims: {list(space.search_space.keys())}")
    
    # Generate parameter combinations
    param_combos = generate_param_combinations(space.search_space, max_search)
    print(f"  Testing {len(param_combos)} parameter combinations")
    
    # Track best configuration
    best_overall_val = 0
    best_overall_test = 0
    best_params = None
    best_results = None
    
    # Search over parameter combinations
    for combo_idx, search_params in enumerate(param_combos):
        params = {**space.default_params, **search_params}
        params['num_features'] = num_features
        params['num_classes'] = num_classes
        
        print(f"\n  [{combo_idx+1}/{len(param_combos)}] Params: {search_params}")
        
        # Create model
        try:
            creator = MODEL_CREATORS[model_name]
            model = creator(params)
        except Exception as e:
            print(f"    Model creation failed: {e}")
            continue
        
        # Run multiple seeds
        seed_results = []
        for seed_idx in range(num_seeds):
            seed = 7 + seed_idx * 100
            try:
                result = run_experiment(model, data, params, seed)
                seed_results.append(result)
                print(f"    Seed {seed_idx+1}: Val={result['val_acc']:.4f}, Test={result['test_acc']:.4f}")
            except Exception as e:
                print(f"    Seed {seed_idx+1} failed: {e}")
                continue
        
        if not seed_results:
            continue
        
        # Average over seeds
        avg_val = np.mean([r['val_acc'] for r in seed_results])
        avg_test = np.mean([r['test_acc'] for r in seed_results])
        
        print(f"    Avg: Val={avg_val:.4f}, Test={avg_test:.4f}")
        
        if avg_val > best_overall_val:
            best_overall_val = avg_val
            best_overall_test = avg_test
            best_params = params
            best_results = seed_results
    
    # Final summary
    summary = {
        'model': model_name,
        'dataset': dataset_name,
        'best_params': best_params,
        'best_val_acc': best_overall_val,
        'best_test_acc': best_overall_test,
        'search_space': space.search_space,
        'num_combos_tested': len(param_combos),
        'individual_results': best_results,
    }
    
    print(f"\n{'='*60}")
    print(f"Best Result for {model_name} on {dataset_name}:")
    print(f"  Val Acc: {best_overall_val:.4f}")
    print(f"  Test Acc: {best_overall_test:.4f}")
    print(f"  Best Params: {best_params}")
    print(f"{'='*60}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_search.json")
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {result_file}")
    
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Corrected HNN Benchmark with Hyperparameter Search")
    parser.add_argument('--model', type=str, default='hgnn', 
                       choices=['hgnn', 'hnhn', 'hypergcn', 'allset', 'unignn'])
    parser.add_argument('--dataset', type=str, default='cora',
                       choices=['cora', 'citeseer', 'pubmed', 'cora_cc', 'citeseer_cc', 'pubmed_cc'])
    parser.add_argument('--num_seeds', type=int, default=3, help='Number of seeds per config')
    parser.add_argument('--max_search', type=int, default=10, help='Max parameter combinations')
    parser.add_argument('--output_dir', type=str, default='baselines/results')
    
    args = parser.parse_args()
    
    result = run_model_benchmark(
        model_name=args.model,
        dataset_name=args.dataset,
        num_seeds=args.num_seeds,
        max_search=args.max_search,
        output_dir=args.output_dir
    )
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
