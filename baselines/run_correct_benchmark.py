"""
Correct Benchmark Script using DHG directly
==========================================

This script directly uses DHG library to load datasets with correct splits,
avoiding issues with HyperFounder's data loader.

Usage:
    python baselines/run_correct_benchmark.py --model hgnn --dataset cora
"""

from __future__ import annotations

import argparse
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Loading (Direct DHG)
# =============================================================================

def load_dhg_dataset(dataset_name: str) -> Dict:
    """Load dataset directly from DHG with correct splits."""
    import dhg
    
    # Map dataset names to DHG classes
    dataset_map = {
        'cora': ('Cora', None),
        'citeseer': ('Citeseer', None),
        'pubmed': ('Pubmed', None),
        'cora_cc': ('CocitationCora', None),
        'citeseer_cc': ('CocitationCiteseer', None),
        'pubmed_cc': ('CocitationPubmed', None),
        'coauthorship_dblp': ('CoauthorshipDBLP', None),
        'imdb_4k': ('IMDB4k', None),
        'cooking_200': ('Cooking200', None),
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    class_name, _ = dataset_map[dataset_name]
    loader = getattr(dhg.data, class_name)
    data = loader()
    
    # Extract data
    result = {
        'name': dataset_name,
        'num_vertices': data['num_vertices'],
        'num_edges': data['num_edges'],
        'features': data['features'],
        'labels': data['labels'],
        'train_mask': data['train_mask'],
        'val_mask': data['val_mask'],
        'test_mask': data['test_mask'],
    }
    
    # Build edge list
    edge_list = data['edge_list']
    
    # Build hyperedge_index
    edge_indices = []
    for edge_idx, edge in enumerate(edge_list):
        for node_id in edge:
            edge_indices.append((node_id, edge_idx))
    
    hyperedge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # Create a dict-like object that mimics data.x attribute
    class DataContainer:
        def __init__(self, data_dict):
            self._data = data_dict
            for k, v in data_dict.items():
                setattr(self, k, v)
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __contains__(self, key):
            return key in self._data
    
    container = DataContainer({
        'x': result['features'],
        'y': result['labels'],
        'hyperedge_index': hyperedge_index,
        'train_mask': result['train_mask'],
        'val_mask': result['val_mask'],
        'test_mask': result['test_mask'],
        'num_vertices': result['num_vertices'],
        'num_edges': result['num_edges'],
    })
    
    return container


# =============================================================================
# Model Wrappers
# =============================================================================

class HGNNWrapper(torch.nn.Module):
    """Wrapper for HGNN (HCHA) model."""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        from baselines.HNN.hgnn import HCHA
        
        class HCHAArgs:
            def __init__(self):
                self.All_num_layers = num_layers
                self.dropout = dropout
                self.HCHA_symdegnorm = False  # DHG-Bench standard
                self.MLP_hidden = hidden_dim
        
        args = HCHAArgs()
        self.model = HCHA(num_features, num_classes, args)
    
    def forward(self, data):
        return self.model(data)


# Model registry
MODEL_REGISTRY = {
    'hgnn': HGNNWrapper,
}


# =============================================================================
# Training and Evaluation
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_epoch(model, data, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)
    out = F.log_softmax(out, dim=1)
    loss = criterion(out, data['y'])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    """Evaluate model on val/test sets."""
    model.eval()
    out, _ = model(data)
    out = F.log_softmax(out, dim=1)
    pred = out.argmax(dim=1)
    
    results = {}
    
    if 'val_mask' in data:
        results['val_acc'] = (pred == data['y'])[data['val_mask']].float().mean().item()
    
    if 'test_mask' in data:
        results['test_acc'] = (pred == data['y'])[data['test_mask']].float().mean().item()
    
    return results


def run_single_experiment(model_class, data, num_classes, hidden_dim, num_layers,
                          dropout, lr, weight_decay, max_epochs, patience, seed):
    """Run a single experiment with one random seed."""
    set_seed(seed)
    
    # Initialize model
    num_features = data['x'].shape[1]
    model = model_class(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_test_acc = 0
    bad_epochs = 0
    
    for epoch in range(max_epochs):
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
        
        if bad_epochs >= patience:
            break
    
    return {
        'val_acc': best_val_acc,
        'test_acc': best_test_acc,
        'epochs_trained': epoch + 1,
    }


def run_benchmark(model_name: str, dataset_name: str, num_seeds: int = 3,
                   hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.5,
                   lr: float = 0.001, weight_decay: float = 0.0,
                   max_epochs: int = 100, patience: int = 50,
                   output_dir: str = "baselines/results") -> Dict:
    """Run benchmark for a single model on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Running Benchmark: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    data = load_dhg_dataset(dataset_name)
    
    num_classes = int(data['y'].max().item()) + 1
    print(f"  Nodes: {data['num_vertices']}, Edges: {data['num_edges']}")
    print(f"  Features: {data['x'].shape[1]}, Classes: {num_classes}")
    print(f"  Train: {data['train_mask'].sum()}, Val: {data['val_mask'].sum()}, Test: {data['test_mask'].sum()}")
    
    # Get model class
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model_class = MODEL_REGISTRY[model_name]
    
    # Run experiments
    results = []
    for seed_idx in range(num_seeds):
        seed = 7 + seed_idx * 100
        print(f"\n  Seed {seed_idx + 1}/{num_seeds} (seed={seed})...")
        
        try:
            result = run_single_experiment(
                model_class=model_class,
                data=data,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
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
        return {'error': f'All experiments failed'}
    
    # Aggregate results
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
    }
    
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Val Acc: {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f}")
    print(f"  Test Acc: {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"{'='*60}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_correct.json")
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {result_file}")
    
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark HNN models with correct splits")
    parser.add_argument('--model', type=str, default='hgnn', choices=['hgnn'])
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='baselines/results')
    
    args = parser.parse_args()
    
    result = run_benchmark(
        model_name=args.model,
        dataset_name=args.dataset,
        num_seeds=args.num_seeds,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
