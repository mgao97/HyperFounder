"""HyperGFSE: a hypergraph-adapted GFSE (WWW'26) structural encoder.

Public surface mirrors GFSE so that existing GFSE downstream pipelines can be
reused: call HyperGFSE(H) to obtain a PSE of shape (N, output_dim), then
concatenate it with raw features or project it into an LLM embedding space.
"""
from .encoding import HypergraphRandomWalkPE
from .hypergps import HyperGPSLayer, HyperMPNN, BiasedAttention
from .encoder import HyperGFSE
from .tasks import (
    PairHead, NodeHead, EmbedHead,
    UncertaintyWeights, compute_hmotif_labels, hypergraph_community_matrix,
    hspd_loss, motif_loss, community_loss, community_loss_pairs, gcl_loss,
)
from .pretrain import Pretrainer, linear_probe_eval, build_pretrain_item
from .load_benchmark import load_benchmark, ALL_BENCHMARK_DATASETS, BenchGraph

__all__ = [
    "HypergraphRandomWalkPE", "HyperGPSLayer", "HyperMPNN", "BiasedAttention",
    "HyperGFSE", "PairHead", "NodeHead", "EmbedHead", "UncertaintyWeights",
    "compute_hmotif_labels", "hypergraph_community_matrix",
    "hspd_loss", "motif_loss", "community_loss", "community_loss_pairs", "gcl_loss",
    "Pretrainer", "linear_probe_eval", "build_pretrain_item",
    "load_benchmark", "ALL_BENCHMARK_DATASETS", "BenchGraph",
]
