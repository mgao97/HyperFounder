# Try to import torch_sparse, but don't fail if it's not available
TORCH_SPARSE_AVAILABLE = False
try:
    import torch_sparse
    TORCH_SPARSE_AVAILABLE = True
except (ImportError, OSError):
    pass

# Models that require torch_sparse or torch_geometric - only import if available
TORCH_GEOMETRIC_AVAILABLE = False

def _get_dummy_model(name):
    """Return None for unavailable models."""
    return None

if TORCH_SPARSE_AVAILABLE:
    try:
        # Base MLP first (safe, doesn't import torch_sparse directly)
        from .mlp import MLP, PlainMLP
        
        # New models that don't require torch_sparse
        from .dphgnn import DPHGNN
        from .edgnn import EquivSetGNN
        from .hjrl import HJRL
        from .tfhnn import TFHNN
        from .phenomnn import PhenomNN
        from .phenomnns import PhenomNNS
        from .hypernd import HyperND
        from .ehnn_transformer import EHNNTransformerConv
        
        # Models that require torch_sparse or torch_geometric
        from .hypergt import HyperGT
        from .hgnn import HCHA
        from .hnhn import HNHN
        from .allset import SetGNN
        from .unignn import UniGNN
        from .legcn import LEGCN
        from .sheafhypergnn import SheafHyperGNN
        from .ehnn import EHNN
        from .cegnn import CEGCN, CEGAT
        from .hypergcn import HyperGCN
        from .unigcn2 import UniGCNII
        from .unigencoder import PlainUnigencoder
        
        TORCH_GEOMETRIC_AVAILABLE = True
    except (ImportError, OSError) as e:
        # Provide dummy exports for models that failed to import
        MLP = PlainMLP = _get_dummy_model('MLP')
        DPHGNN = _get_dummy_model('DPHGNN')
        EquivSetGNN = _get_dummy_model('EquivSetGNN')
        HJRL = _get_dummy_model('HJRL')
        TFHNN = _get_dummy_model('TFHNN')
        PhenomNN = _get_dummy_model('PhenomNN')
        PhenomNNS = _get_dummy_model('PhenomNNS')
        HyperND = _get_dummy_model('HyperND')
        EHNNTransformerConv = _get_dummy_model('EHNNTransformerConv')
        HyperGT = _get_dummy_model('HyperGT')
        HCHA = _get_dummy_model('HCHA')
        HNHN = _get_dummy_model('HNHN')
        SetGNN = _get_dummy_model('SetGNN')
        UniGNN = _get_dummy_model('UniGNN')
        LEGCN = _get_dummy_model('LEGCN')
        SheafHyperGNN = _get_dummy_model('SheafHyperGNN')
        EHNN = _get_dummy_model('EHNN')
        CEGCN = CEGAT = _get_dummy_model('CEGCN/CEGAT')
        HyperGCN = _get_dummy_model('HyperGCN')
        UniGCNII = _get_dummy_model('UniGCNII')
        PlainUnigencoder = _get_dummy_model('PlainUnigencoder')
else:
    # Provide dummy exports
    MLP = PlainMLP = None
    DPHGNN = None
    EquivSetGNN = None
    HJRL = None
    TFHNN = None
    PhenomNN = None
    PhenomNNS = None
    HyperND = None
    EHNNTransformerConv = None
    HyperGT = None
    HCHA = None
    HNHN = None
    SetGNN = None
    UniGNN = None
    LEGCN = None
    SheafHyperGNN = None
    EHNN = None
    CEGCN = CEGAT = None
    HyperGCN = None
    UniGCNII = None
    PlainUnigencoder = None

# Re-export availability flags
__all__ = [
    'HCHA', 'HNHN', 'HyperGCN', 'SetGNN', 'UniGNN', 'UniGCNII',
    'LEGCN', 'HyperND', 'EquivSetGNN', 'PlainUnigencoder', 
    'HJRL', 'SheafHyperGNN', 'EHNN', 'DPHGNN', 'HyperGT',
    'TFHNN', 'PhenomNN', 'PhenomNNS', 'CEGCN', 'CEGAT',
    'EHNNTransformerConv', 'MLP', 'PlainMLP',
    'TORCH_SPARSE_AVAILABLE', 'TORCH_GEOMETRIC_AVAILABLE'
]
