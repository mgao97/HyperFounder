import torch
import math
import numpy as np
import scipy.sparse as sp

def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 

def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def create_coo_from_edge_index(edge_index,is_star=False):

    edge_index_cpu = edge_index.to('cpu')

    row = edge_index_cpu[0].numpy()  
    col = edge_index_cpu[1].numpy()  
    values = np.ones(len(row))  

    num_nodes = row.max() + 1  
    num_hyperedges = col.max() + 1  
    
    if is_star:
        H = sp.coo_matrix((values, (row, col)), shape=(num_hyperedges, num_hyperedges)).astype(np.float32) 
    else:
        H = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_hyperedges)).astype(np.float32) 

    return H

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        M1 = M1.float()
        M2 = M2.float()
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g = g.float()
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g = g.float()
            g2 = torch.mm(M1.t(), g)

        return g1, g2