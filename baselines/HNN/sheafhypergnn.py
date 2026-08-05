import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_geometric.nn.conv import MessagePassing
from baselines.HNN.mlp import MLP
from baselines.HNN.utils import zeros,glorot

def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
        r"""
        Power of a matrix using Eigen Decomposition.
        Args:
            matrices: A batch of matrices.
            p: Power.
            positive_definite: If positive definite
        Returns:
            Power of each matrix in the batch.
        """
        # vals, vecs = torch.linalg.eigh(matrices)
        # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
        vecs, vals, _ = torch.linalg.svd(matrices)
        good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
        vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
        matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
        return matrix_power

def sparse_diagonal(diag, shape):
    r,c = shape
    assert r == c
    indexes = torch.arange(r).to(diag.device)
    indexes = torch.stack([indexes, indexes], dim=0)
    return torch.sparse.FloatTensor(indexes, diag)

def generate_indices_general(indexes, d):
    d_range = torch.arange(d)
    d_range_edges = d_range.repeat(d).view(-1,1) #0,1..d,0,1..d..   d*d elems
    d_range_nodes = d_range.repeat_interleave(d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
    indexes = indexes.unsqueeze(1) 

    large_indexes_0 = d * indexes[0] + d_range_nodes
    large_indexes_0 = large_indexes_0.permute((1,0)).reshape(1,-1)
    large_indexes_1 = d * indexes[1] + d_range_edges
    large_indexes_1 = large_indexes_1.permute((1,0)).reshape(1,-1)
    large_indexes = torch.concat((large_indexes_0, large_indexes_1), 0)

    return large_indexes

# helper functions to predict sigma(MLP(x_v || h_e)) varying how thw attributes for hyperedge are computed
def predict_blocks(x, e, hyperedge_index, sheaf_lin, args):
    # e_j = avg(x_v)
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    return h_sheaf

def predict_blocks_var2(x, hyperedge_index, sheaf_lin, args):
    # e_j = avg(h_v)
    row, col = hyperedge_index
    e = scatter_mean(x[row],col, dim=0)
    
    xs = torch.index_select(x, dim=0, index=row)
    es= torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    
    return h_sheaf

def predict_blocks_var3(x, hyperedge_index, sheaf_lin, sheaf_lin2, args):
    # universal approx according to  Equivariant Hypergraph Diffusion Neural Operators
    # # e_j = sum(φ(x_v))

    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    #φ(x_v)
    x_e = sheaf_lin2(x)
    #sum(φ(x_v)
    e = scatter_add(x_e[row],col, dim=0)  
    es= torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse v ersion of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    
    return h_sheaf

def predict_blocks_cp_decomp(x, hyperedge_index, cp_W, cp_V, sheaf_lin, args):
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    xs_ones = torch.cat((xs, torch.ones(xs.shape[0],1).to(xs.device)), dim=-1) #nnz x f+1
    xs_ones_proj = torch.tanh(cp_W(xs_ones)) #nnz x r
    xs_prod =  scatter(xs_ones_proj, col, dim=0, reduce="mul") #edges x r
    e = torch.relu(cp_V(xs_prod))#edges x f
    e = e + torch.relu(scatter_add(x[row],col, dim=0))
    es= torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    return h_sheaf

# Build the restriction maps for the Diagonal Case
class SheafBuilderDiag(nn.Module):
    def __init__(self, args):
        super(SheafBuilderDiag, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = args.stalk_dim # stalk dinension
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout
        
        self.sheaf_lin = MLP(
                    in_channels=2*self.MLP_hidden, 
                    hidden_channels=args.MLP_hidden,
                    out_channels=self.d,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        elif self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)        

    def reset_parameters(self):
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.sheaf_lin.reset_parameters()
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        
    # this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        #predict (_ x d) elements
        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)
        
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        
        if self.special_head:
            new_head_mask = [1]*(self.d-1) + [0]
            new_head = [0]*(self.d-1) + [1]
            h_sheaf = h_sheaf * torch.tensor(new_head_mask, device=x.device) + torch.tensor(new_head, device=x.device)
        
        self.h_sheaf = h_sheaf #this is stored in self for testing purpose
        h_sheaf_attributes = h_sheaf.reshape(-1) #(d*K)

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # We need to modify indices from the NxE matrix 
        # to correspond to the large Nd x Ed matrix, but restrict only on the element of the diagonal of each block
        # indices: scalar [i,j] -> block dxd with indices [d*i, d*i+1.. d*i+d-1; d*j, d*j+1 .. d*j+d-1]
        # attributes: reshape h_sheaf

        d_range = torch.arange(self.d, device=x.device).view(1,-1,1).repeat(2,1,1) #2xdx1
        hyperedge_index = hyperedge_index.unsqueeze(1) #2x1xK
        hyperedge_index = self.d * hyperedge_index + d_range #2xdxK
        hyperedge_index = hyperedge_index.permute(0,2,1).reshape(2,-1) #2x(d*K)
        h_sheaf_index = hyperedge_index

        #the resulting (index, values) pair correspond to the diagonal of each block sub-matrix
        return h_sheaf_index, h_sheaf_attributes

def normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, d, norm_type='degree_norm'):
    #this will return either D^-1/B^-1 or D^(-1/2)/B^-1
    if norm_type == 'degree_norm':
        #return D_inv and B_inv used to normalised the laplacian (/propagation)
        #normalise using node/hyperedge degrees D_e and D_v in the paper
        D = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    elif norm_type == 'sym_degree_norm':
        #normalise using node/hyperedge degrees D_e and D_v in the paper
        D = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
        D = D ** (-0.5)
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    elif norm_type == 'block_norm':
        #normalise using diag(HHT) and deg_e <- this take into account the values predicted in H as oposed to 0/1 as in the degree
        # this way of computing the normalisation tensor is only valid for diagonal sheaf
        D = scatter_add(alpha*alpha, hyperedge_index[0], dim=0, dim_size=num_nodes*d)
        D = 1.0 / D #can compute inverse like this because the matrix is diagonal
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B
    
    elif norm_type == 'sym_block_norm':
        #normalise using diag(HHT) and deg_e <- this take into account the values predicted in H as oposed to 0/1 as in the degree
        # this way of computing the normalisation tensor is only valid for diagonal sheaf
        D = scatter_add(alpha*alpha, hyperedge_index[0], dim=0, dim_size=num_nodes*d)
        D = D ** (-0.5) #can compute inverse like this because the matrix is diagonal
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

# One layer of Sheaf Diffusion with diagonal Laplacian Y = (I-D^-1/2LD^-1) with L normalised with B^-1
class HyperDiffusionDiagSheafConv(MessagePassing):
    r"""
    
    """
    def __init__(self, in_channels, out_channels, d, device, dropout=0, bias=True, norm_type='degree_norm', 
                left_proj=None, norm=None, residual = False, clear_flag=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm_type = norm_type
        self.left_proj = left_proj
        self.norm = norm
        self.residual = residual

        if self.left_proj:
            self.lin_left_proj = MLP(in_channels=d, 
                        hidden_channels=d,
                        out_channels=d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        
        self.lin = MLP(in_channels=in_channels, 
                        hidden_channels=out_channels,
                        out_channels=out_channels,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)      
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.device = device
        
        self.clear_flag=clear_flag

        self.I_mask = None
        self.Id = None

        self.reset_parameters()

    #to allow multiple runs reset all parameters used
    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x , hyperedge_index,
                alpha, 
                num_nodes,
                num_edges):
        r"""
        Args:
            x (Tensor): Node feature matrix {Nd x F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix Nd x Md} from nodes to edges.
            alpha (Tensor, optional): restriction maps
        """ 
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1,num_nodes * self.d).t()
        x = self.lin(x)
        data_x = x

        #depending on norm_type D^-1 or D^-1/2
        D_inv, B_inv = normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, self.d, self.norm_type)

        if self.norm_type in ['sym_degree_norm', 'sym_block_norm']:
            # compute D^(-1/2) @ X
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse.FloatTensor(hyperedge_index, alpha, size=(num_nodes*self.d, num_edges*self.d))
        H_t = torch.sparse.FloatTensor(hyperedge_index.flip([0]), alpha, size=(num_edges*self.d, num_nodes*self.d))

        #this is because spdiags does not support gpu
        B_inv =  sparse_diagonal(B_inv, shape = (num_edges*self.d, num_edges*self.d))
        D_inv = sparse_diagonal(D_inv, shape = (num_nodes*self.d, num_nodes*self.d))

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(B_inv.indices(), B_inv.values(), H_t.indices(), H_t.values(), B_inv.shape[0],B_inv.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(H.indices(), H.values(), minus_L[0], minus_L[1], H.shape[0],H.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(D_inv.indices(), D_inv.values(), minus_L[0], minus_L[1], D_inv.shape[0],D_inv.shape[1], H_t.shape[1])
        minus_L = torch.sparse_coo_tensor(minus_L[0], minus_L[1], size=(num_nodes*self.d, num_nodes*self.d)).to(self.device)

        #negate the diagonal blocks and add eye matrix
        if self.I_mask is None or self.clear_flag: #prepare these in advance
            I_mask_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            I_mask_indices = generate_indices_general(I_mask_indices, self.d)
            I_mask_values = torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse_coo_tensor(I_mask_indices, I_mask_values).to(self.device)
            self.Id = sparse_diagonal(torch.ones(num_nodes*self.d), shape = (num_nodes*self.d, num_nodes * self.d)).to(self.device)

        minus_L = minus_L.coalesce()
        #this help us changing the sign of the elements in the block diagonal
        #with an efficient lower=memory mask 
        minus_L = torch.sparse_coo_tensor(minus_L.indices(), minus_L.values(), minus_L.size())
        minus_L = minus_L - 2 * minus_L.mul(self.I_mask)
        minus_L = self.Id + minus_L

        minus_L = minus_L.coalesce()
        out = torch_sparse.spmm(minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x)
        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + data_x
        return out
    
class SheafHyperGNN(nn.Module):
    """
        This is a Hypergraph Sheaf Model with 
        the dxd blocks in H_BIG associated to each pair (node, hyperedge)
        being **diagonal**
    """
    def __init__(self, num_features, num_targets, args, sheaf_type='SheafHyperGNNDiag'):
        super(SheafHyperGNN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = num_features
        self.num_targets = num_targets
        self.MLP_hidden = args.MLP_hidden 
        self.d = args.stalk_dim  # dimension of the stalks
        self.init_hedge = args.init_hedge # how to initialise hyperedge attributes: avg or rand
        self.norm_type = args.sheaf_normtype #type of laplacian normalisation degree_norm or block_norm
        self.act = args.sheaf_act # type of nonlinearity used when predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj # multiply with (I x W_1) to the left
        self.args = args
        self.norm = args.AllSet_input_norm
        self.dynamic_sheaf = args.dynamic_sheaf # if True, the sheaf changes from one layer to another
        self.residual = args.residual_sheaf

        self.clear_flag = True if args.task_type == 'hg_cls' else False # 如果是Hypergraph Classification Task，由于要考虑多图情况，需要每轮清空部分参数
           
        self.hyperedge_attr = None
        
        if args.device in [0, 1]:
            self.device = torch.device('cuda:'+str(args.device)
                              if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(args.device)
        
        self.lin = MLP(in_channels=self.num_features, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=False)
        
        # define the model and sheaf generator according to the type of sheaf wanted
        # The diuffusion does not change, however tha implementation for diag and ortho is more efficient
        if sheaf_type == 'SheafHyperGNNDiag':
            ModelSheaf, SheafConv = SheafBuilderDiag, HyperDiffusionDiagSheafConv
        else:
            raise NotImplementedError
        
        self.convs = nn.ModuleList()
        # Sheaf Diffusion layers
        self.convs.append(SheafConv(self.MLP_hidden, self.MLP_hidden, d=self.d, device=self.device, 
                                        norm_type=self.norm_type, left_proj=self.left_proj, 
                                        norm=self.norm, residual=self.residual, clear_flag=self.clear_flag))
                                        
        # Model to generate the reduction maps
        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(ModelSheaf(args))

        for _ in range(self.num_layers-1):
            # Sheaf Diffusion layers
            self.convs.append(SheafConv(self.MLP_hidden, self.MLP_hidden, d=self.d, device=self.device, 
                                        norm_type=self.norm_type, left_proj=self.left_proj, 
                                        norm=self.norm, residual=self.residual, clear_flag=self.clear_flag))
            # Model to generate the reduction maps if the sheaf changes from one layer to another
            if self.dynamic_sheaf:
                self.sheaf_builder.append(ModelSheaf(args))
                
        self.lin2 = nn.Linear(self.MLP_hidden*self.d, self.num_targets, bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

        self.lin.reset_parameters()
        self.lin2.reset_parameters()    

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        #initialize hyperedge attributes either random or as the average of the nodes
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):
        x = data.x
        edge_index = data.hyperedge_index
        num_nodes = data.x.shape[0] #data.edge_index[0].max().item() + 1
        num_edges = data.hyperedge_index[1].max().item() + 1

        #if we are at the first epoch, initialise the attribute, otherwise use the previous ones
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=x, hyperedge_index=edge_index)

        # expand the input N x num_features -> Nd x num_features such that we can apply the propagation
        x = self.lin(x)
        x = x.view((x.shape[0]*self.d, self.MLP_hidden)) # (N * d) x num_features

        hyperedge_attr = self.lin(self.hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.MLP_hidden))

        for i, conv in enumerate(self.convs[:-1]):
            # infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
            if i == 0 or self.dynamic_sheaf:
                h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[i](x, hyperedge_attr, edge_index)
            # Sheaf Laplacian Diffusion
            x = F.elu(conv(x, hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges))
            x = F.dropout(x, p=self.dropout, training=self.training)

        #infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
        if len(self.convs) == 1 or self.dynamic_sheaf:
            h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[-1](x, hyperedge_attr, edge_index) 
        # Sheaf Laplacian Diffusion
        x = self.convs[-1](x,  hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
        x = x.view(num_nodes, -1) # Nd x out_channels -> Nx(d*out_channels)

        x = self.lin2(x) # Nx(d*out_channels)-> N x num_classes 
        
        return x, None