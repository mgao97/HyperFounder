import time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import os
from dhg import Hypergraph
from dhg.data import *
from models import *
from layers import *
from preprocessing import *
from utils import micro_f1, macro_f1

from dhg.random import set_seed
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from torch_geometric.data import Data
from torch_sparse import coalesce
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import os.path as osp
import scipy.sparse as sp
import torch_sparse
from tqdm import tqdm

def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E)
    #     Below we can add different model, such as HyperGCN and so on
    # print(model)
    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   micro F1 Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 5]
            print(f'   macro F1 Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3], best_result[:, 4], best_result[:, 5]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])
    micro_f1_test = micro_f1(data.y[split_idx['test']], out[split_idx['test']])
    macro_f1_test = macro_f1(data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, micro_f1_test, macro_f1_test, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def data_pre(hyperedges, features, labels, n_x, train_percent):
        edge_idx = num_nodes
        node_list = []
        edge_list = []
        for he in hyperedges.keys():
            cur_he = hyperedges[he]
            cur_size = len(cur_he)

            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size

            edge_idx += 1

        edge_index = np.array([node_list + edge_list,
                            edge_list + node_list], dtype = int)
        edge_index = torch.LongTensor(edge_index)

        data = Data(x = features, edge_index = edge_index, labels = labels)

        total_num_node_id_he_id = edge_index.max() + 1
        data.edge_index, data.edge_attr = coalesce(data.edge_index, 
                    None, 
                    total_num_node_id_he_id, 
                    total_num_node_id_he_id)

        num_class = len(np.unique(labels.numpy()))
        # val_lb = int(n_x * train_percent)
        # train_percent_scalar = train_percent.item()
        # percls_trn = int(round(train_percent_scalar * n_x / num_class))

        # percls_trn = int(round(train_percent * n_x / num_class))
        # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
        data.n_x = torch.LongTensor(n_x)
        # add parameters to attribute

        data.train_percent = train_percent
        data.num_hyperedges = torch.LongTensor(hyperedges)
        data.y = labels

        return data


if __name__ == '__main__':

    args = parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='highschool')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllDeepSets')
    parser.add_argument('--epochs', default=200, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=64,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()

    # load data
    with open('contact-high-school/hyperedges-contact-high-school.txt', 'r') as file:
        edge_list = [tuple(map(lambda x: int(x) - 1, line.strip().split(','))) for line in file]

    # print(edge_list)
    num_vertices = len(set(item for sublist in edge_list for item in sublist))

    # print(len(edge_list), num_vertices)
    G = Hypergraph(num_vertices, edge_list)
    print(G)

    labels = []
    with open ('contact-high-school/node-labels-contact-high-school.txt', 'r') as file:
        for line in file:
            labels = [int(line) - 1 for line in file]
    labels = [8] + labels
    # print(labels)
    labels = torch.LongTensor(labels)



    # # 设置随机种子，以确保结果可复现
    random_seed = 42

    v_deg= G.D_v
    X = v_deg.to_dense()/torch.max(v_deg.to_dense())
    features = torch.FloatTensor(X)
    # labels = data["labels"]
    print('X dim:', X.shape)
    print('labels:', len(torch.unique(labels)))

    num_features = torch.tensor(X.shape[0])
    num_classes = len(labels.unique())
    num_nodes = torch.tensor(X.shape[0])
    n_x = X
    train_percent = 0.5


    he_list = [i for i in range(len(G.e[0]))]
    he_value = [list(x) for x in G.e[0]]
    hyperedges = dict(zip(he_list, he_value))


    data = data_pre(hyperedges, features, labels, n_x, train_percent)


    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)
        data = norm_contruction(data, option=args.normtype)

    #     Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx) 
        
    # # Part 2: Load model
    net = parse_method(args, data)
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    net, data = net.to(device), data.to(device)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)

    num_params = count_parameters(net)


    # # Part 3: Main. Training + Evaluation


    logger = Logger(args.runs, args)

    criterion = nn.NLLLoss()
    eval_func = eval_acc

    net.train()
    # print('MODEL:', net)

    ### Training loop ###
    runtime_list = []
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        net.reset_parameters()
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=net.reg_params, weight_decay=0.01),
                dict(params=net.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    #     This is for HNHN only
    #     if args.method == 'HNHN':
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
        best_val = float('-inf')
        for epoch in range(args.epochs):
            #         Training part
            net.train()
            optimizer.zero_grad()
            out = net(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
            result = evaluate(net, data, split_idx, eval_func)
            logger.add_result(run, result[:3])

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                        f'Train Loss: {loss:.4f}, '
                        f'Valid Loss: {result[4]:.4f}, '
                        f'Test  Loss: {result[5]:.4f}, '
                        f'Test micro F1: {result[6]:.4f},'
                        f'Test macro F1: {result[7]:.4f},'
                        f'Train Acc: {100 * result[0]:.2f}%, '
                        f'Valid Acc: {100 * result[1]:.2f}%, '
                        f'Test  Acc: {100 * result[2]:.2f}%')

        end_time = time.time()
        runtime_list.append(end_time - start_time)

        # logger.print_statistics(run)

    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val, best_test, best_microf1, best_macrof1 = logger.print_statistics()
    res_root = 'hyperparameter_tunning'
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}'
        cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
        cur_line += f',{best_microf1.mean():.3f} ± {best_microf1.std():.3f}'
        cur_line += f',{best_macrof1.mean():.3f} ± {best_macrof1.std():.3f}'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s' 
        cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    print('All done! Exit python code')
    quit()

