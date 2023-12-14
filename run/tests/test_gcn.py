"""
Following DeepRobust repo
https://github.com/DSE-MSU/DeepRobust
"""
import torch
import numpy as np
import torch.nn.functional as F
from gcn import GCN
#from deeprobust.graph.defense import Alfred
from DampGCN.src.utils import *
from dataset import Dataset
import argparse
import random

# from pyg_dataset import Dataset, Dpr2Pyg
# from attacked_data import PrePtbDataset


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

"""
--lbda 0 0 --train_lbda 0 is the GCN base case
"""
parser.add_argument('--lbda', '--names-list1', nargs='+', default=[]) # pick lbda (constant)
parser.add_argument('--train_lbda', type=float, default=0)

args = parser.parse_args()



args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LBDA = [float(args.lbda[0]),float(args.lbda[1])]

if LBDA[0]==0 and LBDA[1]==0 and args.train_lbda==0:
    damping = 0
else:
    damping = 1

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

"""
Datasets: cora, polblogs, citeseer
"""
data = Dataset(root='./data/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print(type(features))

def main():
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)

    model = model.to(device)
    model.fit(features, adj, labels, idx_train,idx_val, train_iters=250, verbose=True)
    model.eval()
    acc = model.test(idx_test)
    print("Test accuracy: ",acc)

    
if __name__ == '__main__':
    main()
    
    
    