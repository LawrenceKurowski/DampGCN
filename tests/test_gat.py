import torch
import argparse
from src.pyg_dataset import Dataset, Dpr2Pyg
from src.gat import GAT
from dataset import Dataset
from src.attacked_data import PrePtbDataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='perturbation rate')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

"""
--lbda 0 0 --train_lbda 0 is the base GAT base case
"""


parser.add_argument('--lbda', '--names-list1', nargs='+', default=[]) # pick lbda (constant)
parser.add_argument('--train_lbda', type=float, default=0)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
np.random.seed(args.seed)

torch.manual_seed(args.seed)
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LBDA = [float(args.lbda[0]),float(args.lbda[1])]

if LBDA[0]==0 and LBDA[1]==0 and args.train_lbda==0:
    damping = 0
else:
    damping = 1

# use data splist provided by prognn
data = Dataset(root='./data/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test



gat = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda,damping=damping,
      heads=8,
      device=device)
gat = gat.to(device)







# test on clean graph
print('==================')
print('=== train on clean graph ===')

pyg_data = Dpr2Pyg(data)
print(type(pyg_data[0]))
exit()
gat.fit(pyg_data, verbose=True) # train with earlystopping
gat.test()

# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
print('==================')
print('=== load graph perturbed by Zugner metattack (under seed 15) ===')
perturbed_data = PrePtbDataset(root='/tmp/',
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)
perturbed_adj = perturbed_data.adj
pyg_data.update_edge_index(perturbed_adj) # inplace operation
gat.fit(pyg_data, verbose=True) # train with earlystopping
gat.test()




