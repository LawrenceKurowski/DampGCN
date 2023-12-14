import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from gcn import GCN
from topology_attack import PGDAttack
from utils import *
from dataset import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')



"""
--lbda 0 0 --train_lbda 0 is the GCN base case
"""
parser.add_argument('--lbda', '--names-list1', nargs='+', default=[]) # pick lbda (constant)
parser.add_argument('--train_lbda', type=float, default=0)




args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LBDA = [float(args.lbda[0]),float(args.lbda[1])]

if LBDA[0]==0 and LBDA[1]==0 and args.train_lbda==0:
    damping = 0
else:
    damping = 1


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='./data/', name=args.dataset, setting='gcn')
adj, features, labels = data.adj, data.features, data.labels
# features = normalize_feature(features)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Victim Model
victim_model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)

victim_model = victim_model.to(device)
victim_model.fit(features, adj, labels, idx_train)

# Setup Attack Model

model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)

def test(adj, gcn=None):
    ''' test on GCN '''

    if gcn is None:
        # adj = normalize_adj_tensor(adj)
        gcn = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train) # train without model picking
        gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with validation model picking
        gcn.eval()
        output = gcn.predict().cpu()
    else:
        gcn.eval()
        output = gcn.predict().cpu()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    target_gcn = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)

    target_gcn = target_gcn.to(device)
    target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)

    print('=== testing GCN on clean graph ===')
    test(adj, target_gcn)

    print('=== testing GCN on Evasion attack ===')
    model.attack(features, adj, labels, idx_train, perturbations, epochs=args.epochs)
    modified_adj = model.modified_adj
    test(modified_adj, target_gcn)

    # modified_features = model.modified_features
    print('=== testing GCN on Poisoning attack ===')
    test(modified_adj)

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')

if __name__ == '__main__':
    main()

