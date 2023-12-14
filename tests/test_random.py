import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from gcn import GCN
from random_attack import Random
from utils import *
from dataset import Dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--lbda', '--names-list1', nargs='+', default=[]) # pick lbda (constant)
parser.add_argument('--train_lbda', type=float, default=0)
parser.add_argument('--attack_feat', type=int, default=0)
parser.add_argument('--attack_struc', type=int, default=0)

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


def test(adj,feat,modified_adj,modified_feat):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=feat.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)

    gcn = gcn.to(device)

    optimizer = optim.Adam(gcn.parameters(),
                           lr=0.01, weight_decay=5e-4)
    gcn.fit(feat, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    
    
    output = gcn.forward(modified_feat,modified_adj)
    
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()




if __name__ == '__main__':
    
    


    average_results = []
    for eps in [1]:#,0.09,.12,.15,.18]:
        results = []
        for i in range(1):#5):
            seed = i*42

            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)
            
            data = Dataset(root='./data/', name=args.dataset)
            adj, features, labels = data.adj, data.features, data.labels
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
            idx_unlabeled = np.union1d(idx_val, idx_test)
            
#             adj = adj.to(device)
#             features = features.to(device)
#             labels = labels.to(device)
            
            model = Random(attack_features=args.attack_feat,attack_structure=args.attack_struc)
            # Setup Attack Model
            n_perturbations = int(eps * (adj.sum()//2))
#             adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)
            
            
            model.attack(adj, features,n_perturbations)
#             modified_adj = model.modified_adj#adj
#             modified_feat = features
            if args.attack_feat==1:
                print("attack features")
                modified_feat = model.modified_feat
                modified_adj = adj
            elif args.attack_struc==1:
                print("attack structure")
                modified_feat = features
                modified_adj = model.modified_adj
            else:
                modified_feat = features
                modified_adj = adj
            

            adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)
            
            modified_adj = normalize_adj(modified_adj)
            modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
            modified_adj = modified_adj.to(device)
            
#             modified_feat = normalize_adj(modified_feat)
            modified_feat = sparse_mx_to_torch_sparse_tensor(modified_feat)
            modified_feat = modified_feat.to(device)

#             print(features)

#             print(print(torch.sparse.sum(features==modified_feat)))
#             exit()
            acc_att = test(adj,features,modified_adj,modified_feat)

            results.append(acc_att)
#         print(results)
        mean_val=np.mean(results)
        average_results.append(mean_val)
        print('for eps = ',eps,' mean is = ',str(mean_val))

