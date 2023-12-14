"""
Following DeepRobust repo
https://github.com/DSE-MSU/DeepRobust
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from gcn import GCN
from nettack import Nettack
from utils import *
from dataset import Dataset
from torch_geometric.data import Data
import argparse
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--lbda', '--names-list1', nargs='+', default=[])
parser.add_argument('--train_lbda', type=float, default=0)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()


print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LBDA = [float(args.lbda[0]),float(args.lbda[1])]


if LBDA[0]==0 and LBDA[1]==0 and args.train_lbda==0:
    damping = 0
else:
    damping = 1


def main():
    degrees = adj.sum(0).A1
    # How many perturbations to perform. Default: Degree of the node
    n_perturbations = int(degrees[target_node])

    # direct attack
    model.attack(features, adj, labels, target_node, n_perturbations)
    # # indirect attack/ influencer attack
    # model.attack(features, adj, labels, target_node, n_perturbations, direct=False, n_influencers=5)
    modified_adj = model.modified_adj
    modified_features = model.modified_features
    print(model.structure_perturbations)
    print('=== testing GCN on original(clean) graph ===')
    test(adj, features, target_node)
    print('=== testing GCN on perturbed graph ===')
    test(modified_adj, modified_features, target_node)

def test(adj, features, target_node):
    ''' test on GCN '''
    model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,            
            n_nodes = features.shape[0],
            lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)
    model = model.to(device)

    model.fit(features, adj, labels, idx_train, idx_val, patience=30)
    

    model.eval()
    output = model.predict()

    probs = torch.exp(output[[target_node]])[0]

    
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Overall test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def select_nodes(target_model=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_model is None:
        target_model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda, damping=damping)
        target_model = target_model.to(device)
        target_model.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_model.eval()
    output = target_model.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other

def multi_test_poison():
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes()
    num = len(node_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
    accu_counter=[]
    margins=[]
    for target_node in tqdm(node_list):
#         if args.fgsm==0:
        n_perturbations = int(degrees[target_node])
        model = Nettack(surrogate, damping=damping,nclass=labels.max().item() + 1, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        acc,class_margin = single_test(modified_adj, modified_features, target_node,idx_test)
#         accu_counter.append(whole_accuracy)
        print("acc = ", acc)
        if acc == 0:
            cnt += 1
        margins.append(class_margin)
        print("class_margin = ",class_margin)
    print('misclassification rate : %s' % (cnt/num))

    return cnt/num , margins

def single_test(adj, features, target_node,idx_test, model):
    if model is None:
        model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)
        # test on GCN (poisoning attack)
        model = model.to(device)

        model.fit(features, adj, labels, idx_train, idx_val, patience=30)
        model.eval()
        output= model.predict()
    else:
        output = model.predict(features, adj)
    probs = torch.exp(output[[target_node]])
    
    correct_class = probs[0][labels[target_node]]
    best_wrong = probs[0][int(probs.argsort(1)[0][-1])]
    
    if probs[0][labels[target_node]]==probs[0][int(probs.argsort(1)[0][-1])]:
        best_wrong = probs[0][int(probs.argsort(1)[0][-2])]
#         print("best wrong = ", str(int(probs.argsort(1)[0][-2]))," , probability = ",str(best_wrong))
    else:
        best_wrong = probs[0][int(probs.argsort(1)[0][-1])]
#         print("best wrong = ", str(int(probs.argsort(1)[0][-1]))," , probability = ",str(best_wrong))
#     print("correct class = ", str(labels[target_node]), " , probability = ",str(correct_class))
#     exit()
    class_margin = correct_class - best_wrong
#     probs[0][int(probs.argsort(1)[0][-1])]
    #-probs[0][int(probs.argsort(1)[0][-2])]
# #     print(probs)
# #     print(probs.argsort(1)[0][0])
# #     print(probs[0][int(probs.argsort(1)[0][-1])])
# #     print(probs[0][int(probs.argsort(1)[0][-2])])
# #     print(probs[0][int(probs.argsort(1)[0][-3])])
# #     print()
#     print(float(class_margin))
#     exit()
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
#     print("accuracy at single_test = ",acc_test)
    return acc_test.item(),class_margin.item()

def multi_test_evasion(features,adj,labels, idx_train, idx_val,surrogate):
    # test on 40 nodes on evasion attack
    target_model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,            
            n_nodes = features.shape[0],
            lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)

    target_model = target_model.to(device)

    train_accu,train_loss,val_accu,val_loss=target_model.fit(features, adj, labels, idx_train, idx_val, patience=200)
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes()
    num = len(node_list)
    
    accu_counter = []
    margins=[]
    print('=== [Evasion] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(node_list):

        n_perturbations = int(degrees[target_node])
        model = Nettack(surrogate, damping=damping,nnodes=adj.shape[0], nclass = labels.max().item() + 1,attack_structure=True, attack_features=True, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features

        acc,class_margin = single_test(modified_adj, modified_features, target_node, idx_test,model=target_model)
        
        
#         print("acc = ", acc)
        if acc == 0:
            cnt += 1
#             print("count is ",cnt)
        margins.append(class_margin)
#         print("class_margin = ",margins)
#         print("running misclassification rate = ",cnt/num)
    print('misclassification rate : %s' % (cnt/num))
    
    return cnt/num, margins
        



if __name__ == '__main__':
    
    print("Running Nettack...")
    
    all_rates = []
    all_margins = []
    
    for i in range(5):
        
        np.random.seed(42*i)
        torch.manual_seed(42*i)


        data = Dataset(root='./data/', name=args.dataset)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_unlabeled = np.union1d(idx_val, idx_test)


        surrogate = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA, train_lbda = args.train_lbda,damping=damping)

        surrogate = surrogate.to(device)
        train_accu,train_loss,val_accu,val_loss = surrogate.fit(features, adj, labels, idx_train, idx_val, patience=200)

        # Setup Attack Model
    #     target_node = 0
    #     assert target_node in idx_unlabeled

#         model = Nettack(surrogate,nclass = labels.max().item() + 1, damping=damping,nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
#         model = model.to(device)
    #     miss_rate_poison=multi_test_poison()
    #     print("POISON missclassification rate is ", str(miss_rate_poison))

        miss_rate_evasion,margins =multi_test_evasion(features,adj,labels, idx_train, idx_val,surrogate)
    #     record_rates_evasion.append(miss_rate_evasion)
        print("EVASION missclassification rate is ", str(miss_rate_evasion))
        print("EVASION margins are ", str(margins))
        all_rates.append(miss_rate_evasion)
        all_margins.append(margins)
    print(all_rates)
    print(all_margins)
    print("average rates over 5 runs = ", np.mean(all_rates))
    print("average margins over 5 runs = ", np.mean(all_margins,1))
        
