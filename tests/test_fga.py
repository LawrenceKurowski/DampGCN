"""
Following DeepRobust repo
https://github.com/DSE-MSU/DeepRobust
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from src.gcn import GCN
from src.fga import FGA
from src.utils import *
from dataset import Dataset
from tqdm import tqdm
import argparse
import os

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

LBDA = [float(args.lbda[0]),float(args.lbda[1])]

if LBDA[0]==0 and LBDA[1]==0 and args.train_lbda==0:
    damping = 0
else:
    damping = 1

print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    u = 0 # node to attack
    assert u in idx_unlabeled

    degrees = adj.sum(0).A1
    n_perturbations = int(degrees[u]) # How many perturbations to perform. Default: Degree of the node

    model.attack(features, adj, labels, idx_train, target_node, n_perturbations)

    print('=== testing GCN on original(clean) graph ===')
    test(adj, features, target_node)

    print('=== testing GCN on perturbed graph ===')
    test(model.modified_adj, features, target_node)

def test(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,            
            n_nodes = features.shape[0],
            lbdas = LBDA, train_lbda = args.train_lbda,damping=damping)

    if args.cuda:
        gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train)

    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    print('probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA, train_lbda = args.train_lbda,damping=damping)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

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

def multi_test_poison(target_nodes,target_model):
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes()
    num = len(node_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(target_nodes):
        n_perturbations = int(degrees[target_node])
        model = FGA(target_model,nnodes=adj.shape[0], device=device)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
        modified_adj = model.modified_adj
        acc = single_test(modified_adj, features, target_node)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt/num))
    return cnt/num

def single_test(adj, features, target_node, gcn=None):
    if gcn is None:
        # test on GCN (poisoning attack)
        gcn = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA, train_lbda = args.train_lbda,damping=damping)

        gcn = gcn.to(device)

        gcn.fit(features, adj, labels, idx_train, idx_val, patience=200)
        gcn.eval()
        output = gcn.predict()
    else:
        # test on GCN (evasion attack)
        output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    class_margin = classification_margin(output[target_node], labels[target_node])
    return acc_test.item(),class_margin

def multi_test_evasion(target_nodes,target_model):
    # test on 40 nodes on evasion attack
    # target_gcn = GCN(nfeat=features.shape[1],
    #           nhid=16,
    #           nclass=labels.max().item() + 1,
    #           dropout=0.5, device=device)

    # target_gcn = target_gcn.to(device)
    # target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)

    target_model = target_model
    cnt = 0
    degrees = adj.sum(0).A1
#     node_list = select_nodes(target_gcn)
    num = len(target_nodes)
    class_margins =[]
    print('=== [Evasion] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(target_nodes):
        n_perturbations = int(degrees[target_node])
        model = FGA(target_model, nnodes=adj.shape[0], device=device)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
        modified_adj = model.modified_adj

        acc,class_margin = single_test(modified_adj, features, target_node, gcn=target_model)
        if acc == 0:
            cnt += 1
        class_margins.append(class_margin)
    print('misclassification rate : %s' % (cnt/num))
    
    return cnt/num, class_margins
# def multi_test_evasion(target_nodes,target_model):



#     cnt = 0
#     degrees = adj.sum(0).A1
#     node_list = select_nodes(target_model)
#     num = len(node_list)

#     print('=== [Evasion] Attacking %s nodes respectively ===' % num)
#     for target_node in tqdm(target_nodes):
#         n_perturbations = int(degrees[target_node])
#         model = FGA(target_model, nnodes=adj.shape[0], device=device)
#         model = model.to(device)
#         record=[]
#         i=0
#         for ptbs in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
#             while i==0:
#                 model.attack(features, adj, labels, idx_train, target_node, ptbs)
#                 modified_adj = model.modified_adj

#                 acc = single_test(modified_adj, features, target_node, gcn=target_model)
#                 if acc == 0:
#                     cnt += 1
#                     i = 1
                
#                 record.append(i)
#         print('target node : ', target_node,'record : ', record)
#     print('misclassification rate : %s' % (cnt/num))
    
#     return cnt/num

if __name__ == '__main__':
    record = []
    class_margin_record=[]
    for i in range(5):
        print("Running FGA...")

        np.random.seed(42*i)
        torch.manual_seed(42*i)


        # __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        # data_loc = os.path.join(__location__, name=args.dataset)
        data = Dataset(root='./data/', name=args.dataset)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        
        idx_unlabeled = np.union1d(idx_val, idx_test)


        surrogate = GCN(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout,            
                        n_nodes = features.shape[0],
                        lbdas = LBDA,train_lbda=args.train_lbda,damping=damping)

        surrogate = surrogate.to(device)
        train_accu,train_loss,val_accu,val_loss = surrogate.fit(features, adj, labels, idx_train, idx_val, patience=300)



        target_nodes = select_nodes(surrogate)
        model = FGA(surrogate, nnodes=adj.shape[0], device=device)
        model = model.to(device)

#         miss_rate_poison=multi_test_poison(target_nodes,target_model=surrogate)
        miss_rate_evasion,class_margin=multi_test_evasion(target_nodes,target_model=surrogate)

#         print("Mean POISON missclassification rate over ", str(runs)," runs is ", np.mean(record_rates_poison))
        print("EVASION missclassification is ", miss_rate_evasion)
        print("EVASION class margin is ", class_margin)
#         df2 = pd.DataFrame(record_rates_evasion, columns=["misscalssification_rates_evasion"])
        record.append(miss_rate_evasion)
        class_margin_record.append(class_margin)
    print("Evasion FGA misclassification rates record: ",record)
    print("Evasion FGA class margins record: ",class_margin_record)

