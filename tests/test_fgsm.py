import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from gcn import GCN
from fgsm import fgsm
from utils import *
from dataset import Dataset
import argparse
import random

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
parser.add_argument('--save', type=float, default=0)
parser.add_argument('--runs', type=int, default=5)

parser.add_argument('--if_top_bottom', type=int, default=1)
parser.add_argument('--number_of_random', type=int, default=20)

parser.add_argument('--attack_structure', type=int, default=0)
parser.add_argument('--attack_features', type=int, default=1)

parser.add_argument('--targeted', type=int, default=1)

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

data = Dataset(root='./data/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)


# n_perturbations =int(adj.shape[0]*args.ptb_rate)#int(args.ptb_rate * (adj.sum()//2))# int(features.shape[1]*args.ptb_rate)

def attack_node(feat, adj, labels, target_gcn, target_node, attack_structure, attack_features):
    
    target_gcn.eval()
    
    if attack_features==1:
        feat.requires_grad = True
        
        feat = torch.FloatTensor(np.array(feat.todense()))
        adj = torch.FloatTensor(adj.todense())
        
        output = target_gcn(feat, adj)
        loss_test = F.nll_loss(output[[target_node]], labels[[target_node]])
        grad = torch.autograd.grad(loss_test, feat,retain_graph=True, create_graph=False)[0]
        
    if attack_structure==1:
        adj.requires_grad=True
        output = target_gcn(feat, adjac)
        loss_test = F.nll_loss(output[[target_node]], labels[[target_node]])
        grad = torch.autograd.grad(loss_test, adjac,retain_graph=True, create_graph=False)[0]
    
    grad_sign = grad.sign()
    model.attack(adj,feat, target_node,grad_sign,attack_structure,attack_features)

    modified_adj = model.modified_adj
    modified_feat = model.modified_feat

    return modified_adj, modified_feat

 

def train(adj,feat,labels, model):
    
    gcn = model
    gcn = gcn.to(device)

    optimizer = optim.Adam(gcn.parameters(),
                           lr=0.01, weight_decay=5e-4)
    gcn.fit(feat, adj, labels, idx_train, idx_val,patience=200) 
    return gcn
    

def test(adj,feat,labels,model,target_node):
    output = model.forward(feat,adj)
    class_marg = classification_margin(output[target_node] == labels[target_node])
    print('class_marg = ',class_marg)
    exit()
#     probs = torch.exp(output[[target_node]])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test



if __name__ == '__main__':
    
    print("Running FGSM attack...")
    
    runs=args.runs
    miss_rates=[]
    
    accu=[]

    eps = .1
    i=0
    
    np.random.seed(42*i)
    torch.manual_seed(42*i)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load original dataset (to get clean features and labels)
    data = Dataset(root='./data/', name=args.dataset)
    adj, feat, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    attack_structure = args.attack_structure
    attack_features = args.attack_features

    adj,feat,labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)
    
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,            
                n_nodes = features.shape[0],
                lbdas = LBDA, train_lbda = args.train_lbda,damping=damping)
    
    model = train(adj,feat,labels,model)
    model.eval()
    
    
    print('=== testing GCN on perturbed graph (node-wise) ===')
    
    target_nodes = select_nodes(model)

    for node in target_nodes:
    
        model = fgsm()
        model.attack(adj, feat, node,grad_sign,attack_structure,attack_features)
        
        adj = model.mod_adj
        feat = model.mod_feat
        output = model.forward(feat,adj)
#         exit()
#         print(output[[target_node]])
        exit()
#         class_marg = classification_margin(output[[target_node]] , labels[target_node])
#         print('class_marg = ',class_marg)
#         exit()

        acc = accuracy(output[idx_test], norm_label[idx_test])

        accu.append(acc)

    mean_acc = np.mean(accu)
    print('mean acc over 5 runs for indirected feature FGSM at ptb_rate = ',str(eps),' is  = ',str(mean_acc))

    #     df.to_csv('./results/FGSM_attack(untargted)_dataset='+str(args.dataset)+'_ptb_rate='+str(args.ptb_rate)+'_attack_feat='+str(args.attack_features)+'_attack_struc='+str(args.attack_structure)+'_lbda='+str(LBDA)+'_train_lbda='+str(args.train_lbda)+'.csv', index=False)
