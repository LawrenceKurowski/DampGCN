"""
Following DeepRobust repo
https://github.com/DSE-MSU/DeepRobust
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import *
import utils
from copy import deepcopy
from sklearn.metrics import f1_score

class GClayer(Module):
    """
    Vanilla Kipf
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,bias=True):
        super(GClayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support=input@self.weight#adj@input
        output = adj@support#support@self.weight

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class Damping_layer(Module):
    """
    Adds damping term(s) to the standard KIPF model
    train_lbda == 1 : lambda is a trainable diagonal matrix initialized for each damping layer at 0
    train_lbda == 0, lbda = [a,b] where a,b !=0 : lambda is constant a, b for 2 layers
    train_lbda ==1, lbda = [a,b] where a,b != 0 : lambda is trainable, initialized at constant values a,b placed on diagonal
    """
    def __init__(self,n_nodes,n_features,train_lbda,lbda):
        super(Damping_layer, self).__init__()
        self.train_lbda =train_lbda
        self.init_lbda = float(lbda)
        self.lbda=lbda
        if self.train_lbda == 1:
            lbda_val = Parameter(torch.FloatTensor(n_nodes))
        else:
            lbda_val = lbda
        self.lbda = lbda_val
        self.reset_parameters()

    def reset_parameters(self):
        if self.train_lbda == 1:
            self.lbda.data.fill_(self.init_lbda)

    def forward(self, input_term, kipf_term,spar):
        if self.train_lbda == 1:
            
            lbda = torch.diag(self.lbda)
            output = -(input_term.transpose(0,1)@lbda).T+kipf_term
        else:
            lbda = float(self.lbda)
            output = kipf_term-lbda*input_term
        self.lbda_out = lbda
        return output
    
    
    
class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.
    add PS model
    
    "damping" controls which model we select
        damping==0 means standard GCN
        damping==1 means standard GCN with damping layers followed by FC layers added
    """

    def __init__(self, nfeat, nhid, nclass, n_nodes,lbdas,train_lbda,damping,dropout=0.5,lr=1e-3, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

#         self.nOfhid = nhid
#         self.nOffeat = nfeat
#         self.nOfclass = nclass
        
        self.damping = damping
        
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        
        self.lbda = lbdas
        
        if self.damping==0:
            self.gc1 = GClayer(nfeat, nhid, bias=with_bias)
            self.gc2 = GClayer(nhid, nclass, bias=with_bias)
        else:
            self.gc1 = GClayer(nfeat, nfeat,bias=with_bias)
            self.gc2 = GClayer(nhid, nhid,bias=with_bias)
            self.damp1 = Damping_layer(n_nodes,nfeat,train_lbda,lbdas[0])
            self.damp2 = Damping_layer(n_nodes,nhid,train_lbda,lbdas[1])        
            self.fc1 = torch.nn.Linear(nfeat, nhid)
            self.fc2 = torch.nn.Linear(nhid, nclass)
        
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x,adj):
        
        if self.damping==0:
            
            if self.with_relu:
                x = F.relu(self.gc1(x, adj))
            else:
                x = self.gc1(x, adj)

            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            return F.log_softmax(x, dim=1)

        else:
            output = self.gc1(x, adj)
            output = self.damp1(x,output,spar=1)
            output = self.fc1(output)
            output = F.relu(output)
            output = F.dropout(output, self.dropout, training=self.training)
            output_=self.gc2(output, adj)
            output = self.damp2(output,output_,spar=0)
            output = self.fc2(output)

            return F.log_softmax(output, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        
        if self.damping==1:
            self.damp1.reset_parameters()
            self.damp2.reset_parameters()
            self.fc1.reset_parameters()
            self.fc2.reset_parameters()

#     def nclass(self):
#         return int(self.nOfclass)

#     def nfeat(self):
#         return self.nOffeat

#     def nhid(self):
#         return self.nOfhid
    
    
    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=250, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            train_accu,train_loss=self._train_without_val(labels, idx_train, train_iters, verbose)
            return train_accu,train_loss
        else:
            if patience < train_iters:
                train_accu,train_loss,val_accu,val_loss=self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                train_accu,train_loss,val_accu,val_loss=self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)
            return train_accu,train_loss,val_accu,val_loss

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        train_accu=[]
        train_loss=[]
        val_accu =[]
        val_loss =[]
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            acc_train = accuracy(output[idx_train],labels[idx_train])
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print('Epoch {}, training accuracy: {}'.format(i, acc_train.item()))
            
            train_accu.append(acc_train)
            train_loss.append(loss_train)
        
        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output
        
        return train_accu,train_loss

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        train_accu =[]
        train_loss =[]
        val_accu =[]
        val_loss =[]
        if verbose:
            print('=== training gcn model ===')
        if self.damping==0:
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam([
                {"params": self.gc1.parameters(), "lr": 1e-3},
                {"params": self.damp1.parameters(), "lr": 1e-2},
                {"params": self.damp2.parameters(), "lr": 1e-2},
                {"params": self.gc2.parameters(),"lr": 1e-3},
                {"params": self.fc1.parameters(),"lr": 1e-2},
                {"params": self.fc2.parameters(),"lr": 1e-2}
            ],lr=self.lr,weight_decay=self.weight_decay)
            
        # uncomment this and comment out the above to use a simple optimizer that's identical for all layers
#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                print('Epoch {}, validation acc: {}'.format(i, acc_val.item()))
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
            
            accu=accuracy(output[idx_train],labels[idx_train])
            
            train_accu.append(accu)
            train_loss.append(loss_train)
            val_acc = accuracy(output[idx_val],labels[idx_val])
            val_accu.append(val_acc)
            val_loss.append(F.nll_loss(output[idx_val], labels[idx_val]))
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        
        return train_accu,train_loss,val_accu,val_loss
        

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        train_accu=[]
        train_loss=[]
        val_accu =[]
        val_loss =[]
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            
                       
            train_accu.append(accuracy(output[idx_train],labels[idx_train]))
            train_loss.append(loss_train)
            val_acc = accuracy(output[idx_val],labels[idx_val])
            val_accu.append(accuracy(output[idx_val],labels[idx_val]))
            val_loss.append(F.nll_loss(output[idx_val], labels[idx_val]))
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                print("validation accuracy at iter= ", i, " is = ", val_acc)
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
            
        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        
        return train_accu,train_loss,val_accu,val_loss

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized data

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


        
