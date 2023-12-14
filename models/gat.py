"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import utils
from copy import deepcopy
from torch_geometric.nn import GATConv
import numpy as np

from torch_geometric.utils import to_scipy_sparse_matrix

from utils import * 

class Damping_layer(Module):
    """
    Adds damping term(s) to the standard KIPF model
    train_lbda == 1 : lambda is a trainable diagonal matrix initialized for each damping layer at 0
    train_lbda == 0, lbda = [a,b] where a,b !=0 : lambda is constant a, b for 2 layers
    train_lbda ==1, lbda = [a,b] where a,b != 0 : lambda is trainable, initialized at constant values a,b placed on diagonal
    """
    def __init__(self,n_nodes,n_features,train_lbda,lbda,heads):
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
        
        self.heads=heads
        
    def reset_parameters(self):
        if self.train_lbda == 1:
            self.lbda.data.fill_(self.init_lbda)

    def forward(self, input_term, other_term):
        if self.train_lbda == 1:
            lbda = torch.diag(self.lbda)
            
#             print(input_term.shape)
#             print(other_term.shape)
#             print(lbda.shape)
#             print(((input_term.transpose(0,1)@lbda).T).shape)
#             exit()
            output = -((input_term.transpose(0,1)@lbda).T)+other_term
        else:
            lbda = float(self.lbda)
            output = other_term-lbda*input_term
        self.lbda_out = lbda
#         print(torch.sum(lbda))print(t
#         exit()
        return output

class GAT(nn.Module):
    """ 2 Layer Graph Attention Network based on pytorch geometric.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    heads: int
        number of attention heads
    output_heads: int
        number of attention output heads
    dropout : float
        dropout rate for GAT
    lr : float
        learning rate for GAT
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in GAT weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GAT.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GAT
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gat = gat.to('cpu')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> gat.fit(pyg_data, patience=100, verbose=True) # train with earlystopping
    """

    def __init__(self, nfeat, nhid, nclass, n_nodes, lbdas,train_lbda,damping,heads=8, output_heads=1,dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = nhid
        self.damping = damping
        self.lbda = lbdas
        
        if self.damping==0:
            self.conv1 = GATConv(
                nfeat,
                nhid,
                heads=heads,
                dropout=dropout,
                bias=with_bias)

            self.conv2 = GATConv(  
                nhid * heads,
                nclass,
                heads=output_heads,
                concat=False,
                dropout=dropout,
                bias=with_bias)
        else:
            self.conv1 = GATConv(
                nfeat,
                nhid,
                heads=heads,
                dropout=dropout,
                bias=with_bias)

            self.conv2 = GATConv(  
                nhid*heads,
                nclass,
                heads=output_heads,
                concat=False,
                dropout=dropout,
                bias=with_bias)

            self.damp1 = Damping_layer(n_nodes,nfeat,train_lbda,lbdas[0],heads)
            self.damp2 = Damping_layer(n_nodes,nhid,train_lbda,lbdas[1],heads)  

            self.fc1 = torch.nn.Linear(nfeat, nhid)
            self.fc2 = torch.nn.Linear(nhid*heads, nclass)
        
        
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
    
        self.heads=heads
    
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        labels = data.y
        
        x, adj, _ = utils.to_tensor(x, to_scipy_sparse_matrix(edge_index), labels, device=self.device)
        if self.damping==0:
#             x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
#             x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
            
        else:
            x, edge_index = data.x, data.edge_index
#             x = F.dropout(x, p=self.dropout, training=self.training)
            
            output = self.conv1(x, edge_index)
            side_branch = torch.cat(self.heads*[self.fc1(x)],dim=1)
            x = self.damp1(side_branch,output)

            x = F.elu(x)#F.dropout(F.elu(x,self.dropout))
            output = self.conv2(x, edge_index)
            side_branch = self.fc2(x)
            
            x = self.damp2(side_branch,output)
            
            return F.log_softmax(x, dim=1)
    
    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, pyg_data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        """Train the GAT model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """


        if initialize:
            self.initialize()

        self.data = pyg_data#[0].to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)
#             print(type(output[train_mask]))
#             print(type(labels[train_mask]))
#             exit()
            loss_train = F.nll_loss(output[train_mask], torch.tensor(labels[train_mask],dtype=torch.long))
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], torch.tensor(labels[val_mask],dtype=torch.long))

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
        self.load_state_dict(weights)

    def test(self):
        """Evaluate GAT performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GAT
        """

        self.eval()
        return self.forward(self.data)



if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    # from deeprobust.graph.defense import GAT
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    gat = GAT(nfeat=features.shape[1],
          nhid=8, heads=8,
          nclass=labels.max().item() + 1,
          dropout=0.5, device='cpu')
    gat = gat.to('cpu')
    pyg_data = Dpr2Pyg(data)
    gat.fit(pyg_data, verbose=True) # train with earlystopping
    gat.test()
    print(gat.predict())

