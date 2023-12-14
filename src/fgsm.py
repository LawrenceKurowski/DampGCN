import random
import numpy as np
from src.base_attack import BaseAttack
import torch
import copy


class fgsm(BaseAttack):
    """ Randomly adding edges to the input graph

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import Random
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = Random()
    >>> model.attack(adj, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None,nnodes=None, attack_structure=False, attack_features=False, device='cpu'):
        super(fgsm, self).__init__(model,nnodes,attack_structure, attack_features, device=device)

    def attack(self, adj, feat, target_node,grad_sign,attack_structure,attack_features):
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        """Generate attacks on the input graph.

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of edge removals/additions.
        type: str
            perturbation type. Could be 'add', 'remove' or 'flip'.

        Returns
        -------
        None.

        """
        self.grad_sign = grad_sign
        self.ori_adj = adj
        
        if self.attack_structure==1:
            mod_adj = self.perturb_adj(adj, target_node)
            self.mod_adj = mod_adj
            self.mod_feat = feat
            return mod_adj
        
        if self.attack_features==1:
            mod_feat = self.perturb_feat(feat, target_node)
            self.modfeat = modified_feat
            self.mod_adj = adj
            return mod_feat

    def perturb_feat(self, feat, target_node):
        mod_feat = torch.clone(feat)
        mod_feat[target_node] = feat[target_node] + self.grad_sign[target_node]
        mod_feat[target_node] = torch.clamp(mod_feat[target_node],0,1)
        return mod_feat
    
    def perturb_adj(self, adj, target_node):
        mod_adj = torch.tensor(adj.todense())
        mod_adj[target_node] = mod_adj[target_node] + self.grad_sign[target_node]
        mod_adj[target_node] = torch.clamp(mod_adj[target_node],0,1)

        for node in target_node:
            mod_adj[:,node] = mod_adj[node].T
        
        self.check_adj(mod_adj)
        return mod_adj

           
