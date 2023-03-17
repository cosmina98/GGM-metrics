#disable  
import torch
from evaluation.mol_structure import list_of_smiles_to_nx_graphs
from rdkit import RDLogger 
import os 
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir
import networkx as nx
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from torch_geometric.utils.convert import from_networkx
import torch 
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      


def get_data(name, path='data/smiles/',return_smiles=False):
    splits={}
    RDLogger.EnableLog('rdApp.*')
    RDLogger.DisableLog('rdApp.*')  
    split_names=['train_smiles','test_smiles', 'train1_pos_smiles','train1_neg_smiles','valid_smiles','train_tragets','test_targets', 'valid_targets' ]
    for i,split in enumerate(split_names):
        exact_path=path+'{}/{}.txt'.format(name, split)
        #from data.smiles.carcinogens import test_smiles
        current_list = []
        with open(exact_path) as my_file:
         for line in my_file:
            current_list.append(line.strip())
        splits[split]=current_list     
     
    train_graphs=list_of_smiles_to_nx_graphs(splits['train_smiles'])
    train_targets=string_targets_to_numeric(splits['train_tragets'])
    test_graphs=list_of_smiles_to_nx_graphs(splits['test_smiles'])
    test_targets=string_targets_to_numeric(splits['test_targets'])
    valid_graphs=list_of_smiles_to_nx_graphs(splits['valid_smiles'])
    valid_targets=string_targets_to_numeric(splits['valid_targets'])
    train1_pos_graphs =list(list_of_smiles_to_nx_graphs(splits['train1_pos_smiles']))
    train1_neg_graphs =list(list_of_smiles_to_nx_graphs(splits['train1_neg_smiles']))
    train1_graphs = train1_pos_graphs+ train1_neg_graphs
    train1_targets = np.array([1]*len(train1_pos_graphs) + [0]*len(train1_neg_graphs))
    graphs=[train_graphs,train_targets,test_graphs,test_targets,train1_graphs,train1_targets,valid_graphs, valid_targets]
    if return_smiles:
         return graphs,splits
    else:  return graphs
    
#mock function in place of generated 

def get_mock_data(name, path='data/smiles/',return_smiles=False):
    RDLogger.EnableLog('rdApp.*')
    RDLogger.DisableLog('rdApp.*') 
    splits={}
    split_names=[ 'train1_pos_smiles','train1_neg_smiles']
    for i,split in enumerate(split_names):
        exact_path=path+'{}/{}.txt'.format(name, split)
        #from data.smiles.carcinogens import test_smiles
        current_list = []
        with open(exact_path) as my_file:
         for line in my_file:
            current_list.append(line.strip())
        splits[split]=current_list     

    train1_pos_graphs =list(list_of_smiles_to_nx_graphs(splits['train1_pos_smiles']))
    train1_neg_graphs =list(list_of_smiles_to_nx_graphs(splits['train1_neg_smiles']))

    return train1_pos_graphs,train1_neg_graphs


def string_targets_to_numeric(targets):
    targets=[eval(i) for i in targets]
    return targets

def get_flat_predictions(preds):
    return torch.cat((torch.flatten(torch.stack(preds[:-1])) ,  preds[-1]), dim=0)




class NxGraphstoPyggraphs(BaseEstimator, TransformerMixin):
    
   
    def fit(self, X,y=None):
        return self
    def transform( X, y=None):
       list_of_pyg=[]
       X=deepcopy(X)
                   
       def without( d, keys):
            for k in keys:
                d.pop(k)
            return d
       for i , g in enumerate(X):
    
            s=from_networkx(g)
            s.x=s.attr
            keys=[s for s in s.keys if s not in ['x', 'edge_attr', 'edge_index']]

            s=without(s, keys)
            #print(s)
            """
            s.num_nodes=s.num_nodes
            s.num_node_features=s.x[1]
            s.num_edge_features=s.edge_attr[1]
            s.num_edges=(s.edge_index[1])/2
            """

            if y is  None:
                s.y=None
            else: 
               s.y=torch.tensor(int(y[i]),dtype=torch.long)
            list_of_pyg.append(s)
       return list_of_pyg

        
#transform = T.Compose([
   # T.RandomLinkSplit(num_val=0.05, num_test=0.2, is_undirected=True,
                     # split_labels=True, add_negative_train_samples=True)
#])

def get_pyg_graphs_from_all_nx_sets(train_graphs, train_targets,  test_graphs, test_targets ,\
    train1_graphs , train1_targets,valid_graphs, valid_targets,generated_graphs,generated_targets):
    train_pyg_graphs=   NxGraphstoPyggraphs.transform(train_graphs,train_targets)
    test_pyg_graphs =   NxGraphstoPyggraphs.transform(test_graphs,test_targets)
    train1_pyg_graphs=   NxGraphstoPyggraphs.transform(train1_graphs,train1_targets)
    valid_pyg_graphs=   NxGraphstoPyggraphs.transform(valid_graphs,valid_targets)
    generated_pyg_graphs=   NxGraphstoPyggraphs.transform(generated_graphs,generated_targets)
    all_pyg=train_pyg_graphs+ generated_pyg_graphs
    return train_pyg_graphs,test_pyg_graphs,train1_pyg_graphs,valid_pyg_graphs,generated_pyg_graphs,all_pyg

def get_pyg_graphs_from_nx(X,y):
    pyg_graphs=   NxGraphstoPyggraphs.transform(X,y)
    return pyg_graphs
    
def get_pyg_loader_from_nx(X,y,batch_size):
    pyg_graphs=   get_pyg_graphs_from_nx(X,y)
    loader= DataLoader(pyg_graphs,batch_size=batch_size, shuffle=False)
    return loader
    
def get_all_pyg_loaders(train_graphs, train_targets,  test_graphs, test_targets ,\
    train1_graphs , train1_targets,valid_graphs, valid_targets,generated_graphs,generated_targets,batch_size=10):
    
    train_pyg_graphs,test_pyg_graphs,train1_pyg_graphs,valid_pyg_graphs,generated_pyg_graphs,all_pyg=\
    get_pyg_graphs_from_all_nx_sets(train_graphs, train_targets,  test_graphs, test_targets \
        , train1_graphs , train1_targets,valid_graphs, valid_targets,generated_graphs,generated_targets)
    train_loader = DataLoader(train_pyg_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_pyg_graphs, batch_size=batch_size, shuffle=False)
    valid_loader=DataLoader(valid_pyg_graphs, batch_size=batch_size, shuffle=False)
    generated_loader=DataLoader(generated_pyg_graphs, batch_size=batch_size, shuffle=False)
    train1_loader=DataLoader(train1_pyg_graphs, batch_size=batch_size, shuffle=False)
    all_loader=DataLoader(all_pyg , batch_size=batch_size, shuffle=False)
    return train_loader,test_loader,valid_loader,generated_loader,train1_loader,all_loader


