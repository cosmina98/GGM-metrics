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
from sklearn.utils import shuffle                                                                                                                                                                                                                                                                                                                                                                                                                                                              




def preprocess(nx_dataset, set_label_equal_to_attribute=False,set_attribute_equal_to_label=False):
    discrete_node_label_name='label' #leave it blank if this does not exist, default: 'label'
    #leave blank if this does not exist, default: 'label'
    continuous_node_label_name='attr'  #leave it blank if this does not exist default: 'attr'
    discrete_edge_label_name='label'  #leave it blank if this does not exist  ,default: 'label'
    continuous_edge_label_name='attr' #leave it blank if this does not exist , default: 'attr'
    #dicrete labels should be set to 'label'
    #continous labels should be set to 'attr'

    processed_dataset=[]
    for G in nx_dataset:
        if (discrete_node_label_name!='label'):
            dict=nx.get_node_attributes(G, discrete_node_label_name)
            if dict!='':   
                nx.set_node_attributes(G, dict, 'label') 
            else:  pass
        if (discrete_edge_label_name!='label'):
            dict=nx.get_edge_attributes(G, discrete_edge_label_name)
            if dict!='':  
                nx.set_edge_attributes(G, dict, 'label')
            else:  pass
        if (continuous_node_label_name!='attr'):
            dict=nx.get_node_attributes(G, continuous_node_label_name)
            if dict!='':   
                nx.set_node_attributes(G, dict, 'attr') 
            else:  pass
        if (continuous_edge_label_name!='attr'):
            dict=nx.get_edge_attributes(G, continuous_edge_label_name)
            if dict!='':  
                nx.set_edge_attributes(G, dict, 'attr') 
            else:  pass
        if set_label_equal_to_attribute: 
            dict=nx.get_node_attributes(G, discrete_node_label_name)
            nx.set_node_attributes(G, dict, 'attr')
            dict=nx.get_edge_attributes(G, discrete_edge_label_name)
            nx.set_edge_attributes(G, dict, 'attr') 
        if set_attribute_equal_to_label: 
            dict=nx.get_node_attributes(G, continuous_node_label_name)
            nx.set_node_attributes(G, dict, 'label')
            dict=nx.get_edge_attributes(G, continuous_node_label_name)
            nx.set_edge_attributes(G, dict, 'label') 
            
        H = nx.MultiDiGraph(G)
        processed_dataset.append(H)
    return processed_dataset  

def get_clean_datasets(reference_graphs,generated_graphs):
    generated_graphs=preprocess(generated_graphs)
    reference_graphs=preprocess(reference_graphs)
    for g in generated_graphs:
        if (len(g.edges))==0:
             generated_graphs.remove(g)
    for g in reference_graphs:
        if (len(g.edges))==0:
             reference_graphs.remove(g)
    return reference_graphs,generated_graphs

def get_data(name, path='data/smiles/',return_smiles=False):
    splits={}
    RDLogger.EnableLog('rdApp.*')
    RDLogger.DisableLog('rdApp.*')  
    split_names=['train_smiles','test_smiles', 'train1_pos_smiles','train1_neg_smiles','train2_pos_smiles','train2_neg_smiles','valid_smiles','train_tragets','test_targets', 'valid_targets' ]
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
    train1_graphs, train1_targets = shuffle(train1_graphs, train1_targets)
    train2_pos_graphs =list(list_of_smiles_to_nx_graphs(splits['train2_pos_smiles']))
    train2_neg_graphs =list(list_of_smiles_to_nx_graphs(splits['train2_neg_smiles']))
    train2_graphs = train2_pos_graphs+ train2_neg_graphs
    train2_targets = np.array([1]*len(train2_pos_graphs) + [0]*len(train2_neg_graphs))
    train2_graphs, train2_targets = shuffle(train2_graphs, train2_targets)
    graphs=[train_graphs,train_targets,test_graphs,test_targets,train1_graphs,train1_targets,train2_graphs,train2_targets,valid_graphs, valid_targets]
    if return_smiles:
         return graphs,splits
    else:  return graphs
    

def get_generated_data(name, path='data/smiles/'):
    pos_list, neg_list=[],[]
    path_postives=path+'{}/{}_gen_pos.txt'.format(name,name)
    with open(path_postives) as my_file:
         for line in my_file:
            pos_list.append(line.strip())
    pos_graphs=list_of_smiles_to_nx_graphs(pos_list)
    path_negatives=path+'{}/{}_gen_neg.txt'.format(name,name)
    with open(path_negatives) as my_file:
         for line in my_file:
            neg_list.append(line.strip())
    neg_graphs=list_of_smiles_to_nx_graphs(neg_list)
    return pos_graphs,neg_graphs


 

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


