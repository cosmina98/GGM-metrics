
from evaluation.mol_structure import list_of_nx_graphs_to_smiles
import dgl
import networkx as nx
from rdkit.Chem import Descriptors
from evaluation.moses.metrics import get_all_metrics
from evaluation.evaluator import Evaluator
from evaluation.gin_evaluation import load_feature_extractor, MMDEvaluation
import copy
import fuckit
import os 
import sys
current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir
import enum
from rdkit import RDLogger 
RDLogger.EnableLog('rdApp.*')
RDLogger.DisableLog('rdApp.*')  

def evaluate(reference_nx_graphs, generated_nx_graphs, device,  metrics_type, structural_statistic):
    reference_graphs_dgl = [ dgl.from_networkx(g,node_attrs=['label','attr'], edge_attrs=['label','attr']).to(device) for g in reference_nx_graphs] # Convert graphs to DGL f,rom NetworkX
    generated_graphs_dgl = [ dgl.from_networkx(g,node_attrs=['label','attr'], edge_attrs=['label','attr']).to(device) for g in generated_nx_graphs] # Convert graphs to DGL from NetworkX
    
    input_dim=len(reference_graphs_dgl[0].ndata['attr'][0])
    edge_feat_dim=len(reference_graphs_dgl[0].edata['attr'][0])
    
    metrics={}
    if  'nn' in metrics_type:
    # assert continuous_node_label_name!='' ,'You need continuous features to be able to compute nn-based metrics'
        eval=Evaluator(feature_extractor ='gin',device='cpu', edge_feat_loc='attr' , node_feat_loc='attr', input_dim=input_dim,edge_feat_dim=edge_feat_dim, hidden_dim=36)
        nn_metrics=eval.evaluate_all(generated_dataset=reference_graphs_dgl,reference_dataset=reference_graphs_dgl)
        metrics.update(nn_metrics)
            
    if 'structural' in metrics_type  :
        def fun():
            for structure in structural_statistic:
                eval=Evaluator(feature_extractor ='mmd-structure',device='cpu' ,statistic=structure)
                structural_metrics=eval.evaluate_all(generated_dataset=generated_graphs_dgl,reference_dataset=reference_graphs_dgl)
                metrics.update(structural_metrics)
        try:  
         fun()
        except: print('Cannot compute these metrics for this type of graphs')
        
    if 'molecular'  in metrics_type:
        
        
        def func(reference_nx_graphs, generated_nx_graphs):
            reference_smiles_list=list_of_nx_graphs_to_smiles(reference_nx_graphs)
            generated_smiles_list=list_of_nx_graphs_to_smiles(generated_nx_graphs)
            mol_metrics=get_all_metrics(gen=generated_smiles_list,train=reference_smiles_list)
            metrics.update(mol_metrics)
        try: 
            func(reference_nx_graphs,generated_nx_graphs)
        except: 
            print('Cannot compute molecular metrics for this type of graphs. You might neeed  to manually change the definition of nx_to_mol:)\n')
    else: None
        
    return(metrics)