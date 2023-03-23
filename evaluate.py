from evaluation.mol_structure import list_of_nx_graphs_to_smiles
import dgl
import networkx as nx
from evaluation.moses.metrics import get_all_metrics
from evaluation.evaluator import Evaluator
from evaluation.new_metric import AucRocEvaluation
import os 
import sys
current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)
from rdkit import RDLogger 
RDLogger.EnableLog('rdApp.*')
RDLogger.DisableLog('rdApp.*')  
import math

 
def evaluate(reference_nx_graphs, generated_nx_graphs, device,  metrics_type, structural_statistic=None,train1_graphs=None , train1_targets=None,train2_graphs=None , train2_targets=None, test_graphs=None, test_targets=None, generated_graphs=None, generated_targets=None):
    reference_graphs_dgl = [ dgl.from_networkx(nx.MultiDiGraph(g),node_attrs=['label','attr'], edge_attrs=['label','attr']).to(device) for g in reference_nx_graphs if g.number_of_nodes()>1 and g.number_of_edges()>0] # Convert graphs to DGL f,rom NetworkX
    generated_graphs_dgl=[ dgl.from_networkx(nx.MultiDiGraph(g),node_attrs=['label','attr'], edge_attrs=['label','attr']).to(device) for g in generated_nx_graphs if g.number_of_nodes()>1 and g.number_of_edges()>0 ]

    
    input_dim=len(reference_graphs_dgl[0].ndata['attr'][0])
    edge_feat_dim=len(reference_graphs_dgl[0].edata['attr'][0])
    metrics={}
    if  'nn' in metrics_type:
        try:
         
            print('Now computing classifier based metrics')
            eval=Evaluator(feature_extractor ='gin',device=device, edge_feat_loc='attr' , node_feat_loc='attr', input_dim=input_dim,edge_feat_dim=edge_feat_dim, hidden_dim=36)
            nn_metrics=eval.evaluate_all(generated_dataset=generated_graphs_dgl,reference_dataset=reference_graphs_dgl)
            metrics.update(nn_metrics)
        except: print('Cannot compute the classifier based metrics') 
            
    if 'structural' in metrics_type  :
        
        def fun():
            for structure in structural_statistic:
                eval=Evaluator(feature_extractor ='mmd-structure',device=device ,statistic=structure)
                structural_metrics=eval.evaluate_all(generated_dataset=generated_graphs_dgl,reference_dataset=reference_graphs_dgl)
                metrics.update(structural_metrics)
        try:  
         print('Now computing structural based metrics')
         fun()
          
        except: print('Cannot compute these structural metrics')
        
    if 'molecular'  in metrics_type:
        
        
        def func(reference_nx_graphs, generated_nx_graphs):
            reference_smiles_list=list_of_nx_graphs_to_smiles(reference_nx_graphs)
            generated_smiles_list=list_of_nx_graphs_to_smiles(generated_nx_graphs)
            mol_metrics=get_all_metrics(gen=generated_smiles_list,test=reference_smiles_list)
            metrics.update(mol_metrics)
        try: 
            print('Now computing  molecular specific metrics')
            func(reference_nx_graphs,generated_nx_graphs)
        except: 
            print('Cannot compute molecular metrics for this type of graphs')
    else: None
    
    if 'auc_roc' in metrics_type:
        
        new_metric_dict={} 
        def fun(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets):
            classifier_nn=AucRocEvaluation(classifier_type='nn')
            classifier_nspdk=AucRocEvaluation(classifier_type='scikit')
            try:
                res, time=classifier_nspdk.evaluate(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets)
                for key in list(res.keys()):
                    res[key + '_time'] = time
                metrics.update(res)
            except:print('Error when computing AUC_ROC with NSPDK')
            try:
                res2, time2=classifier_nn.evaluate(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets)
                for key in list(res2.keys()):
                    res2[key + '_time'] = time2
                metrics.update(res2)
            except:print('Error when computing AUC_ROC with an nn classifier')
        try:  
         print('Now computing the auc_roc based  metric')
         fun(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets)
        except:None
    new_metrics = {key: value for (key, value) in metrics.items() if not math.isnan(value)}

    return new_metrics