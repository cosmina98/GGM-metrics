from evaluation.mol_structure import list_of_nx_graphs_to_smiles
import dgl
import networkx as nx
from evaluation.moses.metrics import get_all_metrics
from evaluation.evaluator import Evaluator
from evaluation.new_metric import AucRocEvaluation
from evaluation.stats import eval_graph_list
from evaluation.utils import _preprocess
import os 
import sys
from evaluation.new_structural_metric import symmetric_graph_set_distance,atom,cycle,neighborhood
from evaluation.similarity_metric import graph_set_similarity,atom,cycle,pairwise_neighborhood
current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)
from rdkit import RDLogger 
RDLogger.EnableLog('rdApp.*')
RDLogger.DisableLog('rdApp.*')  
import math
import time

 
def evaluate(reference_nx_graphs, generated_nx_graphs, device,  metrics_type, structural_statistic=None,train1_graphs=None , train1_targets=None,train2_graphs=None , train2_targets=None, test_graphs=None, test_targets=None, generated_graphs=None, generated_targets=None):
    try:
    
        reference_graphs_dgl = [ dgl.from_networkx(nx.MultiDiGraph(g),node_attrs=['label','attr'], edge_attrs=['label','attr']).to(device) for g in train1_graphs if g.number_of_nodes()>1 and g.number_of_edges()>0] # Convert graphs to DGL f,rom NetworkX
        generated_graphs_dgl=[ dgl.from_networkx(nx.MultiDiGraph(g),node_attrs=['label','attr'], edge_attrs=['label','attr']).to(device) for g in generated_graphs if g.number_of_nodes()>1 and g.number_of_edges()>0 ]
        input_dim=len(reference_graphs_dgl[0].ndata['attr'][0])
        edge_feat_dim=len(reference_graphs_dgl[0].edata['attr'][0])
    except:
        generated_nx_graphs=_preprocess(generated_nx_graphs,label=0,cont_label=[0],discrete_node_label_name='',discrete_edge_label_name='',continuous_node_label_name='',continuous_edge_label_name='')
        reference_nx_graphs =_preprocess(reference_nx_graphs,label=0,cont_label=[0],discrete_node_label_name='',discrete_edge_label_name='',continuous_node_label_name='',continuous_edge_label_name='')
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
              
              if (structure=='nspdk')  or (structure=='WL') : 
                eval=Evaluator(feature_extractor ='mmd-structure',device=device ,statistic=structure)
                structural_metrics=eval.evaluate_all(generated_dataset=generated_graphs_dgl,reference_dataset=reference_graphs_dgl)
                metrics.update(structural_metrics)
                #print(metrics)
              elif (structure=='common_substructures'):
                name_decomposition_list = []
                name_decomposition_list.append(['node_edge',atom()])
                name_decomposition_list.append(['unlabelled_graph_cycle',cycle(abstraction_level='unlabelled_graph_process')])
                name_decomposition_list.append(['cycle', cycle()])
                name_decomposition_list.append(['neighborhood r=1', neighborhood(size=1)])
                #name_decomposition_list.append(['neighborhood r=2', neighborhood(size=2)])
                #print('here')
                for name, df in name_decomposition_list:
                      start = time.time()
                      score = graph_set_similarity(reference_nx_graphs, generated_nx_graphs, decomposition_function=df)
                      elapsed = time.time() - start
                      print('%30s: %5.3f   [%4.1f s  (%4.1f m)]'%(name, score, elapsed, elapsed/60))
                      metrics[name +'_similarity']= score
                      metrics[name+ "_similarity_time" ]= elapsed
                      
              else: 
                structural_metrics=eval_graph_list(reference_nx_graphs,generated_nx_graphs, methods=[structure]), 
                metrics.update(structural_metrics[0])
                # print(structural_metrics)
                 
        print('Now computing structural based metrics')
        #try
        fun()
          
        #except: print('Cannot compute these structural metrics')
        
    if 'molecular'  in metrics_type:
        
        
        def func(reference_nx_graphs, generated_nx_graphs):
            reference_smiles_list=list_of_nx_graphs_to_smiles(reference_nx_graphs)
            generated_smiles_list=list_of_nx_graphs_to_smiles(generated_nx_graphs)
            mol_metrics=get_all_metrics(gen=generated_smiles_list,train=reference_smiles_list,test=reference_smiles_list)
            metrics.update(mol_metrics)
        
        print('Now computing  molecular specific metrics')
        func(reference_nx_graphs,generated_nx_graphs)
    
        rint('Cannot compute molecular metrics for this type of graphs')
    else: None
    
    if 'auc_roc' in metrics_type:
        
        new_metric_dict={} 
        def fun(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets):
            #classifier_nn=AucRocEvaluation(classifier_type='nn')
            classifier_nspdk=AucRocEvaluation(classifier_type='scikit')
            try:
                res, time=classifier_nspdk.evaluate(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets)
                for key in list(res.keys()):
                    res[key + '_time'] = time
                metrics.update(res)
            except:
                    try:
                        generated_graphs=_preprocess(generated_graphs,label=0,discrete_node_label_name='',discrete_edge_label_name='',continuous_node_label_name='',continuous_edge_label_name='')
                        train1_graphs=_preprocess(train1_graphs,label=0,discrete_node_label_name='',discrete_edge_label_name='',continuous_node_label_name='',continuous_edge_label_name='')
                        train2_graphs=_preprocess(train2_graphs,label=0,discrete_node_label_name='',discrete_edge_label_name='',continuous_node_label_name='',continuous_edge_label_name='')
                        test_graphs=_preprocess(test_graphs,label=0,discrete_node_label_name='',discrete_edge_label_name='',continuous_node_label_name='',continuous_edge_label_name='')

                        res, time=classifier_nspdk.evaluate(train1_graphs,train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets)
                        for key in list(res.keys()):
                                res[key + '_time'] = time
                        metrics.update(res)
            
                    except:
                          
                         print('Error when computing AUC_ROC with NSPDK')
            #try:
                #res2, time2=classifier_nn.evaluate(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets)
               # for key in list(res2.keys()):
                    #res2[key + '_time'] = time2
                #metrics.update(res2)
            #except:
                    #print('Error when computing AUC_ROC with an nn classifier')

        #try:  
        print('Now computing the auc_roc based  metric')
        fun(train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs,generated_targets)
        #except:None
    #print(metrics)
    new_metrics = {key: value for (key, value) in metrics.items() if not math.isnan(value)}

    return new_metrics