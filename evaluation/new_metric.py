import numpy as np
#from eden.graph import vectorize
#from eden.graph import Vectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from eden.ml.estimator import EdenEstimator
from .new_metric_classifier import GCN,NN_classifier
from .utils import get_pyg_loader_from_nx
import time
from sklearn.utils import shuffle


def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        return results, end - start
    return wrapper
 
 
    
class AucRocEvaluation():
    def __init__(self, classifier_type='scikit', random_state=1,*args, **kwargs):
        if classifier_type=='scikit':
            self.estimator =ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
            self.graph_encoder =EdenEstimator(r=2,d=4)
        if classifier_type=='nn':
            self.batch_size=10
            self.model =GCN(in_channels=9, hidden_channels=64,edge_feat_dim=3,chem_encoder=True) 
            if kwargs.get('model'):
                   self.model=kwargs.get('model')
            self.estimator=NN_classifier(self.model,batch_size=self.batch_size)
            self.graph_encoder =None
        self.random_state=random_state
        self.classifier_type=classifier_type
        super().__init__(*args, **kwargs)
        
        
    def predict(self, test_embedding):
        preds= self.estimator.predict_proba(test_embedding)[:,1]
        return preds

    def extract_features(self, train_graphs,train_targets,test_graphs):
        graph_encoder = self.graph_encoder.fit(train_graphs,train_targets)
        train_embedding=graph_encoder.transform(train_graphs)
        test_embedding = graph_encoder.transform(test_graphs)
        return train_embedding, test_embedding
    
    def compute_auc(self, train_graphs, train_targets, test_graphs, test_targets):
        if self.classifier_type=='scikit':
            train_embedding, test_embedding= self.extract_features(train_graphs,train_targets,test_graphs)
        else: train_embedding, test_embedding= get_pyg_loader_from_nx(train_graphs, train_targets,self.batch_size), get_pyg_loader_from_nx(test_graphs, test_targets,self.batch_size)
        self.estimator.fit( train_embedding,train_targets)
        preds=self.predict(test_embedding )
        auc=roc_auc_score(test_targets, preds)
        return auc
        
    def score(self, r1,r2,r3,r4):
        epsilon=1e-3
        r1=max(0.5, r1)
        r2=max(0.5, r2)
        r3=max(0.5, r3)
        r4=max(0.5, r4)
        if (r2-r1) <epsilon:
            metric=0
        else: 
            score1=min(1,max(0,r4-r1)/max(0, r2-r1))
            score2=min(1,1-max(0,r1-r3)/r1)
            #print(score1, score2)
            metric=np.sqrt(score1*score2)
        return metric
   
    @time_function
    def evaluate_with_random_splits(self,reference_graphs, reference_targets, generated_graphs, generated_targets):
        train_graphs, test_graphs, train_targets, test_targets = train_test_split(reference_graphs, reference_targets, train_size=.7,random_state=self.random_state)
        train1_graphs, _, train1_targets,  _= train_test_split(train_graphs, train_targets, test_size=0.5, random_state= self.random_state) 
        train1_plus_generated_graphs,train1_plus_generated_targets=train1_graphs+generated_graphs,train1_targets+generated_targets   
        auc_1= self.compute_auc(train1_graphs, train1_targets, test_graphs, test_targets)
        auc_2=self.compute_auc(train_graphs, train_targets, test_graphs, test_targets)
        auc_3=self.compute_auc(generated_graphs, generated_targets, test_graphs, test_targets)
        auc_4=self.compute_auc(train1_plus_generated_graphs,train1_plus_generated_targets, test_graphs, test_targets)
        print(auc_1, auc_2,auc_3,auc_4 )
        metric=self.score(auc_1, auc_2,auc_3,auc_4)
        if self.classifier_type=='scikit':
            return {'AUC_ROC_based_metric_with_nspdk':metric}
        else: return {'AUC_ROC_based_metric_with_nn_classifier':metric}
        
    
    @time_function
    def evaluate(self,train1_graphs , train1_targets,train2_graphs , train2_targets, test_graphs, test_targets, generated_graphs, generated_targets):
        train_graphs, train_targets=train1_graphs +  train2_graphs , list(train1_targets)+list(train2_targets)
        train_graphs, train_targets=shuffle(train_graphs, train_targets)
        train1_plus_generated_graphs,train1_plus_generated_targets=train1_graphs+generated_graphs,list(train1_targets)+list(generated_targets)      
        auc_1= self.compute_auc(train1_graphs, train1_targets, test_graphs, test_targets)
        auc_2=self.compute_auc(train_graphs, train_targets, test_graphs, test_targets)
        auc_3=self.compute_auc(generated_graphs, generated_targets, test_graphs, test_targets)
        auc_4=self.compute_auc( train1_plus_generated_graphs,train1_plus_generated_targets, test_graphs, test_targets)
        print(auc_1, auc_2,auc_3,auc_4 )
        metric=self.score(auc_1, auc_2,auc_3,auc_4)
        if self.classifier_type=='scikit':
            return {'AUC_ROC_based_metric_with_nspdk':metric}
        else: return {'AUC_ROC_based_metric_with_nn_classifier':metric}
        
        