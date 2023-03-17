import numpy as np
#from eden.graph import vectorize
#from eden.graph import Vectorizer

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import global_mean_pool
import torch

class GCN(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels,edge_feat_dim, output_channels=2,chem_encoder=False):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.chem_encoder=chem_encoder 
        self.embedding=None  

        if self.chem_encoder:
            self.atom_encoder = AtomEncoder(emb_dim = 100)
            self.bond_encoder = BondEncoder(emb_dim = 100)
            in_channels=100
            edge_feat_dim=100
            
        if edge_feat_dim!=0:
            self.conv1 = GATConv(in_channels, hidden_channels,edge_dim=edge_feat_dim)
            self.conv2 = GATConv(hidden_channels, hidden_channels,edge_dim=edge_feat_dim )
        else: 
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            
        self.lin = Linear(hidden_channels, output_channels)
      
        
    def forward(self, x, edge_index,edge_attr,batch):
        # 1. Obtain node embeddings 
        if self.chem_encoder:
            x=self.atom_encoder(x)
            edge_attr=self.bond_encoder(edge_attr)
        x = self.conv1(x, edge_index,edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index,edge_attr)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x,batch)  # [batch_size, hidden_channels]
        
        self.embedding=x

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        
        return x
    
class NN_classifier():
    
    def __init__ (self,model=None,optimiser=None,loss_function =torch.nn.CrossEntropyLoss(),batch_size=10,epochs=10,verbose=False,
     lr=1e-2):
    
      self.batch_size=batch_size
      self.epochs=epochs
      self.num_h_layers=2
      self.lr=lr
      self.verbose=verbose
      self.shuffle=False
      self.seed=42
      self.running_loss = 0
      self.loss_function = loss_function
      self.sm = torch.nn.Softmax()
      #default optimiser: adam 
      self.h_dim=32
      if model==None:
        model=GCN(in_channels=100,hidden_channels=64,edge_feat_dim=100)
      self.model=model
      if optimiser==None:
        self.optimiser=torch.optim.Adam(self.model.parameters(), lr=self.lr)
      else:self.optimiser=optimiser
      for g in self.optimiser.param_groups:
           g['lr'] = self.lr
               
    def fit(self,X,y):
        self.model.train()
        for epoch in range(self.epochs):  # loop over the dataset n times
          running_loss =self.running_loss
          for  data in X:
            # forward + compute loss+ backward + optimize
            outputs = self.model(data.x, data.edge_index,data.edge_attr, data.batch)
            loss = self.loss_function(outputs, data.y)
            loss.backward()
            self.optimiser.step()
            # zero the parameter gradients
            self.optimiser.zero_grad()
            # print statistics
            running_loss += loss.item()
            if epoch % self.batch_size == (self.batch_size-1):    # print every "batch_size" mini-batches
                if self.verbose==True:
                  print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {running_loss / self.batch_size:.3f}')
                running_loss = 0.0

    def predict_proba(self, X_test): 
        self.model.eval()
        correct = 0
        preds=[]
        for data in X_test:  # Iterate in batches over the training/test dataset.
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)  
                pred=self.sm( out)  
                num_pred=out.argmax(dim=1) 
                preds.append(pred)
                correct += int((num_pred == data.y).sum())  # Check against ground-truth labels.
        #print(X_test)
        acc=correct / len(X_test) 
        x=[x.clone().detach() for x in preds]
        #print(x)
        try:
            preds=torch.tensor(torch.stack((x)).flatten(0,1))
        except:
          try:  preds=torch.cat( (torch.stack((x[:-1])).flatten(0,1) ,  x[-1]) ,dim=0)
          except: preds=None
        return preds #torch.stack(prediction_list)
      
    def get_flat_predictions(preds):
         return torch.cat((torch.flatten(torch.stack(preds[:-1])) ,  preds[-1]), dim=0)