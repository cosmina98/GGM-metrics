import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir
import enum
from rdkit.Chem.rdmolops import AddHs
import networkx as nx
import matplotlib
from networkx.drawing.nx_agraph import graphviz_layout,pygraphviz_layout
from pylab import rcParams
import matplotlib as mpl
from matplotlib import pyplot as plt
import random
import numpy as np
from colour import Color
import os 
import sys
from .atom_bond_encoder import  atom_to_feature_vector,bond_to_feature_vector
import requests
from io import StringIO
import pandas as pd
current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)

#for zinc


"""
def get_dict_of_nodes():
    dict_of_nodes={0: 'C', 1: 'O',2: 'N',3: 'F',4: 'C',5: 'S', 6: 'Cl', 7: 'O', 8: 'N',9: 'Br', 10: 'N', 11: 'N', 12: 'N', 13: 'N', 14: 'S ', 15: 'I', 16: 'P', 17: 'O', 18: 'N', 19: 'O',20: 'S', 21: 'P' ,22: 'P',23: 'C', 24: 'P',25: 'S',26: 'C',27: 'P'}
    return dict_of_nodes
    
def get_atomic_number():
    dict_of_atomic_no ={ 'C':6, 'O':8, 'N':7, 'F':9, 'S':16, 'Cl':17,  'Br':35, 'I':53, 'P':15}
    return dict_of_atomic_no
"""

def foo(g):
  
    nx.draw(g)
    Draw.MolToMPL(nx_to_mol(g))
   
    return None


def rdkmol_to_nx(mol):
    #  rdkit-mol object to nx.graph
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), label=int(atom.GetAtomicNum()), attr=atom_to_feature_vector(atom) ,  label_name=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),label=(int(bond.GetBondTypeAsDouble())),
                       attr=bond_to_feature_vector(bond),
                       edge_label=str(bond.GetBondType()) )
    return graph


def smiles_to_mol(list_of_smiles):
    list_of_rdkit_mols=[]
    for _,smile in enumerate(list_of_smiles):
        try:
            mol = Chem.MolFromSmiles(smile)
            list_of_rdkit_mols.append(mol)
        except:list_of_rdkit_mols.append(Chem.MolFromSmiles('C'))
    return list_of_rdkit_mols
    
def list_of_smiles_to_nx_graphs(smiles):
    list_of_nx_graphs=[]
    for i,smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
           list_of_nx_graphs.append(rdkmol_to_nx(mol))
        else:
            continue
         
    return list_of_nx_graphs

def nx_to_rdkit(graph):
    m = Chem.MolFromSmiles('')
    mw = Chem.RWMol(m)
    atom_index = {}
    for n, d in graph.nodes(data=True):
        atom_index[n] = mw.AddAtom(Chem.Atom(d['label']))
    for a, b, d in graph.edges(data=True):
        start = atom_index[a]
        end = atom_index[b] 
        bond_type=d['edge_label']
        try:
            mw.AddBond(start, end, eval("rdkit.Chem.rdchem.BondType.{}".format(bond_type)))
        except:
            mw.AddBond(start, end, eval("rdkit.Chem.rdchem.BondType.{}".format('SINGLE')))
            #print('exc',bond_type)
            continue
            raise Exception('bond type not implemented')

    mol = mw.GetMol()
    return mol


def list_of_nx_graphs_to_smiles(graphs, file=None):
    # writes smiles strings to a file
    chem = [nx_to_rdkit(graph) for graph in graphs]
    smis = [Chem.MolToSmiles(m) for m in chem]
    if file:
        with open(file, 'w') as f:
            f.write('\n'.join(smis))
    return smis



def Hex_color(bond_type):
    random.seed( np.sum([ord(c) for c in bond_type]) -30+ ord(bond_type[0]))
    L = '0123456789ABCDEF'
    x= Color('#'+ ''.join([random.choice(L) for i in range(6)][:]))
    return x.get_hex()

def get_edge_color_list(mol_nx):
    edge_color=[ Hex_color(data[2]) for data in mol_nx.edges(data = 'edge_label')] 
    #print(edge_color)
    return edge_color

def return_colors_for_atoms(mol_nx):
    random.seed(767)
    color_map = {}
    for idx in mol_nx.nodes():
      if mol_nx.nodes[idx]['label_name'] not in color_map:
          color_map[mol_nx.nodes[idx]['label_name']] ="#%06x" % random.randint(sum([ord(c) for c in mol_nx.nodes[idx]['label']]), 0xFFFFFF) 
    mol_colors = []
    for idx in mol_nx.nodes():
        if (mol_nx.nodes[idx]['label_name'] in color_map):
            mol_colors.append(color_map[mol_nx.nodes[idx]['label_name']])
        else:
            mol_colors.append('gray')
    return mol_colors

def get_labels(mol_nx):
    return nx.get_node_attributes(mol_nx, 'label_name')
#set(nx.get_node_attributes(mol_nx, 'atom_symbol').values())
#colors=[i/len(mol_nx.nodes) for i in range(len(mol_nx.nodes))]

def draw_one_mol(G, ax=None):
    rcParams['figure.figsize'] = 7.5,5
    #color_lookup = {k:v for v, k in enumerate(sorted((nx.get_node_attributes(G, "atom_symbol"))))}
    selected_data = dict( (n, ord(d['label_name'][0])**3 ) for n,d in G.nodes().items() )
    selected_data=[v[1] for k, v in enumerate(selected_data.items())]
    #print(selected_data)
    low, *_, high = sorted(selected_data)
    seed=123
    random.seed(seed)
    np.random.seed(seed)    
    pos=pygraphviz_layout(G)
    nx.draw_networkx_nodes(G,pos,
            #labels= get_labels(G),
            node_size=100,
            #edgecolors='black',
            cmap='tab20c_r',
            vmin=low,
            vmax=high,
            node_color=[selected_data],
            ax=ax)
    nx.draw_networkx_labels(G,pos, get_labels(G),ax=ax)
    nx.draw_networkx_edges(G, pos,
          
            width=3,
            edge_color=get_edge_color_list(G),
         ax=ax)
    #nx.draw_networkx_edge_labels(G, pos, get_edge_labels(G), font_size=10,alpha=0.8,verticalalignment='bottom',
                                #horizontalalignment='center',clip_on='True',rotate='True' )
   


def get_edge_labels(G):
  return nx.get_edge_attributes(G,'edge_label')
   
def get_adjency_matrix(mol_nx):
    # print out the adjacency matrix ---------------------------------------------- 
    matrix = nx.to_numpy_matrix(mol_nx)
    print(matrix)
    return matrix



def draw_graphs(list_of_graph_molecules, num_per_line=3,labels=None):
      ixs=[]
      num_per_line_ax_n=0
      if len(list_of_graph_molecules)%num_per_line==0:
       lines=int(len(list_of_graph_molecules)/num_per_line)
      else:
        lines=len(list_of_graph_molecules)//num_per_line+1
        num_per_line_ax_n=len(list_of_graph_molecules) % num_per_line
        for i in range(num_per_line-num_per_line_ax_n):
          ix=lines-1,num_per_line-i-1
          ixs.append(ix)
          
      #print(lines)
      fig, ax = plt.subplots(lines, num_per_line)
      fig.set_figheight(10)
      fig.set_figwidth(15)
      for i, mol_nx in enumerate(list_of_graph_molecules):
        ix = np.unravel_index(i, ax.shape)
        draw_one_mol(mol_nx, ax=ax[ix])
        if labels is not None:
            ax[ix].set_title(labels[i], fontsize=8)
      for ix in ixs:
           ax[ix].set_axis_off()
 


def load_csv_data_from_a_PubChem_assay(assay_id):
    #url = f'https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid={assay_id}'
    url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{assay_id}/concise/csv'
    #df_raw=pd.read_csv(url)
    df_raw=pd.read_csv(url)
    #print(df_raw.head())
    return(df_raw)

def drop_sids_with_no_cids(df):
    df = df.dropna( subset=['cid'] )
    #Remove CIDs with conflicting activities
    cid_conflict = []
    idx_conflict = []

    for mycid in df['cid'].unique() :
        
        outcomes = df[ df.cid == mycid ].activity.unique()
        
        if len(outcomes) > 1 :
            
            idx_tmp = df.index[ df.cid == mycid ].tolist()
            idx_conflict.extend(idx_tmp)
            cid_conflict.append(mycid)

    #print("#", len(cid_conflict), "CIDs with conflicting activities [associated with", len(idx_conflict), "rows (SIDs).]")
    df = df.drop(idx_conflict)

    #Remove redundant data

    df = df.drop_duplicates(subset='cid')  # remove duplicate rows except for the first occurring row.
    #print(len(df['sid'].unique()))
    #print(len(df['cid'].unique()))
    return df
    
     

def download_smiles_given_cids_from_pubmed(list_of_cids,chunk_size = 200): #returns df of smiles and cids
    df_smiles = pd.DataFrame()

    num_cids = len(list_of_cids)
    list_dfs = []
    if num_cids % chunk_size == 0 :
        num_chunks = int( num_cids / chunk_size )
    else :
        num_chunks = int( num_cids / chunk_size ) + 1

    #print("# CIDs = ", num_cids)
    #print("# CID Chunks = ", num_chunks, "(chunked by ", chunk_size, ")")

    for i in range(0, num_chunks) :
        idx1 = chunk_size * i
        idx2 = chunk_size * (i + 1)
        cidstr = ",".join( str(x) for x in list_of_cids[idx1:idx2] )

        url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + cidstr + '/property/IsomericSMILES/TXT')
        res = requests.request('GET',url)
        data = pd.read_csv( StringIO(res.text), header=None, names=['smiles'] )
        list_dfs.append(data)
        
        time.sleep(0.2)
        
        #if ( i % 5 == 0 ) :
            #print("Processing Chunk ", i)
    df_smiles = pd.concat(list_dfs,ignore_index=True)
    df_smiles[ 'cid' ] = list_of_cids   

    return df_smiles

def load_PUBCHEM_dataset(assay_id,num_graphs=None):
    df_raw=load_csv_data_from_a_PubChem_assay(assay_id=assay_id)
    print(len(df_raw))
    #Drop substances without Inconclusive activity
    df_raw=df_raw[df_raw['Activity Outcome']!='Inconclusive']
    #Select active/inactive compounds for model building
    df=df_raw[ (df_raw['Activity Outcome'] == 'Active' ) | 
             (df_raw['Activity Outcome'] == 'Inactive' ) ].rename(columns={"CID": "cid", "SID":"sid","Activity Outcome": "activity"})
    #drop duplicates, and comnflicting activities, and substances with no cids
    df=drop_sids_with_no_cids(df)
    #label encoding
    df['activity'] = [ 0 if x == 'Inactive' else 1 for x in df['activity'] ]
    df_smiles=download_smiles_given_cids_from_pubmed(df.cid.astype(int).tolist())
    X=df_smiles.smiles.tolist()
    y=df.activity.astype(int).tolist()
    num_graphs=len(X) if num_graphs!=None else num_graphs
    return(list_of_smiles_to_nx_graphs(X)[:num_graphs],y[:num_graphs])

