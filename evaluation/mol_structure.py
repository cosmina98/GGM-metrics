
import networkx as nx
from rdkit.Chem import Descriptors
from evaluation.moses.metrics.utils import *
from evaluation.moses.metrics import *
from evaluation.moses.dataset import *




def get_atomic_number():
    dict_of_atomic_no ={ 'C':6, 'O':8, 'N':7, 'F':9, 'S':16, 'Cl':17,  'Br':35, 'I':53, 'P':15}
    return dict_of_atomic_no

def get_dict_of_nodes():
    dict_of_nodes={0: 'C', 1: 'O',2: 'N',3: 'F',4: 'C',5: 'S', 6: 'Cl', 7: 'O', 8: 'N',9: 'Br', 10: 'N', 11: 'N', 12: 'N', 13: 'N', 14: 'S ', 15: 'I', 16: 'P', 17: 'O', 18: 'N', 19: 'O',20: 'S', 21: 'P' ,22: 'P',23: 'C', 24: 'P',25: 'S',26: 'C',27: 'P'}
    return dict_of_nodes

def nx_to_mol(nx_graph , edge_label='label', node_label='label'):
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    
    #dictionary of atomic numbers
    dict_of_atomic_no=get_atomic_number()
    
    #dictionary of nodes
    node_atom_values=get_dict_of_nodes()
    
    
    for i,n in enumerate(nx_graph.nodes(data=True)):
        a = Chem.Atom( dict_of_atomic_no[node_atom_values[int(n[1][node_label].numpy())]])
        molIdx = mol.AddAtom(a)
        #print(molIdx)
        node_to_idx[i] = molIdx
        
    for edge in nx_graph.edges(data=True):
            bond=int(edge[2][edge_label])
            ix=edge[0]
            iy=edge[1]
            
            # only traverse half the matrix
            if iy <= ix:
                continue
            if bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                
            elif bond == 3:
                    bond_type=Chem.rdchem.BondType.TRIPLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy] ,bond_type)
            elif bond == 4:
                bond_type = Chem.rdchem.BondType.AROMATIC
                mol.AddBond(node_to_idx[ix], node_to_idx[iy] ,bond_type)

        # Convert RWMol to Mol object
    mol = mol.GetMol()  
    try:
        Chem.SanitizeMol(mol)
    except:
        # print(node_list)
        # print(Chem.MolToSmiles(mol))
        return None
    return mol   

def mol_to_smiles(list_of_mols):
    smiles=[Chem.MolToSmiles(g) for g in losters_mol]
    return smiles

def foo():
    for g in lobsters:
        nx.draw(g)
        Draw.MolToMPL(nx_to_mol(g))
        break
    return None
