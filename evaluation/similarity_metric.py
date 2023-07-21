import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import time

from IPython.core.display import HTML
from toolz import curry
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from toolz import partition_all
import multiprocessing_on_dill as mp
import inspect
from toolz import curry
from collections import defaultdict
from collections import Counter
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import random
from toolz import partition_all
import multiprocessing_on_dill as mp
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack


def hash_list(seq):
    return hash(tuple(seq))


def hash_value(value, context=1, nbits=None):
    if nbits is None: bitmask = 4294967295
    else: bitmask = pow(2, nbits) - 1
    code = hash((value, context)) & bitmask
    #code += 1 # do this to reserve code 0: REMOVED so to work with 2^n sizes
    return code


@curry
def node_neighborhood_hash(u, graph=None):
    uh = hash(graph.nodes[u]['label'])
    edges_h = [hash((hash(graph.nodes[v]['label']), hash(graph.edges[u, v]['label']))) for v in graph.neighbors(u)]
    nh = hash_list(sorted(edges_h))
    ext_node_h = hash((uh, nh))
    return ext_node_h


def rooted_breadth_first_hash(graph, root):
    def invert_dict(mydict):
        reversed_dict = defaultdict(list)
        for key, value in mydict.items(): reversed_dict[value].append(key)
        return reversed_dict

    node_neighborhood_hash_func = node_neighborhood_hash(graph=graph)
    gid_dist_dict = nx.single_source_shortest_path_length(graph, root)
    dist_gids_dict = invert_dict(gid_dist_dict)
    distance_based_hashes = [sorted(list(map(node_neighborhood_hash_func, dist_gids_dict[d]))) for d in sorted(dist_gids_dict)]
    hash_bfs = [hash_list(seq) for seq in distance_based_hashes]
    return hash_list(hash_bfs)


def nocontext_nodes_hashes(graph):
    nocontext_nodes_hashes_list = [rooted_breadth_first_hash(graph, u) for u in graph.nodes()]
    return nocontext_nodes_hashes_list


def nocontext_edges_hashes(graph):
    nocontext_nodes_hashes_list = nocontext_nodes_hashes(graph)
    nocontext_nodes_hashes_dict = {u:nocontext_node_hash for u, nocontext_node_hash in zip(graph.nodes(), nocontext_nodes_hashes_list)} 
    nocontext_edges_hashes_list = [(*sorted([nocontext_nodes_hashes_dict[u], nocontext_nodes_hashes_dict[v]]), hash(graph.edges[u, v]['label'])) for u,v in graph.edges()]
    return nocontext_edges_hashes_list, nocontext_nodes_hashes_list


def nodes_hash(orig_graph, context=1, nbits=None, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False):
    if use_node_unlabelled_graph or use_edge_unlabelled_graph: graph = orig_graph.copy()
    else: graph = orig_graph
    if use_node_unlabelled_graph: 
        for u in graph.nodes(): graph.nodes[u]['label'] = '-'
    if use_edge_unlabelled_graph: 
        for e in graph.edges(): graph.edges[e]['label'] = '-'
    nocontext_edges_hashes_list, nocontext_nodes_hashes_list = nocontext_edges_hashes(graph)
    g_hash = hash_list(sorted(nocontext_edges_hashes_list))
    nodes_hashes_list = [hash_value((g_hash,nocontext_node_hash), context, nbits) for nocontext_node_hash in nocontext_nodes_hashes_list]
    return nodes_hashes_list


def graph_hash(orig_graph, context=1, nbits=None, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False):
    if use_node_unlabelled_graph or use_edge_unlabelled_graph: graph = orig_graph.copy()
    else: graph = orig_graph
    if use_node_unlabelled_graph: 
        for u in graph.nodes(): graph.nodes[u]['label'] = '-'
    if use_edge_unlabelled_graph: 
        for e in graph.edges(): graph.edges[e]['label'] = '-'
    nocontext_edges_hashes_list, nocontext_nodes_hashes_list = nocontext_edges_hashes(graph)
    g_hash = hash_list(sorted(nocontext_nodes_hashes_list)+sorted(nocontext_edges_hashes_list))
    g_hash = hash_value(g_hash, context, nbits)
    return g_hash

def function_signature(function_arguments_dict):
    if 'graphofgraph' in function_arguments_dict: function_arguments_dict.pop('graphofgraph','')
    function_name = inspect.stack()[1][3]
    signature = function_name + str(function_arguments_dict)
    return signature

def serial_decomposition(graphs, decomposition_function, nbits):
    graphofgraphs = [decomposition_function(construct(graph, nbits=nbits)) for graph in graphs]
    return graphofgraphs

def parallel_decomposition(graphs, decomposition_function, nbits):
    def _make_decomposition_func(decomposition_function, nbits):
        def decomposition_func(graphs):
            return serial_decomposition(graphs, decomposition_function, nbits=nbits)
        return decomposition_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    decomposition_func = _make_decomposition_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(decomposition_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx

def decomposition(graphs, decomposition_function, nbits, parallel=True):
    if parallel == True:
        graphofgraphs = parallel_decomposition(graphs, decomposition_function, nbits)
    else:
        graphofgraphs = serial_decomposition(graphs, decomposition_function, nbits)
    return graphofgraphs


def make_signature(underlying_signature='', added_signature='', min_size=None, max_size=None, use_positive=None, use_negative=None):
    sfx = ''
    if use_positive is True:
        sfx += '+'
    if use_negative is True:
        sfx += '-'
    signature = '%s%s'%(added_signature,sfx)
    if min_size is not None and max_size is not None:
        signature += '%d:%d'%(min_size,max_size)
    if underlying_signature != 'base':
        signature += '(%s)'%(underlying_signature)
    return signature

def make_edges_of_graph_of_graph(graphofgraph):
    nbits = graphofgraph.graph['base'].graph['nbits']
    for edge_id in graphofgraph.edges():
        u,v = edge_id
        edge_signature = graphofgraph.edges[edge_id]['signature']
        signature_hash = hash(edge_signature)
        edge_label = hash(tuple(sorted([graphofgraph.nodes[u]['label'],graphofgraph.nodes[v]['label']])))
        edge_label = hash_value(edge_label, context=signature_hash, nbits=nbits)
        graphofgraph.edges[edge_id]['label'] = edge_label
        process_hash = hash_value(edge_signature, context=signature_hash, nbits=nbits)
        graphofgraph.edges[edge_id]['process_hash'] = process_hash
    return graphofgraph

def make_subgraphs_graph_of_graph(base_graph, bunches=None, signatures=[], subgraph_mode='node', abstraction_level='graph_process'):
    graphofgraph = nx.Graph()
    nbits = base_graph.graph['nbits']
    for u in base_graph.nodes(): base_graph.nodes[u]['location_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['location_node_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['location_edge_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['location_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['node_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['edge_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['process_hash'] = []
    
    #add graph_process_hash and process_hash to graphofgraph nodes
    for u, (bunch, signature) in enumerate(zip(bunches, signatures)):
        if subgraph_mode == 'node':
            subgraph = nx.subgraph(base_graph, bunch)
        elif subgraph_mode == 'edge':
            subgraph = nx.edge_subgraph(base_graph, bunch)
        signature_hash = hash(signature)
        process_hash = hash_value(signature, context=signature_hash, nbits=nbits)
        graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False)
        node_unlabelled_graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=False)     
        edge_unlabelled_graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=True)
        unlabelled_graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=True)
        if abstraction_level=='graph_process': label = graph_process_hash
        if abstraction_level=='node_unlabelled_graph_process': label = node_unlabelled_graph_process_hash
        if abstraction_level=='edge_unlabelled_graph_process': label = edge_unlabelled_graph_process_hash
        if abstraction_level=='unlabelled_graph_process': label = unlabelled_graph_process_hash
        if abstraction_level=='process': label = process_hash
        graphofgraph.add_node(
            u, 
            label=label, 
            graph_process_hash=graph_process_hash, 
            node_unlabelled_graph_process_hash=node_unlabelled_graph_process_hash,
            edge_unlabelled_graph_process_hash=edge_unlabelled_graph_process_hash,
            unlabelled_graph_process_hash=unlabelled_graph_process_hash,
            process_hash=process_hash, 
            subgraph=nx.Graph(subgraph), 
            signature=signature)
    
    #add location_graph_process_hash to base_graph nodes
    #append graph_process_hash and process_hash for each graphofgraph nodes they are in to base_graph nodes
    for u in graphofgraph.nodes():
        subgraph = graphofgraph.nodes[u]['subgraph']
        graph_process_hash = graphofgraph.nodes[u]['graph_process_hash']
        node_unlabelled_graph_process_hash = graphofgraph.nodes[u]['node_unlabelled_graph_process_hash']
        edge_unlabelled_graph_process_hash = graphofgraph.nodes[u]['edge_unlabelled_graph_process_hash']
        unlabelled_graph_process_hash = graphofgraph.nodes[u]['unlabelled_graph_process_hash']
        process_hash = graphofgraph.nodes[u]['process_hash']
        location_graph_process_hash_list = nodes_hash(subgraph, context=graph_process_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False)
        location_node_unlabelled_graph_process_hash_list = nodes_hash(subgraph, context=node_unlabelled_graph_process_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=False)
        location_edge_unlabelled_graph_process_hash_list = nodes_hash(subgraph, context=edge_unlabelled_graph_process_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=True)
        location_unlabelled_graph_process_hash_list = nodes_hash(subgraph, context=unlabelled_graph_process_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=True)
        for node_id, location_graph_process_hash, location_node_unlabelled_graph_process_hash, location_edge_unlabelled_graph_process_hash, location_unlabelled_graph_process_hash in zip(subgraph.nodes(),location_graph_process_hash_list, location_node_unlabelled_graph_process_hash_list, location_edge_unlabelled_graph_process_hash_list, location_unlabelled_graph_process_hash_list):
            base_graph.nodes[node_id]['location_graph_process_hash'].append(location_graph_process_hash)
            base_graph.nodes[node_id]['location_node_unlabelled_graph_process_hash'].append(location_node_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['location_edge_unlabelled_graph_process_hash'].append(location_edge_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['location_unlabelled_graph_process_hash'].append(location_unlabelled_graph_process_hash)
        for node_id in graphofgraph.nodes[u]['subgraph'].nodes():
            base_graph.nodes[node_id]['graph_process_hash'].append(graph_process_hash)
            base_graph.nodes[node_id]['node_unlabelled_graph_process_hash'].append(node_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['edge_unlabelled_graph_process_hash'].append(edge_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['unlabelled_graph_process_hash'].append(unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['process_hash'].append(process_hash)
    graphofgraph.graph['base'] = base_graph
    return graphofgraph


def make_graph_of_graph(base_graph, node_bunches=None, edge_bunches=None, edges=None, node_signatures=[], edge_signatures=[], abstraction_level='graph_process'):
    """
    Make a graph of graph starting from the base graph in 'base_graph' and a list of lists of node ids or list of lists of edges ids. 

    node_bunches: Each list of nodes is used to induce a subgraph to be associated to a node of the graph of graph.
    edge_bunches: Each subgraph can be identified by a list of nodes, or a list of edges (in this case it is a edge induced subgraph).
    edges: Edges can be provided explicitly between nodes of the graph of graph. 
    The label of a node of the graph of graph is computed as a specific permutation invariant hash of the subgraph.
    A 'signature' string is used to seed the hash function so that two isomophic subgraphs that are produced by different procedures get a distinct encoding. 
    """
    if node_bunches is not None:
        return make_subgraphs_graph_of_graph(base_graph, bunches=node_bunches, signatures=node_signatures, subgraph_mode='node', abstraction_level=abstraction_level)
    if edge_bunches is not None:
        return make_subgraphs_graph_of_graph(base_graph, bunches=edge_bunches, signatures=edge_signatures, subgraph_mode='edge', abstraction_level=abstraction_level)
    return graphofgraph


def get_node_bunches(graphofgraph):
    return [list(graphofgraph.nodes[u]['subgraph'].nodes()) for u in graphofgraph.nodes()]


def get_node_subgraphs(graphofgraph):
    return [graphofgraph.nodes[u]['subgraph'] for u in graphofgraph.nodes()]


def get_edge_subgraphs(graphofgraph):
    return [(graphofgraph.nodes[u]['subgraph'],graphofgraph.nodes[v]['subgraph']) for u,v in graphofgraph.edges()]


def construct(graph, attribute_label='vec', nbits=16):
    """
    Construct a graph of graph from a base graph.

    A graph of graph is a graph that has as nodes subgraphs of a base graph and as edges relations between these subgraphs.
    The default constructor builds a graph of graph made of a single node which has a subgraph the whole base graph.
    The attribute_label is the dictionary key that allow access to real valued arrays for each node in the base graph.
    """
    base_graph = nx.Graph(graph)
    base_graph = nx.convert_node_labels_to_integers(base_graph)
    base_graph.graph['nbits'] = nbits
    base_graph.graph['bitmask'] = pow(2, nbits) - 1
    base_graph.graph['feature_size'] = base_graph.graph['bitmask'] + 1
    base_graph.graph['attribute_label'] = attribute_label
    node_bunches = [list(base_graph.nodes())]
    graphofgraph = make_graph_of_graph(base_graph, node_bunches=node_bunches, node_signatures=['base'])
    return graphofgraph


@curry
def atom(graphofgraph, use_nodes=True, use_edges=True):
    signature = function_signature(locals())
    node_bunches = []
    node_signatures = []
    for u in graphofgraph.nodes():
        subgraph = graphofgraph.nodes[u]['subgraph']
        if use_nodes is True:
            for n in subgraph.nodes():
                node_bunches.append([n])
                node_signature = make_signature(underlying_signature=graphofgraph.nodes[u]['signature'], added_signature=signature+'+n')
                node_signatures.append(node_signature)
        if use_edges is True:
            for i,j in subgraph.edges():
                node_bunches.append([i,j])
                node_signature = make_signature(underlying_signature=graphofgraph.nodes[u]['signature'], added_signature=signature+'+e')
                node_signatures.append(node_signature)
    out_graphofgraph = make_graph_of_graph(base_graph=graphofgraph.graph['base'], node_bunches=node_bunches, node_signatures=node_signatures)
    return out_graphofgraph

def get_edges_from_cycle(cycle):
    for i, c in enumerate(cycle):
        j = (i + 1) % len(cycle)
        u, v = cycle[i], cycle[j]
        if u < v:
            yield u, v
        else:
            yield v, u


def get_cycle_basis_edges(g):
    ebunch = []
    cs = nx.cycle_basis(g)
    for c in cs:
        ebunch += list(get_edges_from_cycle(c))
    return ebunch


def edge_complement(g, ebunch):
    edge_set = set(ebunch)
    other_ebunch = [e for e in g.edges() if e not in edge_set]
    return other_ebunch


def edge_subgraph(g, ebunch):
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for u, v in ebunch:
        g2.add_edge(u, v)
        g2.edges[u, v].update(g.edges[u, v])
    return g2


def edge_complement_subgraph(g, ebunch):
    """Induce graph from edges that are not in ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for e in g.edges():
        if e not in ebunch:
            u, v = e
            g2.add_edge(u, v)
            g2.edges[u, v].update(g.edges[u, v])
    return g2


def cycle_basis_and_non_cycle_decomposition(g, min_size=None, max_size=None):
    cs = nx.cycle_basis(g)
    cycle_components = list(map(set, cs))
    if min_size is not None and max_size is not None:
        cycle_components = [cyc for cyc in cycle_components if len(cyc)>= min_size and len(cyc)<= max_size ]
    cycle_ebunch = get_cycle_basis_edges(g)
    g2 = edge_complement_subgraph(g, cycle_ebunch)
    non_cycle_components = nx.connected_components(g2)
    non_cycle_components = [c for c in non_cycle_components if len(c) >= 2]
    non_cycle_components = list(map(set, non_cycle_components))
    if min_size is not None and max_size is not None:
        non_cycle_components = [cyc for cyc in non_cycle_components if len(cyc)>= min_size and len(cyc)<= max_size ]
    return cycle_components, non_cycle_components


@curry
def cycle(graphofgraph, size=None, min_size=None, max_size=None, use_positive=True, use_negative=True, abstraction_level='graph_process'):
    signature = function_signature(locals())
    if min_size is None:
        min_size = size 
    if max_size is None:
        max_size = size
    node_bunches = []
    node_signatures = []
    for u in graphofgraph.nodes():
        subgraph = graphofgraph.nodes[u]['subgraph']
        positive_components, negative_components = cycle_basis_and_non_cycle_decomposition(subgraph, min_size=min_size, max_size=max_size)
        if use_positive:
            node_bunches.extend(positive_components)
            node_signature = make_signature(underlying_signature=graphofgraph.nodes[u]['signature'], added_signature=signature, use_positive=True, use_negative=None)
            node_signatures.extend([node_signature]*len(positive_components))
        if use_negative: 
            node_bunches.extend(negative_components)
            node_signature = make_signature(underlying_signature=graphofgraph.nodes[u]['signature'], added_signature=signature, use_positive=None, use_negative=True)
            node_signatures.extend([node_signature]*len(negative_components))
    out_graphofgraph = make_graph_of_graph(base_graph=graphofgraph.graph['base'], node_bunches=node_bunches, node_signatures=node_signatures, abstraction_level=abstraction_level)
    return out_graphofgraph


def neighborhood_decomposition(graph, cutoff=1):
    node_bunches = []
    for u in graph.nodes():
        ego_graph = nx.ego_graph(graph, u, radius=cutoff)
        node_bunches.append(list(ego_graph.nodes()))
    return node_bunches


@curry
def neighborhood(graphofgraph, size=1, min_size=None, max_size=None):
    signature = function_signature(locals())
    if min_size is None:
        min_size = size 
    if max_size is None:
        max_size = size
    node_bunches = []
    node_signatures = []
    for s in range(min_size, max_size+1):
        for u in graphofgraph.nodes():
            subgraph = graphofgraph.nodes[u]['subgraph']
            components = neighborhood_decomposition(subgraph, cutoff=s)
            node_bunches.extend(components)
            node_signature = make_signature(underlying_signature=graphofgraph.nodes[u]['signature'], added_signature=signature)
            node_signatures.extend([node_signature]*len(components))
    out_graphofgraph = make_graph_of_graph(base_graph=graphofgraph.graph['base'], node_bunches=node_bunches, node_signatures=node_signatures)
    return out_graphofgraph


def invert_dict(mydict):
    reversed_dict = defaultdict(list)
    for key, value in mydict.items():
        reversed_dict[value].append(key)
    return reversed_dict


def get_distances(graph, cutoff=None):
    return {node_id:invert_dict(nx.single_source_shortest_path_length(graph, node_id, cutoff=cutoff)) for node_id in graph.nodes()}


def get_neighborhood(node_id, radius, distances_dict):
    nbunch = []
    for dist in range(radius+1):
        nbunch.extend(distances_dict[node_id][dist])
    return nbunch

@curry
def pairwise_neighborhood(graphofgraph, size=1, min_size=None, max_size=None, distance=0, min_distance=None, max_distance=None):
    signature = function_signature(locals())
    if min_size is None:
        min_size = size 
    if max_size is None:
        max_size = size
    if min_distance is None:
        min_distance = distance 
    if max_distance is None:
        max_distance = distance

    cutoff = max(max_distance, max_size)
    node_bunches = []
    node_signatures = []
    for u in graphofgraph.nodes():
        subgraph = graphofgraph.nodes[u]['subgraph']
        distances_dict = get_distances(subgraph, cutoff)
        for radius in range(min_size, max_size+1):  
            for i in subgraph.nodes():
                neighborhood_i = get_neighborhood(i, radius, distances_dict)
                for dist in range(min_distance, max_distance+1):
                    js = distances_dict[i][dist]
                    for j in js:
                        neighborhood_j = get_neighborhood(j, radius, distances_dict)
                        component = set(neighborhood_i+neighborhood_j)
                        node_bunches.append(component)
                        node_signature = make_signature(underlying_signature=graphofgraph.nodes[u]['signature'], added_signature=signature)
                        node_signatures.append(node_signature)
    out_graphofgraph = make_graph_of_graph(base_graph=graphofgraph.graph['base'], node_bunches=node_bunches, node_signatures=node_signatures)
    return out_graphofgraph
def counts_and_size_to_vec(vec, feature_counts, feature_size_dict):
    features_counter = Counter(feature_counts)
    for feature_id in features_counter:
        vec[0,feature_id] += features_counter[feature_id] * feature_size_dict[feature_id]
    return vec
    

def vectorize_vec(graphofgraphs):
    feature_size = graphofgraphs.graph['base'].graph['feature_size']
    vec = lil_matrix((1, feature_size), dtype=np.int8)

    node_feature_counts = [graphofgraphs.nodes[u]['label'] for u in graphofgraphs.nodes()]
    node_feature_size_dict = {graphofgraphs.nodes[u]['label']:graphofgraphs.nodes[u]['subgraph'].number_of_nodes() for u in graphofgraphs.nodes()}
    vec = counts_and_size_to_vec(vec, node_feature_counts, node_feature_size_dict)
    
    edge_feature_counts = [graphofgraphs.edges[e]['label'] for e in graphofgraphs.edges()]
    edge_feature_size_dict = {graphofgraphs.edges[e]['label']:(graphofgraphs.nodes[e[0]]['subgraph'].number_of_nodes()+graphofgraphs.nodes[e[1]]['subgraph'].number_of_nodes()) for e in graphofgraphs.edges()}
    vec = counts_and_size_to_vec(vec, edge_feature_counts, edge_feature_size_dict)

    vec[0,0] = graphofgraphs.graph['base'].number_of_nodes() #feature 0 encodes the existence of each node
    return vec


def vectorize(graphs, decomposition_function=None, nbits=16):
    mtx = vstack([vectorize_vec(decomposition_function(construct(graph, nbits=nbits))) for graph in graphs])
    return csr_matrix(mtx)


# NOTE: when there is a collision with the hashed edge label, then the encodings in vectorize and node_vectorize can differ, 
# this is because edge_feature_size_dict can store only one value per edge label hash, while in the node_vectorize
# the values are summed up for each colliding edge label hash. In a sufficiently large hash domain, this discrepancy vanishes.


#------------------------------------------------------------------------------------------------------------------
# vectorization of each base node in a graph: the features are the hashed subgraphs which include the given base node and all edge features involving that base node    

def node_vectorize_vec(graphofgraphs):
    base_graph = graphofgraphs.graph['base']
    feature_size = base_graph.graph['feature_size']
    nbits = base_graph.graph['nbits']
    number_of_nodes = base_graph.number_of_nodes()
    mtx = lil_matrix((number_of_nodes, feature_size), dtype=np.int8)
    mtx[:,0] = 1 #feature 0 encodes the existence of a node
    for node_id in base_graph.nodes():
        for feature_id in base_graph.nodes[node_id]['location_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['location_node_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['location_edge_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['location_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['node_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['edge_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['process_hash']: mtx[node_id,feature_id] += 1
    for edge_id in graphofgraphs.edges():
        feature_id = graphofgraphs.edges[edge_id]['label']
        for node_id in graphofgraphs.nodes[edge_id[0]]['subgraph'].nodes(): mtx[node_id,feature_id] += 1
        for node_id in graphofgraphs.nodes[edge_id[1]]['subgraph'].nodes(): mtx[node_id,feature_id] += 1
    return csr_matrix(mtx)


def node_vectorize(graphs, decomposition_function=None, nbits=16):
    list_of_mtx = [node_vectorize_vec(decomposition_function(construct(graph, nbits=nbits))) for graph in graphs]
    return list_of_mtx

def graph_node_vectorize(graphs, decomposition_function=None, nbits=16):
    mtx = vstack([csr_matrix(node_vectorize_vec(decomposition_function(construct(graph, nbits=nbits))).sum(axis=0)) for graph in graphs])
    return csr_matrix(mtx)

#------------------------------------------------------------------------------------------------------------------
# vectorization of each base edge in a graph: features are hashed subgraphs that involve that base edge and all edge features involving that base edge


def edge_vectorize_vec(graphofgraphs):
    feature_size = graphofgraphs.graph['base'].graph['feature_size']
    number_of_edges = graphofgraphs.graph['base'].number_of_edges()
    mtx = lil_matrix((number_of_edges, feature_size), dtype=np.int8)
    mtx[:,0] = 1 #feature 0 encodes the existence of an edge
    for u in graphofgraphs.nodes():
        feature_id = graphofgraphs.nodes[u]['label']
        for edge_id, (edge_u, edge_v) in enumerate(graphofgraphs.nodes[u]['subgraph'].edges()):
            mtx[edge_id,feature_id] += 1
    for e in graphofgraphs.edges():
        feature_id = graphofgraphs.edges[e]['label']
        for edge_id, (edge_u, edge_v) in enumerate(graphofgraphs.nodes[e[0]]['subgraph'].edges()):
            mtx[edge_id,feature_id] += 1
        for edge_id, (edge_u, edge_v) in enumerate(graphofgraphs.nodes[e[1]]['subgraph'].edges()):
            mtx[edge_id,feature_id] += 1
    return csr_matrix(mtx)


def edge_vectorize(graphs, decomposition_function=None, nbits=16):
    list_of_mtx = [edge_vectorize_vec(decomposition_function(construct(graph, nbits=nbits))) for graph in graphs]
    return list_of_mtx


#------------------------------------------------------------------------------------------------------------------
# vectorization of an attributed graph: each feature is associated to the sum of the node attributes in the subgraph. 
# All node features are then summed up to obtain the graph vectorization. 


def get_node_attributes_matrix(graph, attribute_label='vec'):
    """Return a n_nodes x n_attribute_features matrix."""
    vec_list = [graph.nodes[u][attribute_label] for u in graph.nodes() if attribute_label in graph.nodes[u]]
    if len(vec_list) == 0:
        return csr_matrix(np.ones((graph.number_of_nodes(),1)))
    if sp.sparse.issparse(vec_list[0]):
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        attribute_data_matrix = np.vstack(vec_list)
    #determine the dimensionality of the attribute
    n = attribute_data_matrix.shape[1]
    #add a zero vec = [1,0,0,...,0] when attribute is missing
    zero_vec = np.zeros((1,n))
    zero_vec[0,0] = 1
    if sp.sparse.issparse(vec_list[0]):
        vec_list = [graph.nodes[u].get(attribute_label,csr_matrix(zero_vec)) for u in graph.nodes()]
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        vec_list = [graph.nodes[u].get(attribute_label,zero_vec) for u in graph.nodes()]
        attribute_data_matrix = np.vstack(vec_list)

    attribute_data_matrix = csr_matrix(attribute_data_matrix)
    return attribute_data_matrix


def get_edge_attributes_matrix(graph, attribute_label='vec'):
    """Return a n_edges x n_attribute_features matrix."""
    vec_list = [graph.edges[e][attribute_label] for e in graph.edges() if attribute_label in graph.edges[e]]
    if len(vec_list) == 0:
        return csr_matrix(np.ones((graph.number_of_edges(),1)))
    if sp.sparse.issparse(vec_list[0]):
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        attribute_data_matrix = np.vstack(vec_list)
    #determine the dimensionality of the attribute
    n = attribute_data_matrix.shape[1]
    #add a zero vec = [1,0,0,...,0] when attribute is missing
    zero_vec = np.zeros((1,n))
    zero_vec[0,0] = 1
    if sp.sparse.issparse(vec_list[0]):
        vec_list = [graph.edges[e].get(attribute_label,csr_matrix(zero_vec)) for e in graph.edges()]
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        vec_list = [graph.edges[e].get(attribute_label,zero_vec) for e in graph.edges()]
        attribute_data_matrix = np.vstack(vec_list)

    attribute_data_matrix = csr_matrix(attribute_data_matrix)
    return attribute_data_matrix


def structures(graphofgraphs):
    graph = graphofgraphs.graph['base']
    attribute_label = graphofgraphs.graph['base'].graph['attribute_label']
    
    #for nodes
    node_structure_data_matrix = node_vectorize_vec(graphofgraphs) # n_nodes x n_features_sparse
    node_attribute_data_matrix = get_node_attributes_matrix(graph, attribute_label) # n_nodes x n_attribute_features 
    
    #for edges
    edge_structure_data_matrix = edge_vectorize_vec(graphofgraphs) # n_edges x n_features_sparse
    edge_attribute_data_matrix = get_edge_attributes_matrix(graph, attribute_label) # n_edges x n_attribute_features 

    return node_structure_data_matrix, node_attribute_data_matrix, edge_structure_data_matrix, edge_attribute_data_matrix
    
def attributed_vectorize_vec(graphofgraphs):
    node_structure_data_matrix, node_attribute_data_matrix, edge_structure_data_matrix, edge_attribute_data_matrix = structures(graphofgraphs)

    #for nodes
    # sum all attributes vectors for each node in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    node_attributed_data_matrix = node_structure_data_matrix.T.dot(node_attribute_data_matrix) # n_features_sparse x n_attribute_features
    # concatenate all vectors to flatten the representation
    vec_node_attributed_data_matrix = csr_matrix(node_attributed_data_matrix.reshape(1,-1)) # 1 x (n_attribute_features * n_features_sparse)
    
    #for edges
    # sum all attributes vectors for each edge in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    edge_attributed_data_matrix = edge_structure_data_matrix.T.dot(edge_attribute_data_matrix) # n_features_sparse x n_attribute_features
    # concatenate all vectors to flatten the representation
    vec_edge_attributed_data_matrix = csr_matrix(edge_attributed_data_matrix.reshape(1,-1)) # 1 x (n_attribute_features * n_features_sparse)
    
    vec_attributed_data_matrix = sp.sparse.hstack([vec_node_attributed_data_matrix, vec_edge_attributed_data_matrix])

    return vec_attributed_data_matrix


def attributed_vectorize(graphs, decomposition_function=None, nbits=16):
    mtx = vstack([attributed_vectorize_vec(decomposition_function(construct(graph, nbits=nbits))) for graph in graphs])
    return csr_matrix(mtx)


def attributed_nodes_edges_vectorize_vec(graphofgraphs):
    node_structure_data_matrix, node_attribute_data_matrix, edge_structure_data_matrix, edge_attribute_data_matrix = structures(graphofgraphs)

    #for nodes
    # sum all attributes vectors for each node in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    node_attributed_data_matrix = node_structure_data_matrix.T.dot(node_attribute_data_matrix) # n_features_sparse x n_attribute_features
    
    #for edges
    # sum all attributes vectors for each edge in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    edge_attributed_data_matrix = edge_structure_data_matrix.T.dot(edge_attribute_data_matrix) # n_features_sparse x n_attribute_features
    
    return node_attributed_data_matrix, edge_attributed_data_matrix

def attributed_nodes_edges_vectorize(graphs, decomposition_function=None, nbits=16):
    list_of_mtx = [attributed_nodes_edges_vectorize_vec(decomposition_function(construct(graph, nbits=nbits))) for graph in graphs]
    return list_of_mtx

#------------------------------------------------------------------------------------------------------------------
# annotate adds a sparse vector to each node (and edge) corresponding to the graph kernel feature description for that node


def annotate(graphs, decomposition_function=None, nbits=16, attribute_label='vec', concatenate_attributes=True):
    node_mtx_list = node_vectorize(graphs, decomposition_function, nbits)
    edge_mtx_list = edge_vectorize(graphs, decomposition_function, nbits)
    
    
    out_graphs = []
    for graph, node_mtx, edge_mtx in zip(graphs, node_mtx_list, edge_mtx_list):
        if concatenate_attributes:
            node_attribute_data_matrix = get_node_attributes_matrix(graph, attribute_label) # n_nodes x n_attribute_features 
            node_mtx = sp.sparse.hstack([csr_matrix(node_attribute_data_matrix), csr_matrix(node_mtx)]) # n_nodes x (n_attribute_features + n_features_sparse)
            node_mtx = node_mtx.todense().A
            edge_attribute_data_matrix = get_edge_attributes_matrix(graph, attribute_label) # n_edges x n_attribute_features 
            edge_mtx = sp.sparse.hstack([csr_matrix(edge_attribute_data_matrix), csr_matrix(edge_mtx)]) # n_edges x (n_attribute_features + n_features_sparse)
            edge_mtx = edge_mtx.todense().A

        out_graph = nx.Graph(graph)
        for vec,u in zip(node_mtx, graph.nodes()):
            out_graph.nodes[u][attribute_label] = vec.flatten()
        for vec,e in zip(edge_mtx, graph.edges()):
            out_graph.edges[e][attribute_label] = vec.flatten()
        out_graphs.append(out_graph)
    return out_graphs



def parallel_vectorize(graphs, decomposition_function=None, nbits=None):
    def _make_vectorize_func(decomposition_function, nbits):
        def vectorize_func(graphs):
            mtx = vectorize(graphs, decomposition_function, nbits=nbits)
            return mtx
        return vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    vectorize_func = _make_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(vectorize_func, graphs_list)
    pool.close()
    data_mtx = sp.sparse.vstack(results)
    return data_mtx


def parallel_graph_node_vectorize(graphs, decomposition_function=None, nbits=None):
    def _make_vectorize_func(decomposition_function, nbits):
        def vectorize_func(graphs):
            mtx = graph_node_vectorize(graphs, decomposition_function, nbits=nbits)
            return mtx
        return vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    vectorize_func = _make_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(vectorize_func, graphs_list)
    pool.close()
    data_mtx = sp.sparse.vstack(results)
    return data_mtx


def parallel_node_vectorize(graphs, decomposition_function=None, nbits=None):
    def _make_node_vectorize_func(decomposition_function, nbits):
        def node_vectorize_func(graphs):
            mtx = node_vectorize(graphs, decomposition_function, nbits=nbits)
            return mtx
        return node_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    node_vectorize_func = _make_node_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(node_vectorize_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx


def parallel_edge_vectorize(graphs, decomposition_function=None, nbits=None):
    def _make_edge_vectorize_func(decomposition_function, nbits):
        def edge_vectorize_func(graphs):
            mtx = edge_vectorize(graphs, decomposition_function, nbits=nbits)
            return mtx
        return edge_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    edge_vectorize_func = _make_edge_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(edge_vectorize_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx


def parallel_attributed_vectorize(graphs, decomposition_function=None, nbits=16):
    def _make_attributed_vectorize_func(decomposition_function, nbits):
        def attributed_vectorize_func(graphs):
            mtx = attributed_vectorize(graphs, decomposition_function, nbits=nbits)
            return mtx
        return attributed_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    attributed_vectorize_func = _make_attributed_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(attributed_vectorize_func, graphs_list)
    pool.close()
    data_mtx = sp.sparse.vstack(results)
    return data_mtx


def parallel_attributed_nodes_edges_vectorize(graphs, decomposition_function=None, nbits=None):
    def _make_attributed_nodes_edges_vectorize_func(decomposition_function, nbits):
        def attributed_nodes_edges_vectorize_func(graphs):
            list_of_mtx = attributed_nodes_edges_vectorize(graphs, decomposition_function, nbits=nbits)
            return list_of_mtx
        return attributed_nodes_edges_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    attributed_nodes_edges_vectorize_func = _make_attributed_nodes_edges_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(attributed_nodes_edges_vectorize_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx

#------------------------------------------------------------------------------------------------------------------
# annotate adds a sparse vector to each node (and edge) corresponding to the graph kernel feature description for that node


def parallel_annotate(graphs, decomposition_function=None, nbits=16, attribute_label='vec', concatenate_attributes=True):
    node_mtx_list = parallel_node_vectorize(graphs, decomposition_function, nbits)
    edge_mtx_list = parallel_edge_vectorize(graphs, decomposition_function, nbits)
    
    
    out_graphs = []
    for graph, node_mtx, edge_mtx in zip(graphs, node_mtx_list, edge_mtx_list):
        if concatenate_attributes:
            node_attribute_data_matrix = get_node_attributes_matrix(graph, attribute_label) # n_nodes x n_attribute_features 
            node_mtx = sp.sparse.hstack([csr_matrix(node_attribute_data_matrix), csr_matrix(node_mtx)]) # n_nodes x (n_attribute_features + n_features_sparse)
            node_mtx = node_mtx.todense().A
            edge_attribute_data_matrix = get_edge_attributes_matrix(graph, attribute_label) # n_edges x n_attribute_features 
            edge_mtx = sp.sparse.hstack([csr_matrix(edge_attribute_data_matrix), csr_matrix(edge_mtx)]) # n_edges x (n_attribute_features + n_features_sparse)
            edge_mtx = edge_mtx.todense().A

        out_graph = nx.Graph(graph)
        for vec,u in zip(node_mtx, graph.nodes()):
            out_graph.nodes[u][attribute_label] = vec.flatten()
        for vec,e in zip(edge_mtx, graph.edges()):
            out_graph.edges[e][attribute_label] = vec.flatten()
        out_graphs.append(out_graph)
    return out_graphs




def graph_set_similarity(src_graphs, dst_graphs, decomposition_function, nbits=19, metric='cosine', parallel=True):
    if parallel:
        src_embeddings = parallel_vectorize(src_graphs, decomposition_function=decomposition_function, nbits=nbits)
        dst_embeddings = parallel_vectorize(dst_graphs, decomposition_function=decomposition_function, nbits=nbits)
    else:
        src_embeddings = vectorize(src_graphs, decomposition_function=decomposition_function, nbits=nbits)
        dst_embeddings = vectorize(dst_graphs, decomposition_function=decomposition_function, nbits=nbits)
    #remove n nodes feature and just ensure that no two instances are orthogonal because one is null
    src_embeddings[:,0] = 1
    dst_embeddings[:,0] = 1
    Kss = pairwise_kernels(src_embeddings, src_embeddings, metric=metric).mean()
    Kdd = pairwise_kernels(dst_embeddings, dst_embeddings, metric=metric).mean()
    Ksd = pairwise_kernels(src_embeddings, dst_embeddings, metric=metric).mean()
    K_sim = Ksd / np.sqrt(Kss * Kdd)
    return K_sim

@curry
def random_path_graph(n):
    g = nx.path_graph(n)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_tree_graph(n):
    g = nx.random_tree(n)
    g = nx.convert_node_labels_to_integers(g)
    return g

@curry
def random_cycle_graph(n):
    g = nx.random_tree(n)
    terminals = [u for u in g.nodes()if g.degree(u) == 1]
    random.shuffle(terminals)
    for i in range(0, len(terminals), 2):
        e_start = terminals[i]
        if i + 1 < len(terminals):
            e_end = terminals[i + 1]
            g.add_edge(e_start, e_end)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_regular_graph(d, n):
    g = nx.random_regular_graph(d, n)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_degree_seq(n, dmax):
    sequence = np.linspace(1, dmax, n).astype(int)
    g = nx.expected_degree_graph(sequence)
    g = nx.convert_node_labels_to_integers(g)
    return g

@curry
def random_dense_graph(n, m):
    # a graph is chosen uniformly at random from the set of all graphs with n nodes and m edges
    g = nx.dense_gnm_random_graph(n, m)
    max_cc = max(nx.connected_components(g), key=lambda x: len(x))
    g = nx.subgraph(g, max_cc)
    g = nx.convert_node_labels_to_integers(g)
    return g

def make_graph_generator(GRAPH_TYPE, instance_size):
    graph_generator = None
    if GRAPH_TYPE == 'path':
        graph_generator = random_path_graph(n=instance_size)

    if GRAPH_TYPE == 'tree':
        graph_generator = random_tree_graph(n=instance_size)

    if GRAPH_TYPE == 'cycle':
        graph_generator = random_cycle_graph(n=instance_size)

    if GRAPH_TYPE == 'degree':
        n = instance_size
        dmax = 4
        graph_generator = random_degree_seq(n, dmax)
        while nx.is_connected(graph_generator) is not True:
            graph_generator = random_degree_seq(n, dmax)

    if GRAPH_TYPE == 'regular':
        graph_generator = random_regular_graph(d=3, n=instance_size)

    if GRAPH_TYPE == 'dense':
        graph_generator = random_dense_graph(n=instance_size, m=instance_size + instance_size // 2)
        
    assert graph_generator is not None, 'Unknown graph generator type:%s'%GRAPH_TYPE
    return graph_generator

class AttributeGenerator(object):
    def __init__(self, data_mtx, targets):
        self.target_classes = sorted(list(set(targets)))
        self.num_classes = len(self.target_classes)
        self.attributes = [data_mtx[[i for i,y in enumerate(targets) if y==t]] for t in self.target_classes]
    
    def transform(self, class_seq):
        attribute_list = []
        for y in class_seq:
            attributes = self.attributes[y]
            idx = np.random.randint(len(attributes))
            attribute_list.append(attributes[idx].flatten())
        return attribute_list
    
@curry
def make_graph(graph_generator, num_classes, alphabet_size, attribute_generator):
    G = graph_generator
    nx.set_edge_attributes(G, '-', 'label')
    
    labels = np.random.randint(num_classes, size=nx.number_of_nodes(G))
    labels_dict = {node_idx:label for node_idx,label in enumerate(labels)}
    nx.set_node_attributes(G, labels_dict, 'true_label')
    
    labels_dict = {node_idx:label%alphabet_size for node_idx,label in enumerate(labels)}
    nx.set_node_attributes(G, labels_dict, 'label')
    
    if attribute_generator is not None:
        attributes = attribute_generator.transform(labels)
        attributes_dict = {node_idx:attribute for node_idx,attribute in enumerate(attributes)}
        nx.set_node_attributes(G, attributes_dict, 'vec')
    return G.copy()

def link_graphs(graph_source, graph_target, n_link_edges=0):
    n = nx.number_of_nodes(graph_source)
    graph_source_endpoints = np.random.randint(nx.number_of_nodes(graph_source), size=n_link_edges)
    graph_target_endpoints = np.random.randint(nx.number_of_nodes(graph_target), size=n_link_edges)
    graph = nx.disjoint_union(graph_source, graph_target)
    for u,v in graph_source.edges():
        graph.edges[u,v]['true_label'] = 'source'
    for u,v in graph_target.edges():
        graph.edges[u+n,v+n]['true_label'] = 'destination'
    for s,t in zip(graph_source_endpoints, graph_target_endpoints):
        graph.add_edge(s,t+n, label='-', true_label='joint')
    return graph

def make_graphs(graph_generator_target_type, graph_generator_context_type, target_size, context_size, num_classes, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=True):
    context_graphs = []
    for i in range(num_graphs):
        graph_generator = make_graph_generator(graph_generator_context_type, context_size)
        G_context = make_graph(graph_generator, num_classes, alphabet_size, attribute_generator)
        context_graphs.append(G_context.copy())

    if use_single_target:
        graph_generator = make_graph_generator(graph_generator_target_type, target_size)
        G_target = make_graph(graph_generator, num_classes, alphabet_size, attribute_generator)
        target_graphs = [G_target.copy()]*num_graphs
    else:
        target_graphs = []
        for i in range(num_graphs):
            graph_generator = make_graph_generator(graph_generator_target_type, target_size)
            G_target = make_graph(graph_generator, num_classes, alphabet_size, attribute_generator)
            target_graphs.append(G_target.copy())
    
    graphs = [link_graphs(graph_source=G_target, graph_target=G_context, n_link_edges=n_link_edges) for G_target, G_context in zip(target_graphs, context_graphs)]
    return graphs 

def make_dataset(graph_generator_target_type, graph_generator_context_type, target_size, context_size, num_classes, alphabet_size, attribute_generator, n_link_edges, num_graphs):
    pos_graphs = make_graphs(graph_generator_target_type, graph_generator_context_type, target_size, context_size, num_classes, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=True)
    neg_graphs = make_graphs(graph_generator_target_type, graph_generator_context_type, target_size, context_size, num_classes, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=False)
    targets = np.array([1]*len(pos_graphs)+[0]*len(neg_graphs))
    graphs = pos_graphs + neg_graphs
    return graphs, targets, pos_graphs, neg_graphs



@curry
def similarity_analysis(src_graphs, dst_graphs, name_decomposition_list=None):
    for name, df in name_decomposition_list:
        start = time.time()
        score = graph_set_similarity(src_graphs, dst_graphs, decomposition_function=df)
        elapsed = time.time() - start
        print('%30s: %5.3f   [%4.1f s  (%4.1f m)]'%(name, score, elapsed, elapsed/60))



