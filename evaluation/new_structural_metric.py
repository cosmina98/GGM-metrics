
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from collections import defaultdict
import networkx as nx
from toolz import curry


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


import networkx as nx
from toolz import curry
from collections import defaultdict

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
import networkx as nx
from toolz import curry


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

import networkx as nx
from toolz import curry


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


@curry
def node(graphofgraph, flag=None):
    return atom(graphofgraph, use_nodes=True, use_edges=False)

@curry
def edge(graphofgraph, flag=None):
    return atom(graphofgraph, use_nodes=False, use_edges=True)
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from toolz import partition_all
import multiprocessing_on_dill as mp
import inspect

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
import numpy as np
import scipy as sp
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
import random
from collections import Counter

class GraphSetDistanceEstimator(object):
    def __init__(self, decomposition_function, nbits=19, metric='cosine', num_iter=5, parallel=True):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.metric = metric
        self.num_iter = num_iter
        self.parallel = parallel
        
    def graph_set_feature_histogram(self, graphofgraphs):
        L = [graphofgraph.nodes[u]['label'] for graphofgraph in graphofgraphs for u in graphofgraph.nodes()]    
        feature_ids, counts = np.unique(L,return_counts=True)
        data = counts
        col = feature_ids
        row = np.zeros(feature_ids.shape)
        hist = csr_matrix((data, (row, col)), shape=(1,2**self.nbits))
        return hist

    def graph_set_distance(self, source_graphofgraphs, destination_graphofgraphs, metric='cosine'):
        source_feature_histogram = self.graph_set_feature_histogram(source_graphofgraphs)
        source_feature_histogram[0,0] = 0 #remove feature that counts the number of nodes
        source_feature_distribution = source_feature_histogram/source_feature_histogram.sum()
        
        destination_feature_histogram = self.graph_set_feature_histogram(destination_graphofgraphs)
        destination_feature_histogram[0,0] = 0 #remove feature that counts the number of nodes
        destination_feature_distribution = destination_feature_histogram/destination_feature_histogram.sum()
        
        distance = pairwise_distances(X=source_feature_distribution, Y=destination_feature_distribution, metric=metric).flatten()[0]
        return distance

    def fit(self, graphs):
        self.graphofgraphs = decomposition(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, parallel=self.parallel)
        distances = [self.random_half_split_self_distance(self.graphofgraphs) for it in range(self.num_iter)]
        self.distance_mean = np.mean(distances)
        self.distance_std = np.std(distances)
        return self
    
    def random_half_split_self_distance(self, graphofgraphs):
        lim = len(graphofgraphs)//2
        random.shuffle(graphofgraphs)
        distance = self.graph_set_distance(graphofgraphs[:lim], graphofgraphs[lim:], metric=self.metric)
        return distance
    
    def estimate(self, graphs):
        graphofgraphs = decomposition(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, parallel=self.parallel)
        distance = self.graph_set_distance(graphofgraphs, self.graphofgraphs, metric=self.metric)
        distance_score = np.absolute(distance-self.distance_mean)/self.distance_std
        return distance_score

def symmetric_graph_set_distance(graphs_1, graphs_2, decomposition_function, nbits=19, metric='cosine', num_iter=5, parallel=True):
    distance_12 = GraphSetDistanceEstimator(decomposition_function=decomposition_function, nbits=nbits, metric=metric, num_iter=num_iter, parallel=parallel).fit(graphs_1).estimate(graphs_2)
    distance_21 = GraphSetDistanceEstimator(decomposition_function=decomposition_function, nbits=nbits, metric=metric, num_iter=num_iter, parallel=parallel).fit(graphs_2).estimate(graphs_1)
    return np.mean([distance_12, distance_21])