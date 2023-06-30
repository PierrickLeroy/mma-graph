"""A module to evaluate performance in network"""

from scipy.stats import uniform
import networkx as nx
import numpy as np
import pandas as pd

def generate_nodes(strengths):
    """generates n nodes from a strength vector of n values 

    Args:
        strengths (array of float): _description_

    Returns:
        nx.Graph: _description_
    """
    n = len(strengths)
    G =  nx.DiGraph()
    G.add_nodes_from(range(n))
    nx.set_node_attributes(G, {k:{"strength":v} for k, v in dict(zip(range(n), strengths)).items()})
    nx.set_node_attributes(G, {k:{"fraud":False} for k in range(n)})
    return G

def specify_fraudster(G, index=0, strength=None):
    """In network G, specify node at index index as a fraudster with a given strength.
    Basically it labels a specific node as a fraudster.

    Args:
        G (nx.Graph): _description_
        index (int, optional): _description_. Defaults to 0.
        strength (floats array, optional): _description_. Defaults to None.
    """
    if strength is not None:
        nx.set_node_attributes(G, {index:{"strength":strength}})
    nx.set_node_attributes(G, {index:{"fraud":True}})

def generate_edges(n, p, strengths):
    """Generate directed edges according to an erdos renyi scheme.

    Args:
        n (int): _description_
        p (float): probability of a random link (undirected)
        strengths (float array): relative strength between a source and target node sets direction of edge
            Direction always goes from weakest to strongest node. 

    Returns:
        array: array of edges
    """
    G = nx.erdos_renyi_graph(n, p)
    df = pd.DataFrame(G.edges, columns=["v1", "v2"]
                    ).merge(pd.Series(strengths, name="v1_strength"), left_on="v1", right_index=True
                   ).merge(pd.Series(strengths, name="v2_strength"), left_on="v2", right_index=True)
    df["target"] = np.where(df["v1_strength"] > df["v2_strength"], df["v1"], df["v2"])
    df["source"] = np.where(df["v1_strength"] > df["v2_strength"], df["v2"], df["v1"])
    return df[["source", "target"]].values

def generate_fraudstersEdges(G, n, strengths, fraud_probability=0.5):
    """Generates a list of fraudulous edges and regular edges for fraudsters in the network

    Args:
        G (nx.Graph): _description_
        n (int): _description_
        strengths (float array): _description_
        fraud_probability (float, optional): _description_. Defaults to 0.5.
    """
    list_fraudsters = [x for x, y in G.nodes(data=True) if y["fraud"]]
    for f in list_fraudsters:
        f_strength = G.nodes[f]["strength"]
        f_degree = G.degree[f]
        G.remove_node(f)
        G.add_node(f, strength=f_strength, fraud=True)
        n_fraudulous_links = round(f_degree*fraud_probability)
        n_regular_links = f_degree - n_fraudulous_links
        easy_win_neighbours = np.random.choice((np.nonzero(strengths<f_strength))[0], n_fraudulous_links, replace=False)
        regular_neighbours = np.random.choice(np.nonzero(~np.isin(np.array(range(n)), np.append(easy_win_neighbours, f)))[0], n_regular_links, replace=False)
        fraudulous_links = [(source, f) for source in easy_win_neighbours]
        regular_links = list(zip(np.where(strengths[regular_neighbours]>f_strength, f, regular_neighbours),
                 np.where(strengths[regular_neighbours]>f_strength, regular_neighbours, f)))
        G.add_edges_from(fraudulous_links, fraudulous=True)
        G.add_edges_from(regular_links, fraudulous=False)

def generate_network(n, p, distribution=uniform, fraud=True, fraudster_index=None, fraudster_strength=None, fraud_probability=None,
                     strength_scheme=False, iterations=1000, decreasing_function=lambda x : 1/x):
    """Generates a network with a fraudster inside it.

    Args:
        n (int): number of nodes in the network
        p (float): probability of a random link (erdos renyi scheme)
        distribution (scipy.stats._continuous_distns, optional): distribution for strengths vector.
        Defaults to uniform.
        fraudster_index (int, optional): _description_. Defaults to 0.
        fraudster_strength (float, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if fraudster_index is None:
        fraudster_index = np.random.choice(n)
    if fraud_probability is None:
        fraud_probability = uniform.rvs()
    strengths = np.sort(distribution.rvs(size=n))
    G = generate_nodes(strengths)
    G.add_edges_from(generate_edges(n, p, strengths))
    if strength_scheme:
        rewire_strengthScheme(G, iterations=iterations,
                              decreasing_function=decreasing_function, verbose=False)
    if fraud:
        specify_fraudster(G, index=fraudster_index, strength=fraudster_strength)
        generate_fraudstersEdges(G, n, strengths, fraud_probability=fraud_probability)
    return G


def remove_randomEdge(G, node):
    edges = list(G.out_edges(node)) + list(G.in_edges(node))
    if len(edges)==0:  # if node has no edges
        return None
    rand_int = np.random.randint(len(edges))
    random_edge = edges[rand_int]
    G.remove_edge(*random_edge)
    return random_edge

def choose_newNeighbour(G, node, decreasing_function):
    neighbours = list(G.to_undirected().neighbors(node))
    not_neighbours = [v for v in G.nodes if v not in neighbours + [node]]
    s_strengths = pd.Series(
        {k:v for k,v in dict(G.nodes(data="strength")).items() if k in not_neighbours}
                            ).sort_index()
    node_strength = G.nodes[node]["strength"]
    list_ordered_nodes = s_strengths.index[np.argsort(np.abs(s_strengths.values - node_strength))]
    weights = decreasing_function(np.arange(len(list_ordered_nodes)) + 1)
    return np.random.choice(list_ordered_nodes, p=weights/weights.sum())  # return new neighbour

def add_newNeighbour(G, node, new_neighbour):
    if node > new_neighbour:
        if G.has_edge(new_neighbour, node):
            raise ValueError(f"Edge {new_neighbour, node} already exists")
        G.add_edge(new_neighbour, node)
    else:
        if G.has_edge(node, new_neighbour):
            raise ValueError(f"Edge {node, new_neighbour} already exists")
        G.add_edge(node, new_neighbour)

def rewire_strengthScheme(G, iterations=1000, decreasing_function=lambda x : 1/x, verbose=False):
    n_edges = len(G.edges)
    for _ in range(iterations):
        random_node = np.random.choice(G.nodes)
        if verbose:
            print(f"Random node: {random_node}")
        random_edge = remove_randomEdge(G, random_node)
        if random_edge is None:
            continue
        if verbose:
            print(f"Random edge dropped: {random_edge}")
        new_neighbour = choose_newNeighbour(G, random_node, decreasing_function)
        if verbose:
            print(f"New neighbour: {new_neighbour}")
        add_newNeighbour(G, random_node, new_neighbour)
        if len(G.edges)!=n_edges:
            raise ValueError("Number of edges changed")
