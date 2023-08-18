"""A module to evaluate performance in network"""

from scipy.stats import uniform
import networkx as nx
import numpy as np
import pandas as pd
import torch

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
        strengths (float array): array of strengths of each node.
            Edge direction always goes from weakest to strongest node. 

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

def generate_fraudstersEdges(G, n, network_strength_scheme,
                               fraud_probability=0.5, fraud_scheme=None,
                               decreasing_function=lambda x : 1/x):
    """Generates a list of fraudulous edges and regular edges for fraudsters in the network

    Args:
        G (nx.Graph): _description_
        n (int): _description_
        strengths (float array): _description_
        fraud_probability (float, optional): _description_. Defaults to 0.5.
    """
    strengths = np.array(list(nx.get_node_attributes(G,"strength").values()))
    list_fraudsters = [x for x, y in G.nodes(data=True) if y["fraud"]]
    for f in list_fraudsters:
        f_strength = G.nodes[f]["strength"]
        f_degree = G.degree[f]
        G.remove_node(f)
        G.add_node(f, strength=f_strength, fraud=True)
        n_fraudulous_links = round(f_degree*fraud_probability)
        n_regular_links = f_degree - n_fraudulous_links
        if fraud_scheme=="strength":
            p = decreasing_function(np.arange(f-1)+1)
            easy_win_neighbours = np.random.choice((np.arange(f-1)+1)[::-1],
                                                   p=p/p.sum(),
                                                   size=n_fraudulous_links, replace=False)
        else:
            easy_win_neighbours = np.random.choice((np.nonzero(strengths<f_strength))[0],
                                                   n_fraudulous_links, replace=False)

        if network_strength_scheme:
            regular_neighbours = choose_newNeighbour(G, f, decreasing_function,
                                                     size=n_regular_links)
        else:
            regular_neighbours = np.random.choice(
                np.nonzero(~np.isin(np.array(range(n)),np.append(easy_win_neighbours, f)))[0],
                n_regular_links, replace=False)

        fraudulous_links = [(source, f) for source in easy_win_neighbours]
        regular_links = list(zip(np.where(strengths[regular_neighbours]>f_strength,
                                          f, regular_neighbours),
                                 np.where(strengths[regular_neighbours]>f_strength,
                                          regular_neighbours, f)))
        G.add_edges_from(fraudulous_links, fraudulous=True)
        G.add_edges_from(regular_links, fraudulous=False)

def generate_network(n,
                     p,
                     distribution=uniform,
                     fraud=True,
                     fraudster_index=None,
                     fraudster_strength=None,
                     fraud_probability=None,
                     strength_scheme=False,
                     iterations=1000,
                     decreasing_function=lambda x : 1/x,
                     fraud_scheme=None):
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
    G.add_edges_from(generate_edges(n, p, strengths), fraudulous=False)
    if strength_scheme:
        rewire_strengthScheme(G, iterations=iterations,
                              decreasing_function=decreasing_function, verbose=False)
    if fraud:
        specify_fraudster(G, index=fraudster_index, strength=fraudster_strength)
        generate_fraudstersEdges(G, n, network_strength_scheme=strength_scheme,
                               fraud_probability=fraud_probability,
                               fraud_scheme=fraud_scheme,
                               decreasing_function=decreasing_function)
    return G


def remove_randomEdge(G, node):
    """Removes a random edge from a node"""
    edges = list(G.out_edges(node)) + list(G.in_edges(node))
    if len(edges)==0:  # if node has no edges
        return None
    rand_int = np.random.randint(len(edges))
    random_edge = edges[rand_int]
    G.remove_edge(*random_edge)
    return random_edge

def choose_newNeighbour(G, node, decreasing_function, size=1):
    """Chooses a new neighbour for a node based on the decreasing function"""
    neighbours = list(G.to_undirected().neighbors(node))
    not_neighbours = [v for v in G.nodes if v not in neighbours + [node]]
    s_strengths = pd.Series(
        {k:v for k,v in dict(G.nodes(data="strength")).items() if k in not_neighbours}
                            ).sort_index()
    node_strength = G.nodes[node]["strength"]
    list_ordered_nodes = s_strengths.index[np.argsort(np.abs(s_strengths.values - node_strength))]
    weights = decreasing_function(np.arange(len(list_ordered_nodes)) + 1)
    return np.random.choice(list_ordered_nodes,
                            p=weights/weights.sum(),
                            size=size, replace=False)  # return new neighbour

def add_newNeighbour(G, node, new_neighbour):
    """Adds a new neighbour to a node"""
    if node > new_neighbour:
        if G.has_edge(new_neighbour, node):
            raise ValueError(f"Edge {new_neighbour, node} already exists")
        G.add_edge(new_neighbour, node, fraudulous=False)
    else:
        if G.has_edge(node, new_neighbour):
            raise ValueError(f"Edge {node, new_neighbour} already exists")
        G.add_edge(node, new_neighbour, fraudulous=False)

def rewire_strengthScheme(G, iterations=1000, decreasing_function=lambda x : 1/x, verbose=False):
    """Rewires a network according to the strength scheme"""
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
        new_neighbour = choose_newNeighbour(G, random_node, decreasing_function)[0]
        if verbose:
            print(f"New neighbour: {new_neighbour}")
        add_newNeighbour(G, random_node, new_neighbour)
        if len(G.edges)!=n_edges:
            raise ValueError("Number of edges changed")

### ====================================================================
### ====================================================================
### Permutation distances
### ====================================================================
### ====================================================================

def kendall_tauDistanceSlow(p1, p2, verbose=False):
    """Slow computation of kendall tau distance between permutation p1 and p2"""
    assert len(p1) == len(p2)
    s = 0
    n = len(p2)
    for i in range(n-1):
        for j in range(i+1, n):
            if verbose:
                print(f"i = {i}\tj={j}")
                print(f"p1(i)={p1[i]}\tp1(j)={p1[j]}")
            x = np.nonzero(p2==p1[i])[0][0]
            y = np.nonzero(p2==p1[j])[0][0]
            if verbose:
                print(f"x={x}\ty={y}")
            if x==y:
                raise ValueError("x should not be equal to y")
            if x>=y:
                s += 1
    return s

def kendall_tauDistance(p1,p2):
    """Computes kendall tau distance between permutations p1 and p2"""
    n = len(p1)
    p2_inv = np.ones(n)
    for i in range(n):
        p2_inv[p2[i]] = i
    p2_inv = p2_inv[p1]
    return sum((p2_inv[i]>p2_inv[i+1:]).sum() for i in range(len(p2_inv)-1))

### ====================================================================
### ====================================================================
### Rating functions and utils of rating functions
### ====================================================================
### ====================================================================

# ----------------------------------------------------------------------
# delta degrees
# ----------------------------------------------------------------------

def rate_deltaDegrees(G):
    """Returns the difference between in and out degrees"""
    return (pd.Series(dict(G.in_degree)) - pd.Series(dict(G.out_degree))).sort_index().values

# ----------------------------------------------------------------------
# f_alpha_t
# ----------------------------------------------------------------------

def ranking_winLossDraw(w, l, d, alpha=0, offset=0):
    """simple ranking based on win/loss/draw informations.

    Args:
        w (int): number of wins
        l (int): number of losses
        d (int): number of draws
        alpha (int, optional): penalization term for the losses in [-1, +inf]. Defaults to 0.
        offset (int, optional): offset to avoid fighters with few matches to be 
            extremely well rated. Defaults to 0.

    Returns:
        _type_: ranking between 0 and 1
    """
    return (w - alpha*l +.5*d - 0.5*alpha*d)/(w+l+d+offset)

def convert_graphToDataFrame(G):
    """Converts a graph generated with generate_network() to a dataframe.
    Specific to f_alpha_t rating function."""
    df = pd.DataFrame([x[1] for x in list(G.nodes(data=True))])
    s = pd.Series([x[0] for x in list(G.nodes(data=True))], name="node")
    df = pd.concat([df, s,
            pd.Series(dict(G.in_degree()), name="wins"),
            pd.Series(dict(G.out_degree()), name="losses")],
            axis=1)
    df["draws"] = 0
    return df

def generate_trainingData(generate_network_function, n=10):
    """Generates training data for the f_alpha_t rating function.

    Args:
        generate_network_function (function): function that generates a network
        n (int, optional): number of independent network generations. Defaults to 10.

    Returns:
        tuple of tensors: X, y where X is the input and y the output of the rating function
            - input : (wins, losses, draws)
            - output : strength
    """
    l_df = []
    for _ in range(n):
        G = generate_network_function()
        df = convert_graphToDataFrame(G)
        l_df.append(df)
    df = pd.concat(l_df)
    return torch.from_numpy(df[["wins","losses","draws"]].values).float(), torch.from_numpy(df["strength"].values).float()  # pylint: disable=no-member

class WinLossDraw(torch.nn.Module):
    """Rating function based on win/loss/draw informations"""
    def __init__(self) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor([1.]))
        self.t = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        """Forward pass"""
        w, l, d = x[:,0], x[:,1], x[:,2]
        h = (w - self.alpha*l +.5*d - 0.5*self.alpha*d)/(w+l+d+self.t)
        return torch.nn.Sigmoid()(h)
