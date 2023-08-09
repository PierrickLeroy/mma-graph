"""[OUTDATED] [KEPT FOR LEGACY] Simple module to work with mma-graph project"""

import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import percentileofscore

# ========================================================
# =================== Nodes attributes ===================
# ========================================================

def add_nodeAttributes(G, df_fights, df_fighters):
    """Add several node attributes to the graph of fighters

    Args:
        G (nx.Graph): graph of fighters as nodes
        df_fights (DataFrame): contains info about fights
            - "R_fighter", "B_fighter", "weight_class", "R_Weight_lbs", "B_Weight_lbs"
        df_fighters (DataFrame): contains info about fighters
    """

    add_primaryWeightClassAttribute(G, df_fights)
    nx.set_node_attributes(G, dict(df_fighters["fighter"]), name="Name")
    mean_weight = add_meanWeightAttribute(G, df_fights)
    add_meanWeightPercentile(G, mean_weight)

def add_primaryWeightClassAttribute(G, df_fights):
    """_summary_

    Args:
        G (nx.Graph): graph of fighters as nodes
        df_fights (DataFrame): contains history of fights and\
            columns ["R_fighter", "B_fighter", "weight_class"]
    """
    df = pd.melt(df_fights[["R_fighter", "B_fighter", "weight_class"]],
                 id_vars="weight_class"
                 ).drop(columns="variable")
    weight_classes = df.groupby("value").agg(pd.Series.mode).reset_index()["weight_class"]
    weight_classes = weight_classes.apply(lambda u : "-".join(u) if isinstance(u,np.ndarray) else u)
    nx.set_node_attributes(G, dict(weight_classes), name="primary_weightClass")

def add_meanWeightPercentile(G, mean_weight):
    """Adds mean weight percentile to G. mean_weight arg is raw weights"""
    mean_weight_sorted = sorted(mean_weight)
    mean_weight_percentile = mean_weight.apply(lambda x: percentileofscore(mean_weight_sorted, x))
    nx.set_node_attributes(G, dict(mean_weight_percentile), name="mean_weight_percentile")

def add_meanWeightAttribute(G, df_fights):
    """adds mean weight to nodes representing fighters

    Args:
        G (nx.Graph): graph of fighters as nodes
        df_fights (DataFrame): _description_

    Returns:
        _type_: _description_
    """
    df = pd.concat([df_fights[["R_fighter", "R_Weight_lbs"]
                            ].rename(columns={"R_fighter":"fighter", "R_Weight_lbs":"weight_lbs"}),
                    df_fights[["B_fighter", "B_Weight_lbs"]
                            ].rename(columns={"B_fighter":"fighter", "B_Weight_lbs":"weight_lbs"})])
    mean_weight = df.groupby("fighter").agg('mean').reset_index(drop=True)["weight_lbs"]
    mean_weight = mean_weight.fillna(mean_weight.mean())
    nx.set_node_attributes(G, dict(mean_weight), name="mean_weight")
    return mean_weight
