"""module for scholar network science analysis"""

import datetime
import itertools
import math
from scholarly import scholarly
import pandas as pd
import numpy as np
import networkx as nx

###############################
############ UTILS ############
###############################

def as_2digits(n):
    """convert month, days.. to 2 digits"""
    return f"0{str(n)}" if len(str(n))==1 else str(n)

def get_currentDateFormated():
    """returns a formatted date representing now()"""
    d = datetime.datetime.now()
    l = [d.year, d.month, d.day, d.hour, d.minute]
    l = [as_2digits(x) for x in l]
    return "".join(l)

###############################
############ MAIN #############
###############################

####### GLOBAL VARIABLES ######

d_aliasAuthors = {  "Will Hamilton":"William L Hamilton",
                    "William Leif Hamilton":"William L Hamilton",
                    "William Hamilton":"William L Hamilton",
                    "Jurij Leskovec":"Jure Leskovec",
                    "Michael M Bronstein":"Michael Bronstein",
                    "M Bronstein":"Michael Bronstein",
                    "MM Bronstein":"Michael Bronstein",
                    "Albert-Laszlo Barabasi":"Albert-László Barabási",
                    "AL Barabasi":"Albert-László Barabási",
                    "A-L Barabási":"Albert-László Barabási",
                    "Albert‐László Barabási":"Albert-László Barabási",
                    "A.-L. Barabási":"Albert-László Barabási",
                    "Albert-Laszlo Barabâsi":"Albert-László Barabási",
                    "T Bonald":"Thomas Bonald"}

############ AUTHOR ###########

def add_author(df_authors, author_id):
    """adds an author specified by author_id to df_authors and returns it"""
    author = scholarly.search_author_id(author_id)
    author_filled = scholarly.fill(author)
    df = pd.DataFrame([author_filled])
    df["selected"] = True
    return pd.concat([df_authors, df]).reset_index(drop=True)

# add/remove orgs to authors

def set_authorCurrentOrg(df_authors, name, org):
    """adds current organization to author"""
    fill_initSecondaryOrgs(df_authors)
    df_authors.loc[df_authors["name"]==name, "primary_org"] = org
    if df_authors.loc[df_authors["name"]==name, "secondary_orgs"].isna().any():
        df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0] = [org]
    elif isinstance(df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0], list):
        if org not in df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0]:
            df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0].append(org)

def add_authorPastOrg(df_authors, name, org):
    """adds current organization to author"""
    fill_initSecondaryOrgs(df_authors)
    if df_authors.loc[df_authors["name"]==name, "secondary_orgs"].isna().any():
        df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0] = [org]
    if org not in df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0]:
        df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0].append(org)

def delete_authorPastOrg(df_authors, name, org):
    """deletes an org from an auhtor's list of orgs"""
    df_authors.loc[df_authors["name"]==name, "secondary_orgs"].iloc[0].remove(org)

def fill_initSecondaryOrgs(df_authors):
    """initialization of secondary orgs columns to []"""
    df_authors["secondary_orgs"] = df_authors["secondary_orgs"].apply(
        lambda u : [] if (not(isinstance(u, list)) and math.isnan(u)) else u)

def create_mappingTableAuthors(df_authors):
    """Returns a mapping table and verify unicity of query_names"""
    df_res = pd.concat([df_authors[['query_name', "name"]].dropna().drop_duplicates(),
                    pd.DataFrame([{"query_name":k, "name":v}for k, v in d_aliasAuthors.items()])])
    if df_res.groupby("query_name").size().max() != 1:
        raise KeyError("There is a query name that lead to different authors name")
    return df_res.reset_index(drop=True)

######### PUBLICATIONS #########

def compute_dfPublications(df_authors):
    """computes df_publications from df_authors' publications column"""
    df_publications = pd.DataFrame(
        data=list(itertools.chain.from_iterable(list(df_authors["publications"].values))))
    df_publications["pub"] = list(
        itertools.chain.from_iterable(list(df_authors["publications"].values)))
    df_publications["title"] = df_publications["bib"].apply(lambda u : u.get("title"))
    df_publications["pub_year"] = df_publications["bib"].apply(lambda u : u.get("pub_year"))
    df_publications["citation"] = df_publications["bib"].apply(lambda u : u.get("citation"))
    df_publications = df_publications.drop(
        columns=["container_type", "source", "bib",
                 "filled", "public_access", "citation", "citedby_url", "cites_id"])
    df_publications["author"] = df_publications["author_pub_id"].apply(lambda u : u.split(":")[0])
    df_publications["id"] = df_publications["author_pub_id"].apply(lambda u : u.split(":")[-1])
    df_publications["rank_from_author"] = df_publications.groupby("author")["num_citations"
                                                                    ].rank("first", ascending=False)
    df_publications["n_pub_from_author"] = df_publications.groupby("author")["num_citations"
                                                                    ].transform(len)
    return df_publications.reset_index(drop=True)

def filter_dfPublications(df_publications):
    """filtering df_publications before requesting to fill informations"""
    df_publications = df_publications[df_publications["num_citations"]>=10]
    df_publications = df_publications[df_publications["rank_from_author"]<=50]
    df_publications = df_publications.drop_duplicates(subset=["author", "title"], keep="first")
    return df_publications.reset_index(drop=True)

def get_newPublications(df_publications, df_authors):
    """Returns publications that are not in df_publications \
        but from authors in df_authors"""
    df = df_authors[df_authors["selected"]].copy()
    df = filter_dfPublications(compute_dfPublications(df))
    df = df[~(df["title"].isin(df_publications["title"]))]
    return df

######### PUBLICATIONS->AUTHORS #########

def get_newAuthorsList(df_publications, df_authors):
    """_summary_

    Args:
        df_publications (_type_): _description_
        df_authors (_type_): _description_

    Returns:
        _type_: _description_
    """
    s = set(map(lambda u : d_aliasAuthors[u] if u in d_aliasAuthors else u, {x for u in df_publications["authors"].values for x in u}))
    s = [x for x in s if (x not in df_authors["name"].values and x not in df_authors["query_name"].values)]
    return s

def update_dfAuthors(df_authors, new_authors_data):
    """_summary_

    Args:
        df_authors (_type_): _description_
        new_authors_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(new_authors_data)==0:
        print("No new authors, nothing to update")
        return df_authors
    df_new_authors = pd.DataFrame(new_authors_data)
    df_new_authors["selected"] = False
    return pd.concat([df_authors, df_new_authors]).reset_index(drop=True).drop_duplicates()

####### GRAPH ######



# add authors nodes with attributes
def add_nodesAuthorsWithAttributes(G, df_authors):
    """Add author nodes to a graph G with attributes, mainly number of citations
    Args:
        G (nx.Graph): Graph
        df_authors (DataFrame): DataFrame with one row per author. Column "node"\
            is used instead of name
    """
    df = df_authors.drop_duplicates(subset="node")[["node", "citedby", "selected"]]
    df["bipartite"] = np.where(df["selected"], "selected_author", "author")
    df["citedby"] = df["citedby"].fillna(0)
    G.add_nodes_from(df["node"])
    d = dict(zip(df["node"], df[["citedby", "bipartite"]].values))
    attrs = {k:{"citedby":v[0], "bipartite":v[1]} for k, v in d.items()}
    nx.set_node_attributes(G, attrs)

# add publications with attributes
def add_nodesPublicationsWithAttributes(G, df_publications):
    """Add publications nodes to a nx.Graph G from the DataFrame of publications"""
    df = df_publications[["title", "num_citations"]].copy()
    df["bipartite"] = "publication"
    G.add_nodes_from(df["title"])
    d = dict(zip(df["title"], df[["num_citations", "bipartite"]].values))
    attrs = {k:{"num_citations":v[0], "bipartite":v[1]} for k, v in d.items()}
    nx.set_node_attributes(G, attrs)

# add organisations
def add_nodesOrganisationWithAttributes(G, df_authors):
    """Add organisation nodes to a nx.Graph G from the DataFrame of authors"""
    orgs = set().union(*df_authors[~df_authors["primary_org"].isna()]\
                       .apply(
                           lambda u : (set([u["primary_org"]] + u["secondary_orgs"])), axis=1
                           ).tolist())
    G.add_nodes_from(orgs)
    attrs = {k:{"bipartite" : "organisation"} for k in orgs}
    nx.set_node_attributes(G, attrs)

# add author-pub links
def add_edgesAuthorsPublications(G, df_publications, df_authors):
    """Adds edges that represent authorship to G.\
        If authors or publication title is unknown (not in G), throws an error."""
    mapping_t = create_mappingTableAuthors(df_authors)
    df_edges_title_author = pd.melt(
        pd.merge(df_publications[["title"]], pd.DataFrame(df_publications["authors"].tolist()),
                how="left", right_index=True, left_index=True),
        id_vars="title"
        ).drop(columns="variable").dropna(how="any").rename(columns={"value":"author"}
                                                            ).reset_index(drop=True)
    df_edges_title_author = pd.merge(df_edges_title_author, mapping_t,
            how="left", left_on="author", right_on="query_name")
    df_edges_title_author["node"] = df_edges_title_author["name"]\
                                        .combine_first(df_edges_title_author["author"])
    df_edges_title_author = df_edges_title_author[["title", "node"]]
    edges = df_edges_title_author.values.tolist()
    for e in edges[:]:
        if G.has_node(e[1]) and G.has_node(e[0]):
            G.add_edge(*e)
        else:
            raise ValueError(f"Node : {e[1]} is not in G")

# add author-orgs links
def get_authorsOrgsDataframe(df_authors, primary_weight):
    """Compute author-organisation-weight DataFrame

    Args:
        df_authors (DataFrame)
        primary_weight (int): how important is primary org compared to past orgs
    """
    df = df_authors[~df_authors["primary_org"].isna()].copy()
    df = pd.merge(df["node"], pd.DataFrame(
        df.apply(\
            lambda u : [u["primary_org"]]*(primary_weight-1) +\
                list(set([u["primary_org"]] + u["secondary_orgs"])),
            axis=1).tolist()),
            how="left", left_index=True, right_index=True)
    df = pd.melt(df, id_vars="node"
                    ).drop(columns="variable").dropna().rename(columns={"value":"organisation"})
    df["weight"] = 1
    df = df.groupby(["node", "organisation"]).agg('sum').reset_index()
    return df

def add_edgesAuthorsOrganisation(G, df_authors, primary_weight=3):
    """Adds memberships of authors to organisation if both authors and org are\
        in the graph, else throws an error.

    Args:
        G (nx.Graph): already contains authors and organisation nodes
        df_authors (DataFrame): df of authors
        primary_weight (int, optional): how important is primary org compared to other orgs.\
            Defaults to 3.

    Raises:
        ValueError: one of the nodes (author, org) is not in the graph
    """
    df = get_authorsOrgsDataframe(df_authors, primary_weight)
    edges = df.values.tolist()
    for u,v,w in edges:
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, weight=w)
        else:
            raise ValueError(f"Node : {u} or {v} is not in G")

### Size of nodes computation

# authors
def compute_sizeAuthorsCitations(G, df_authors):
    """Computes an attribute 'size' based on number of citations."""
    df = df_authors[["node", "citedby"]].copy()
    df["citedby"] = df["citedby"].fillna(0)
    attrs = {k:{"size_citations":v} for k,v in dict(df.values).items()}
    nx.set_node_attributes(G, attrs)

def compute_sizePublicationsCitations(G, df_publications):
    """Computes an attribute 'size' based on number of citations."""
    df = df_publications[["title", "num_citations"]].copy()
    df["num_citations"] = df["num_citations"].fillna(0)
    attrs ={k:{"size_citations":v} for k,v in dict(df.values).items()}
    nx.set_node_attributes(G, attrs)


# compute size of org citations
def compute_sizeOrganisationsCitations(G, df_authors, primary_weight):
    """_summary_

    Args:
        G (nx.Graph): graph containing organisation nodes (as name)
        df_authors (DataFrame)
        primary_weight (scalar): how important is primary org compared to past orgs
    """
    df = get_authorsOrgsDataframe(df_authors, primary_weight=primary_weight)
    df = pd.merge(df, df_authors[["node","citedby"]],
            how="left", on="node")[["organisation", "weight", "citedby"]]
    df["size_citations"] = df["weight"] / primary_weight * df["citedby"]
    df = df.groupby("organisation").agg({"size_citations":"sum"}).reset_index()
    attrs = {k : {"size_citations":v} for k, v in dict(zip(df["organisation"], df["size_citations"])).items()}
    nx.set_node_attributes(G, attrs)
