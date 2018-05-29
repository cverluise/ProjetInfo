#!python
# -*-coding:utf-8 -*

"""This module provides tools for querying our database"""

import numpy as np


def topic_query(df, query_list, search_in=("title", "keywords")):
    """
    Return indexes corresponding to a given query

    :param df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_list: (list) list of keywords
    :param search_in: (tuple) The columns to include for the query

    :return: (list) the list of result indexes, one per column in search_in
    """
    inds = []
    for col in search_in:
        first = True
        for query in query_list:
            bool_ind = df[col].str.contains(query, regex=False)
            bool_ind = bool_ind.replace(np.nan, False)
            if first:
                inds_query = df[bool_ind].index
                first = False
            else:
                inds_query = inds_query.intersection(df[bool_ind].index)
        inds.append(inds_query)
    return inds


def indexlist_inter(indexlist):
    """
    Intersection of all elements of a list of indexes

    :param indexlist: (list) the list of pandas indexes (pandas.indexes.range.RangeIndex)

    :return: (pandas.indexes.range.RangeIndex) The intersected index
    """
    first = True
    for ind in indexlist:
        if first:
            inter = ind
            first = False
        else:
            inter = inter.intersection(ind)
    return inter


def indexlist_union(indexlist):
    """
    Union of all elements of a list of indexes

    :param indexlist: (list) the list of pandas indexes (pandas.indexes.range.RangeIndex)

    :return: The union index (pandas.indexes.range.RangeIndex)
    """
    first = True
    for ind in indexlist:
        if first:
            inter = ind
            first = False
        else:
            inter = inter.union(ind)
    return inter


def topic_subgraph_root(df, query_list, search_in=("title", "keywords"), how="union"):
    """
    Root nodes for building a subgraph relevant to a topic based query

    :param df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_list: (list) list of keywords
    :param search_in: (tuple) The columns to include for the query
    :param how: (str) How to join the indexes in the list ?

    :return: the index of the nodes (pandas.indexes.range.RangeIndex)
    """
    inds = topic_query(df, query_list, search_in)
    if how == "inter":
        return indexlist_inter(inds)
    else:
        return indexlist_union(inds)


def similarity_subgraph_root(nodes_list, graph):
    """
    Root nodes for building a subgraph relevant to similarity query

    :param nodes_list: the list of articles of our similar to request
    :param graph: (networkx.classes.digraph.DiGraph) the graph

    :return list of root nodes for our similarity request
    """
    root_nodes = []
    for node in nodes_list:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        root_nodes += successors + predecessors
    return list(set(root_nodes))


def expand_root(root_nodes, graph, d):
    """
    Expand root nodes by including their successors and some of their predecessors
    (d to be exact)

    :param root_nodes: (list-like) the roots nodes
    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param d: how many predecessors to include at most ?

    :return: the expanded nodes list (list).
    """
    nodes = []
    all_nodes = set(graph.nodes())
    root_nodes = set(root_nodes).intersection(all_nodes)
    for node in root_nodes:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        if len(predecessors) >= d:
            np.random.shuffle(predecessors)
            new_nodes = set(successors + predecessors[0: d])
        else:
            new_nodes = set(successors + predecessors)
        nodes += new_nodes
    return nodes


def topic_query_subgraph(graph, d, df, query_list, search_in=("title", "keywords"), how="union"):
    """
    Find expanded subgraph for a topic query

    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param d: how many predecessors to include at most ?
    :param df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_list: (list) List of keywords
    :param search_in: (tuple) The columns to include for the query
    :param how: (str) How to join the indexes in the list ?

    :return: (networkx.classes.digraph.DiGraph) the expanded subgraph for the query
    """
    root_nodes = topic_subgraph_root(df, query_list, search_in, how)
    expanded = expand_root(root_nodes, graph, d)
    return graph.subgraph(expanded)


def similarity_query_subgraph(nodes_list, graph, d):
    """
    Find expanded subgraph for a similarity query

    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param d: how many predecessors to include at most ?
    :param nodes_list: the list of articles of our similar to request

    :return: (networkx.classes.digraph.DiGraph) the expanded subgraph for the similarity query
    """
    root_nodes = similarity_subgraph_root(nodes_list, graph)
    expanded = expand_root(root_nodes, graph, d)
    return graph.subgraph(expanded)
