#!python
# -*-coding:utf-8 -*

"""This module provides tools for computing Hubs and Authorities in a directed Graph"""

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def iterate_hubs_auths(subgraph, k=20):
    """
    Compute hubs and authorities coefficients the iterative way

    :param subgraph: (networkx.classes.digraph.DiGraph) a subgraph
    :param k: (int) number of iterations
    :return: x, y, nodes respectively vector of authorities coefs, hubs coefs and nodes
    ordering used for computations
    """
    nodes = list(subgraph)
    nodes.sort()
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=nodes)  # .asfptype()
    n = len(nodes)
    y = np.ones((n, ))
    x = np.ones((n, ))
    for i in range(0, k):
        x = sparse.csr_matrix.dot(sparse.csr_matrix.transpose(A), y)
        y = sparse.csr_matrix.dot(A, x)
        x *= (1 / np.linalg.norm(x))
        y *= (1 / np.linalg.norm(y))
    results_df = pd.DataFrame(index=nodes)
    results_df["xauth_0"] = x
    results_df["xhubs_0"] = y
    return results_df


def compute_authorities(subgraph, neigs=1):
    """
    Compute authorities coefficients the eigen vectors way

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param neigs: number of principal vectors wanted
    :return: xstar, nodes : respectively eigen vector stacked as columns and nodes ordering used for computations
    """
    nodes = list(subgraph.nodes())
    nodes.sort()
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=nodes).asfptype()
    AT = sparse.csr_matrix.transpose(A)
    ATA = sparse.csr_matrix.dot(AT, A)
    accept = False
    # Sometimes the sparse eigen solver does not converge to the right solution
    # Yielding a principal vector with very very small (order 1e-18), all negative components
    # When this is the case the result is rejected and the solver is launched again
    while not accept:
        w, xstar = sparse.linalg.eigs(ATA, k=neigs, which="LM")
        xstar = np.real(xstar)
        xstar[np.abs(xstar) < 1e-10] = 0
        accept = True
        for i in range(0, neigs):
            if np.all(xstar[:, i] <= 0):
                accept = False
    return xstar, nodes


def compute_hubs(subgraph, neigs=1):
    """
    Compute hubs coefficients the eigen vectors way

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param neigs: number of principal vectors wanted
    :return: xstar, nodes : respectively eigen vectors stacked as columns and nodes ordering used for computations
    """
    nodes = list(subgraph.nodes())
    nodes.sort()
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=nodes).asfptype()
    AT = sparse.csr_matrix.transpose(A)
    AAT = sparse.csr_matrix.dot(A, AT)
    accept = False
    # Sometimes the sparse eigen solver does not converge to the right solution
    # Yielding a principal vector with very very small (order 1e-18), all negative components
    # When this is the case the result is rejected and the solver is launched again
    while not accept:
        w, ystar = sparse.linalg.eigs(AAT, k=neigs, which="LM")
        ystar = np.real(ystar)
        ystar[np.abs(ystar) < 1e-10] = 0
        accept = True
        for i in range(0, neigs):
            if np.all(ystar[:, i] <= 0):
                accept = False
    return ystar, nodes


def hubs_authorities_eigen(subgraph, neigs=1):
    """
    Wraps the result from compute_hubs and compute_authorities functions in a dataframe

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param neigs: number of principal vectors wanted
    :return: Dataframe containing the principal vectors, indexed with nodes
    """
    xstar, nodes = compute_authorities(subgraph, neigs)
    ystar, nodes = compute_hubs(subgraph, neigs)
    results_df = pd.DataFrame(index=nodes)
    for i in range(0, neigs):
        results_df["xauth_" + str(i)] = xstar[:, i]
        results_df["xhub_" + str(i)] = ystar[:, i]
    return results_df


def non_principal_authorities(eigs_vecs_df, c):
    """
    Find c top authorities from set of non principal eigen vectors

    :param eigs_vecs_df: DataFrame, should have 2 columns :
    the i-th eigen vector of ATA and the i-th eigen vector of AAT
    :param c: Number of authorities to retrieve
    :return: Top c authorities
    """
    x = eigs_vecs_df.iloc[0].as_matrix()
    y = eigs_vecs_df.iloc[1].as_matrix()
    xy = np.concatenate((x, y))
    conc_index = eigs_vecs_df.index.append(eigs_vecs_df.index)
    cmax_inds = np.argsort(xy)[::-1][:c]
    cmax = conc_index[cmax_inds]
    return cmax


def non_principal_hubs(eigs_vecs_df, c):
    """
    Find c top hubs from set of non principal eigen vectors

    :param eigs_vecs_df: DataFrame, should have 2 columns :
    the i-th eigen vector of ATA and the i-th eigen vector of AAT
    :param c: Number of authorities to retrieve
    :return: Top c hubs
    """
    x = eigs_vecs_df.iloc[0].as_matrix()
    y = eigs_vecs_df.iloc[1].as_matrix()
    xy = np.concatenate((x, y))
    conc_index = eigs_vecs_df.index.append(eigs_vecs_df.index)
    cmax_inds = np.argsort(xy)[:c]
    cmax = conc_index[cmax_inds]
    return cmax


def get_zero_cits_nodes(graph):
    """
    Get nodes that have 0 citations

    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :return: pandas integer index, the nodes that gets zero citations
    """
    indegrees = dict(graph.in_degree())
    ncits_df = pd.DataFrame.from_dict(data=indegrees, orient="index")
    ncits_df.rename(columns={0: "ncits"}, inplace=True)
    return ncits_df[ncits_df["ncits"] == 0].index


def get_top_authorities(hubs_auths_df, graph=None, drop_zeroscits=False):
    """
    Get top authorities with possibility to eliminate the nodes
    that gets 0 citations from the ranking

    :param hubs_auths_df: DataFrame having columns ["xauth_0", "xhub_0"]
    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param drop_zeroscits: (bool), should nodes with 0 citations be included in the ranking
    :return: pandas Series, index is the node, the value is the rank
    """
    if not drop_zeroscits:
        top_auths = hubs_auths_df.sort_values(by="xauth_0", ascending=False)["xauth_0"]
    else:
        cits_nodes = hubs_auths_df.index.difference(get_zero_cits_nodes(graph))
        top_auths = hubs_auths_df.loc[cits_nodes, :].sort_values(by="xauth_0", ascending=False)["xauth_0"]
    auths_ranks = pd.Series(index=top_auths.index, data=range(0, len(top_auths)))
    auths_ranks.sort_index(inplace=True)
    return auths_ranks


def plot_hubs_authorities(subgraph,
                          auths_rank,
                          hubs_rank,
                          kauths=5,
                          khubs=5,
                          layout=nx.spring_layout,
                          other_authorities=None,
                          other_hubs=None):
    """
    Returns a plot featuring hubs and authorities.

    :param subgraph: (networkx.classes.digraph.DiGraph)
    :param auths_rank:
    :param hubs_rank:
    :param kauths: (int) number of authorities to be shown in special style
    :param khubs: (int) number of hubs to be shown in special style
    :param layout: networkx.layout
    :param other_authorities:
    :param other_hubs:
    :return: Plot of the graph featuring hubs and authorities
    """
    pos = layout(subgraph)
    nx.draw_networkx_nodes(subgraph, pos,
                           nodelist=list(auths_rank[kauths:]),
                           node_color='C0',
                           node_size=75)  # alpha=0.8)
    nx.draw_networkx_nodes(subgraph, pos,
                           nodelist=list(auths_rank[:kauths]),
                           node_color='C1',
                           node_size=150,
                           label="Top authorities")  # alpha=0.8)
    nx.draw_networkx_nodes(subgraph, pos,
                           nodelist=list(hubs_rank[:khubs]),
                           node_color='C2',
                           node_size=150,
                           label="Top hubs")  # alpha=0.8)
    if other_authorities:
        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=other_authorities,
                               node_color='C3',
                               node_size=150,
                               label="Non principal authorities (2nd largest eigen value)")
    if other_hubs:
        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=other_hubs,
                               node_color='C4',
                               node_size=150,
                               label="Non principal hubs (2nd largest eigen value)")
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)
    plt.legend()
