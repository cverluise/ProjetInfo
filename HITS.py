import importlib
import pandas as pd
import networkx as nx
import seaborn as sns
import os
import time
import CitNet
from CitNet import HubsAuths as HA
from CitNet import Query as Q
from CitNet import GraphCN


#####################################################################
# HITS
#
# Input: attrs_nos.csv
# Output: - plot of hubs and authorities
#         - global ranking
#####################################################################


#####################################################################
# Section 1. Load data
#####################################################################

# Path to the data
path = os.getcwd() + "/Tables/"
# Load attributes for terms matching
attrs = pd.read_csv(path + "attrs_nos.csv", index_col=0)
# Load refs and cites edges dataframes
cits_edgesdf = pd.read_csv(path + "cits_edges.csv")
refs_edgesdf = pd.read_csv(path + "refs_edges.csv")
# Convert them to list of edges
cits_edges = GraphCN.edgesdf_to_edgeslist(cits_edgesdf)
refs_edges = GraphCN.edgesdf_to_edgeslist(refs_edgesdf)
# Stack refs and cits edges together
all_edges = cits_edges + refs_edges
# Construct nx.DiGraph from stacked edges (refs + cits)
cits_refs_graph = nx.DiGraph(all_edges)


#####################################################################
# Section 2. Test queries on Topics
#####################################################################
# Create the expanded subgraph on which to perform the algo
d = 1000
query_list = ["asymmetry", "trading"]
subtest_topic = Q.topic_query_subgraph(cits_refs_graph, d, attrs, query_list)
# compute hubs and authorities in an iterative fashion
hubs_auths_df = HA.iterate_hubs_auths(subtest_topic, k=1000)
# compute authorities in the eigen vector search fashion
hubs_auths_eig = HA.hubs_authorities_eigen(subtest_topic, neigs=1)
# nodes sorted by authority coef
top_auths_topic = hubs_auths_df.sort_values(by="xauth_0", ascending=False).index
# print(top_auths_topic)
# nodes sorted by "hubness" coef
top_hubs_topic = hubs_auths_df.sort_values(by="xhubs_0", ascending=False).index
# print(top_hubs_topic)
# Draw & print
HA.plot_hubs_authorities(subtest_topic, top_auths_topic, top_hubs_topic)
attrs.loc[top_auths_topic[:10], ["title","authors"]]


#####################################################################
# Section 3. Test queries on Similarity
#####################################################################


# Create the expanded subgraph on which to perform the algo
d = 300
pages = [23721]  # Melitz
subtest_similarity = Q.similarity_query_subgraph(pages, cits_refs_graph, d)
# Compute hubs and authorities using eigenvector approach
hubs_auths_sim = HA.hubs_authorities_eigen(subtest_similarity, neigs=1)
# Top auths of principal vector
top_auths_sim = hubs_auths_sim.sort_values(by="xauth_0", ascending=False).index
# Top hubs of principal vector
top_hubs_sim = hubs_auths_sim.sort_values(by="xhub_0", ascending=False).index
# Draw & print
# HA.plot_hubs_authorities(subtest_similarity, top_auths_sim, top_hubs_sim)
attrs.loc[top_auths_sim[:10], ["title","authors"]]

#####################################################################
# Section 4. Compute Hubs and Authorities on whole Graph
#####################################################################
start = time.clock()
# hubs_auths_whole = hubs_authorities_eigen(cits_refs_graph, neigs=1)
hubs_auths_whole = HA.iterate_hubs_auths(cits_refs_graph, k=1000)
end = time.clock()
print(end - start)
# nodes sorted by authority coef
top_auths_whole = hubs_auths_whole.sort_values(by="xauth_0", ascending=False).index
print(top_auths_whole)
attrs.loc[top_auths_whole[:50], ["title","authors"]]
# nodes sorted by "hubness" coef
top_hubs_whole = hubs_auths_whole.sort_values(by="xhubs_0", ascending=False).index
print(top_hubs_whole)
# Correlation plot between ncitations and authority score
cits_ranks = GraphCN.get_citations_ranking(cits_refs_graph, drop_zeros=True)
auths_ranks = HA.get_top_authorities(hubs_auths_whole, cits_refs_graph, drop_zeroscits=True)
df = pd.DataFrame(columns=["cits_rank", "authority_rank"])
df["cits_rank"] = cits_ranks
df["authority_rank"] = auths_ranks
sns.jointplot("cits_rank", "authority_rank", df, kind="kde")
