# Projet Python


```shell
├── README.md
├── CitNet
│   ├── DisambName.py
│   ├── GraphCN.py
│   ├── HubsAuths.py
│   ├── PageRank.py
│   ├── Query.py
│   ├── ScrapIR.py
│   ├── Utils.py
│   ├── __pycache__
├── DbScrap.py
├── DisambAuth.py
├── AuthorsGraph.py
├── RefsCitsGraph.py
├── HITS.py
├── DescStat.ipynb
├── Ranking.ipynb
├── Tables
|   └──...
├── Figures
|   └──...
├── Documentation
|   └──...
└── References
    └──...

```
## CitNet Module

**Installation**

```shell
cd CitNet
pip install -e .
pip install -r requirements.txt
...
import CitNet
```
**Documentation**

See [here](Documentation/CitNet.html)

## Scripts

### DbScrap.py

**Purpose**:

Script to scrap database of articles from <https://ideas.repec.org>. Makes use of the url structure of article pages.


| Structure | Example | Content |
|:----------|:--------|:--------|
|*root* | <https://ideas.repec.org/>| Homepage |
|~ + *article* | <https://ideas.repec.org/a/>| List of editors' repositories |
|~ + *editeur* | <https://ideas.repec.org/a/oup/>| List of journalss' repositories |
|~ + *editeur* | <https://ideas.repec.org/a/oup/qjecon/>| List of articles |
|~ + *id*      | <https://ideas.repec.org/a/oup/qjecon/v1y1886i1p1-27..html> | Article's page |

**Output**: 

- `attrs.csv`[^*]: dataset of articles with attributes of interest (authors, date, editor, journal, references, etc). Restriction to articles in top-30 journals since 1880 (IR all time ranking).
- `cits.csv`, `refs.csv`: 

### DisambAuth.py

**Purpose**: 

Script to disambiguate authors' names[^1]. Based on the Levenshtein distance. 

**Output**:

- `Authors.csv`: 2 col dataset with one uniformat names and list of occurences in the scraped dataset
- `attrs.csv`[^*]: adds uniformat authors' list


### AuthorsGraph.py

**Purpose**: 

Script to build graph network of co-authors. G = {V, E} with V = {authors} and E={(author_i, author_j), ...} if i,j co-authored a paper. Undirected (potentially weighted) graph.

**Output**:

- `AdjMat_Auth.npz`: sparse adjacency matrix of co-authors

### RefsCitsGraph

**Purpose**:

Script to create the closed citation network. Baselayer for G={V,E} where V={articles} and E={(article_i, article_j),...} if i cites j. Directed unweighted graph.

**Output**

- `cits_edges.csv`: list of citations
- `refs_edges.csv`: list of edges

### DisambAuth.py

**Purpose**:

Script to disambiguate names[^1] in the raw database. 

**Output**

- `authors.csv`: correspondence table of authors names.

### DescStat.ipynb

**Purpose**:

Notebook to explore the database and explore robustness.

**Output**

- Figures
- Descriptive statistics

### HITS.py

**Purpose**:

Script to implement the HITS algorithm

**Output**

- `attrs.csv`: adds HITS global ranking
- Figures

### Ranking.ipynb

**Purpose**:

Notebook to implement various exploration on rankings.

**Output**:

- `df_CitNet.csv`: final dataset


[^*]: Use `encoding=="ISO-8859-1", index_col=0` for loading

[^1]: Names in the raw dataset might appear under different formats. For example, Joseph Stiglitz appeared under no less that twenty aliases, sometimes including typos, such as J. Stiglitz, J. E. Stiglitz, Joseph Stiglitz etc.  

