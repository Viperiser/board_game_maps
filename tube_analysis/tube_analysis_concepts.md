Here you go. Clean, repo-ready markdown.

---

````md
# London Underground Graph Analysis Ideas

This document outlines potential analyses and experiments using the canonical Underground station graph.

---

## 1. Station Importance (Centrality)

Different centrality measures capture different notions of “importance”.

### Betweenness Centrality

- Measures how often a station lies on shortest paths
- Identifies **choke points**

### Closeness Centrality

- Measures average distance to all other stations
- Identifies globally well-connected stations

### Degree

- Number of adjacent stations
- Highlights major interchanges

```python
import networkx as nx

bet = nx.betweenness_centrality(G)
close = nx.closeness_centrality(G)
deg = dict(G.degree())

sorted(bet.items(), key=lambda x: x[1], reverse=True)[:10]
```
````

Expected high-ranking stations:

- King’s Cross St. Pancras
- Oxford Circus
- Green Park
- Bank-Monument

---

## 2. Network Robustness (“What breaks London?”)

### Random Failure

```python
import random

G2 = G.copy()
nodes = list(G2.nodes())
random.shuffle(nodes)

for i, n in enumerate(nodes):
    G2.remove_node(n)
    largest = len(max(nx.connected_components(G2), key=len))
    print(i, largest)
```

### Targeted Attack (by centrality)

```python
G2 = G.copy()

for n, _ in sorted(bet.items(), key=lambda x: x[1], reverse=True):
    G2.remove_node(n)
    largest = len(max(nx.connected_components(G2), key=len))
    print(n, largest)
```

Insight:

- The network is **robust to random failure**
- But **fragile to targeted removal**

---

## 3. Shortest Paths vs Fewest Changes

Current graph:

- Optimises **fewest stations**

Real journeys:

- Often optimise **fewest line changes**

### Approach:

Build a new graph:

- Nodes: `(station, line)`
- Edges:
  - Along line → cost = 1
  - Line change → cost = penalty (e.g. 5)

```python
nx.shortest_path(G, "Amersham", "Morden")
```

Compare:

- naive shortest path
- line-aware path

---

## 4. Line Overlap Analysis

Edges may belong to multiple lines.

```python
edges_df[edges_df["n_lines"] > 1]
```

This reveals:

- Shared infrastructure
- Line entanglement

### Line Interaction Graph

- Nodes: lines
- Edge if two lines share a segment

---

## 5. Cycle Analysis

```python
cycles = nx.cycle_basis(G)
len(cycles)
```

Insights:

- Identifies loops and redundancy
- Not just the Circle line

---

## 6. Diameter (“Size of the Network”)

```python
nx.diameter(G)
nx.periphery(G)
```

Answers:

- Longest shortest journey
- Most distant stations

---

## 7. Bipartiteness

```python
nx.is_bipartite(G)
```

Result:

- False (due to odd cycles)

Optional:

```python
nx.find_cycle(G)
```

---

## 8. Community Detection

```python
import networkx.algorithms.community as nx_comm

communities = nx_comm.greedy_modularity_communities(G)
```

Expected clusters:

- West London
- Central
- East London
- Northern suburbs

Observation:

- Geography emerges from topology

---

## 9. Line Reconstruction

Given:

- Graph + edge metadata

Try:

- Reconstruct original lines

This tests:

- Whether the data representation is complete

---

## 10. Visualisation

Basic layout:

```python
import matplotlib.pyplot as plt

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_size=10, with_labels=False)
plt.show()
```

Note:

- Result will not resemble the Tube map
- Useful only for structural intuition

---

## 11. Graph-Theoretic Properties

### Connectivity

```python
nx.node_connectivity(G)
nx.edge_connectivity(G)
```

### Planarity

```python
nx.check_planarity(G)
```

Result:

- Non-planar (contains a Kuratowski subgraph)

### Possible extensions

- Treewidth
- Minimum cut sets

---

## 12. Advanced Question

> What is the smallest set of stations whose removal makes the graph planar?

Approach:

- Remove nodes iteratively
- Re-test planarity

This is a non-trivial structural problem.

---

## Recommended Starting Points

If prioritising:

1. Betweenness centrality → critical stations
2. Targeted attack → network fragility
3. Line-aware shortest paths → realistic journeys

---

## Next Step

Build a `(station, line)` graph to properly model:

- interchange costs
- passenger routing behaviour

This moves from pure graph theory toward transport modelling.

`````
Fair. Here’s a clean, concise addendum in the same style.

---

````md
## Addendum: Network Robustness and Coverage

This sub-project explores two structural questions about the Underground graph:

1. How fragile is the network to station removal?
2. Is it possible to visit every station without repetition?

---

### A. Minimum Station Removal (Network Connectivity)

**Question:**
What is the smallest number of stations that must be removed to disconnect the network?

**Approach:**
- Compute the **node connectivity** of the graph
- Identify a **minimum node cut** (a set of stations whose removal splits the network)

```python
import networkx as nx

k = nx.node_connectivity(G)
cut_nodes = nx.minimum_node_cut(G)
`````

**Output:**

- `k`: minimum number of stations required
- `cut_nodes`: one specific set achieving this

**Follow-up:**

- Remove these stations and inspect resulting components
- Compare with targeted removal of high-centrality stations

**Interpretation:**

- Small `k` → structurally fragile network
- Large `k` → robust, redundant connectivity

---

### B. Visiting Every Station Once

**Question:**
Is there a route that visits every station exactly once?

**Interpretation:**

- This is the **Hamiltonian path problem**
- Likely **no exact solution** exists for the full network

**Practical approach:**

- Attempt heuristic or approximate solutions
- Alternatively, solve a relaxed version:

> Find a route that visits every station with minimal repetition

**Possible methods:**

- Greedy traversal using shortest paths
- Approximate travelling salesman on station distances
- Tree-based traversal with shortcutting

**Outcome:**

- Either:
  - A valid Hamiltonian path (unlikely), or
  - A near-optimal covering route

---

### Summary

This sub-project shifts focus from:

- **static structure** (planarity, connectivity)

to:

- **dynamic resilience** (how the network fails)
- **traversability** (how efficiently it can be explored)

Together, these give a more complete picture of the Underground as a graph.

```

---

If you want, the next step would be to actually **compute and interpret your specific minimum cut**, which is usually quite revealing for this network.
```
