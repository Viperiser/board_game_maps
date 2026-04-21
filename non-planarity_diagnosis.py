import pandas as pd
import networkx as nx


def read_graph_from_excel(path):
    """
    Expect a square adjacency matrix with labels in first row + first column.
    """
    df = pd.read_excel(path, index_col=0)
    labels = list(df.index)

    G = nx.Graph()

    for u in labels:
        G.add_node(u)

    for i, u in enumerate(labels):
        for j, v in enumerate(labels):
            if j <= i:
                continue
            if df.iloc[i, j] != 0:
                G.add_edge(u, v)

    return G


def _orient_path(path, start, end):
    """
    Return path oriented from start to end.
    """
    if path[0] == start and path[-1] == end:
        return path
    if path[0] == end and path[-1] == start:
        return list(reversed(path))
    raise ValueError(f"Path {path} does not connect {start} to {end}")


def suppress_degree_two_nodes(H):
    """
    Collapse degree-2 nodes while preserving a simple path for each reduced edge.

    Each reduced edge gets:
        edge["path"] = [u, ..., v]
    """
    H = H.copy()

    # initialise paths on original edges
    for u, v in H.edges():
        H[u][v]["path"] = [u, v]

    changed = True
    while changed:
        changed = False

        for n in list(H.nodes()):
            if H.degree(n) != 2:
                continue

            u, v = list(H.neighbors(n))
            if u == v:
                continue

            path_un = _orient_path(H[u][n]["path"], u, n)
            path_nv = _orient_path(H[n][v]["path"], n, v)

            new_path = path_un + path_nv[1:]

            H.remove_node(n)

            if H.has_edge(u, v):
                old_path = H[u][v].get("path", [u, v])
                if len(new_path) < len(old_path):
                    H[u][v]["path"] = new_path
            else:
                H.add_edge(u, v, path=new_path)

            changed = True
            break

    return H

    return H


def diagnose_nonplanarity(path):
    G = read_graph_from_excel(path)

    is_planar, embedding = nx.check_planarity(G, counterexample=True)

    print(f"Planar: {is_planar}")

    if is_planar:
        return

    H = embedding  # this is the Kuratowski subgraph

    core = suppress_degree_two_nodes(H)

    print("\n--- Kuratowski subgraph ---")
    print("Nodes:")
    for n in H.nodes():
        print(" ", n)

    print("\nEdges:")
    for u, v in H.edges():
        print(f"  {u} -- {v}")

    # Optional: show degrees to spot branch vertices
    print("\nDegrees:")
    for n, d in H.degree():
        print(f"  {n}: {d}")

    # Identify likely branch vertices (degree >= 3)
    branch_nodes = [n for n, d in H.degree() if d >= 3]

    print("\nLikely branch vertices (K5 / K3,3 core):")
    for n in branch_nodes:
        print(" ", n)

    print("\n--- Reduced core with paths ---")
    for u, v, data in core.edges(data=True):
        path = data["path"]
        print(f"{u} -- {v}")
        print("  path:", " -> ".join(path))
        print("  repeated nodes:", len(path) != len(set(path)))


if __name__ == "__main__":
    diagnose_nonplanarity("20260421-Zones 1 and 2.xlsx")
