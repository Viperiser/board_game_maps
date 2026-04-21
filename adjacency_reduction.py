from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def read_adjacency_matrix_from_excel(path):
    """
    Read a square adjacency matrix from an Excel file.

    Assumes:
    - first row contains column labels
    - first column contains row labels
    - row and column labels match in order
    """
    df = pd.read_excel(path, index_col=0)

    labels = list(df.index)
    matrix = df.to_numpy()

    if df.shape[0] != df.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    if list(df.columns) != labels:
        raise ValueError("Row and column labels must match exactly.")

    return labels, matrix


def graph_from_adjacency_matrix(labels, matrix):
    """
    Build an undirected simple graph from a symmetric 0/1 adjacency matrix.
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError("matrix must be square")

    if len(labels) != n_rows:
        raise ValueError("number of labels must match matrix size")

    if not np.array_equal(matrix, matrix.T):
        raise ValueError("matrix must be symmetric for an undirected graph")

    G = nx.Graph()
    G.add_nodes_from(labels)

    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            if matrix[i, j] != 0:
                G.add_edge(labels[i], labels[j])

    return G


def suppress_degree_two_nodes(G):
    """
    Repeatedly suppress degree-2 nodes.

    If node n has neighbours u and v, remove n and add edge (u, v).
    This treats degree-2 nodes as mere path intermediates.

    No attempt is made to track which original nodes were removed.
    """
    H = G.copy()

    changed = True
    while changed:
        changed = False

        for n in list(H.nodes()):
            if H.degree(n) != 2:
                continue

            u, v = list(H.neighbors(n))

            # Ignore degenerate self-loop case
            if u == v:
                H.remove_node(n)
                changed = True
                break

            H.remove_node(n)
            H.add_edge(u, v)
            changed = True
            break

    return H


def adjacency_matrix_from_graph(G):
    """
    Return labels and symmetric 0/1 adjacency matrix from a graph.
    Labels are sorted alphabetically for stable output.
    """
    labels = sorted(G.nodes(), key=str)
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((n, n), dtype=int)

    for u, v in G.edges():
        i = index[u]
        j = index[v]
        matrix[i, j] = 1
        matrix[j, i] = 1

    return labels, matrix


def remove_degree_one_nodes(G):
    """
    Repeatedly remove degree-1 nodes until none remain.
    """
    H = G.copy()

    changed = True
    while changed:
        changed = False

        for n in list(H.nodes()):
            if H.degree(n) == 1:
                H.remove_node(n)
                changed = True

        # loop again in case new danglers were created

    return H


def save_adjacency_matrix_to_excel(labels, matrix, path):
    """
    Save adjacency matrix to an Excel file with labels as both row and column headers.
    """
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_excel(path)


def reduce_adjacency_matrix_excel(input_path, remove_danglers=False):
    """
    Read adjacency matrix from Excel, suppress degree-2 nodes, optionally
    remove degree-1 nodes, and save reduced matrix to a new Excel file.

    If remove_danglers is True, repeatedly removes nodes of degree 1 after
    suppressing degree-2 nodes.

    Output file name:
    - normal:   originalname-reduced.xlsx
    - danglers: originalname-reduced-nodanglers.xlsx
    """
    input_path = Path(input_path)

    labels, matrix = read_adjacency_matrix_from_excel(input_path)
    G = graph_from_adjacency_matrix(labels, matrix)

    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    H = suppress_degree_two_nodes(G)
    print(
        f"After suppressing degree-2 nodes: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges"
    )

    if remove_danglers:
        H = remove_degree_one_nodes(H)
        print(
            f"After removing danglers:        {H.number_of_nodes()} nodes, {H.number_of_edges()} edges"
        )

    reduced_labels, reduced_matrix = adjacency_matrix_from_graph(H)

    if remove_danglers:
        output_path = input_path.with_name(
            f"{input_path.stem}-reduced-nodanglers{input_path.suffix}"
        )
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}-reduced{input_path.suffix}"
        )

    save_adjacency_matrix_to_excel(reduced_labels, reduced_matrix, output_path)

    print(f"Saved reduced adjacency matrix to: {output_path}")

    return output_path


if __name__ == "__main__":
    INPUT_FILE = "20260421-Zone 1-Bank-Mon.xlsx"
    reduce_adjacency_matrix_excel(INPUT_FILE, remove_danglers=True)
