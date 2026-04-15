import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import font_manager
import matplotlib.patheffects as pe


def read_adjacency_matrix_from_excel(
    filepath: str, sheet_name: str | int = 0
) -> tuple[list[str], np.ndarray]:
    """
    Read an adjacency matrix from an Excel sheet structured like:

        [blank], N1, N2, N3, ...
        N1,      0,  1,  0, ...
        N2,      1,  0,  1, ...
        N3,      0,  1,  0, ...

    Returns
    -------
    labels : list[str]
        The node labels, e.g. ["N1", "N2", "N3"]
    matrix : np.ndarray
        A square numpy array of 0s and 1s
    """

    # Read the sheet, treating the first column as row labels
    df = pd.read_excel(filepath, sheet_name=sheet_name, index_col=0)

    # Force labels to strings and strip whitespace
    row_labels = [str(x).strip() for x in df.index.tolist()]
    col_labels = [str(x).strip() for x in df.columns.tolist()]

    # Check that row and column labels match
    if row_labels != col_labels:
        raise ValueError(
            f"Row labels and column labels do not match.\n"
            f"Rows:    {row_labels}\n"
            f"Columns: {col_labels}"
        )

    # Convert entries to integers
    try:
        matrix = df.to_numpy(dtype=int)
    except Exception as e:
        raise ValueError(f"Could not convert matrix entries to integers: {e}") from e

    # Check that entries are binary
    unique_values = set(np.unique(matrix))
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Matrix entries must be binary 0/1, found values: {sorted(unique_values)}"
        )

    return row_labels, matrix


def graph_from_adjacency_matrix(labels, matrix):

    # Note that the GRAPH is the abstract object, not yet an embedding

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


def get_planar_embedding(G, require_planar=True):
    """
    Check planarity and return a PlanarEmbedding if planar.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    require_planar : bool, default True
        If True, raise an error when graph is not planar

    Returns
    -------
    is_planar : bool
    embedding : nx.PlanarEmbedding or None
    """
    is_planar, emb = nx.check_planarity(G)

    if require_planar and not is_planar:
        raise ValueError("Graph is not planar")

    return is_planar, emb


labels, matrix = read_adjacency_matrix_from_excel(
    "20260414-King is Dead Adjacency.xlsx"
)
print("Labels:", labels)
print("Matrix:\n", matrix)

G = graph_from_adjacency_matrix(labels, matrix)
print("Graph nodes:", G.nodes())
print("Graph edges:", G.edges())

is_planar, embedding = get_planar_embedding(G)
print("Is planar:", is_planar)
print("Planar embedding:", embedding)

for v in embedding:
    print(v, list(embedding.neighbors_cw_order(v)))

visual_data = get_visual_data(embedding)
print("Visual data:", visual_data)

fig, ax = draw_visual_data(visual_data, colour="#0077CC")
plt.show()
