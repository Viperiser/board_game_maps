import pandas as pd
import numpy as np


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


labels, matrix = read_adjacency_matrix_from_excel(
    "20260414-King is Dead Adjacency.xlsx"
)
