from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx


def load_graphml_graph(graphml_path: str | Path) -> nx.Graph:
    """
    Load a graph saved as GraphML.

    Returns a NetworkX graph object.
    """
    graphml_path = Path(graphml_path)
    return nx.read_graphml(graphml_path)


def analyse_planarity(
    G: nx.Graph,
    *,
    verbose: bool = True,
) -> tuple[bool, object]:
    """
    Check whether G is planar.

    Returns
    -------
    (is_planar, certificate)
        If planar, certificate is a PlanarEmbedding.
        If non-planar, certificate is a Kuratowski subgraph.
    """
    is_planar, certificate = nx.check_planarity(G, counterexample=True)

    if verbose:
        print(f"Planar: {is_planar}")

        if is_planar:
            print("Returned certificate type: PlanarEmbedding")
            # Optional sanity check on the embedding structure.
            try:
                certificate.check_structure()
                print("Embedding structure check: OK")
            except Exception as exc:
                print(f"Embedding structure check failed: {exc}")
        else:
            print("Returned certificate type: Kuratowski subgraph")
            print(
                f"Counterexample has "
                f"{certificate.number_of_nodes()} nodes and "
                f"{certificate.number_of_edges()} edges."
            )

    return is_planar, certificate


def print_counterexample_edges(counterexample: nx.Graph) -> None:
    """
    Print the edges of a non-planarity counterexample subgraph.
    """
    print("\nCounterexample edges:")
    for u, v in sorted(counterexample.edges()):
        print(f"  {u}  <->  {v}")


def print_counterexample_nodes(counterexample: nx.Graph) -> None:
    """
    Print the nodes of a non-planarity counterexample subgraph.
    """
    print("\nCounterexample nodes:")
    for node in sorted(counterexample.nodes()):
        print(f"  {node}")


def save_counterexample_graphml(
    counterexample: nx.Graph,
    output_path: str | Path,
) -> None:
    """
    Save the Kuratowski subgraph so you can inspect it elsewhere.
    """
    output_path = Path(output_path)
    nx.write_graphml(counterexample, output_path)
    print(f"Saved counterexample GraphML to: {output_path}")


def extract_witness_segments(counterexample: nx.Graph) -> list[list[str]]:
    """
    Extract all maximal branch-to-branch segments from a Kuratowski witness.

    A 'branch vertex' is any node whose degree is not 2.
    Each returned segment is a full path in the original witness graph,
    including all intermediate degree-2 vertices.

    Returns
    -------
    list[list[str]]
        A list of paths, where each path is a list of station names.
    """
    G = nx.Graph(counterexample)

    branch_vertices = {n for n, d in G.degree() if d != 2}
    visited_directed = set()
    segments: list[list[str]] = []

    for start in branch_vertices:
        for nbr in G.neighbors(start):
            directed_edge = (start, nbr)
            if directed_edge in visited_directed:
                continue

            path = [start, nbr]
            visited_directed.add((start, nbr))
            visited_directed.add((nbr, start))

            prev = start
            cur = nbr

            while cur not in branch_vertices:
                next_nodes = [x for x in G.neighbors(cur) if x != prev]

                if len(next_nodes) != 1:
                    raise ValueError(
                        f"Expected exactly one onward neighbour at degree-2 node "
                        f"{cur}, found {len(next_nodes)}"
                    )

                nxt = next_nodes[0]
                path.append(nxt)
                visited_directed.add((cur, nxt))
                visited_directed.add((nxt, cur))

                prev, cur = cur, nxt

            segments.append(path)

    # Canonicalise orientation so duplicates cannot sneak in.
    canonical_segments = []
    seen = set()

    for path in segments:
        forward = tuple(path)
        backward = tuple(reversed(path))
        canonical = min(forward, backward)

        if canonical not in seen:
            seen.add(canonical)
            canonical_segments.append(list(canonical))

    canonical_segments.sort(key=lambda p: (p[0], p[-1], len(p), tuple(p)))
    return canonical_segments


def print_witness_segments(counterexample: nx.Graph) -> None:
    """
    Print all witness segments in a human-checkable form.
    """
    segments = extract_witness_segments(counterexample)

    print("Witness segments:")
    for i, seg in enumerate(segments, start=1):
        print(f"{i:>2}. " + " -> ".join(seg))


def witness_segments_df(counterexample: nx.Graph):
    """
    Return the witness segments as a pandas DataFrame.
    """
    import pandas as pd

    segments = extract_witness_segments(counterexample)

    rows = []
    for i, seg in enumerate(segments, start=1):
        rows.append(
            {
                "segment_id": i,
                "start": seg[0],
                "end": seg[-1],
                "n_nodes": len(seg),
                "n_edges": len(seg) - 1,
                "path_str": " -> ".join(seg),
                "path_nodes": tuple(seg),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = load_graphml_graph("tube_analysis/underground.graphml")

    is_planar, certificate = analyse_planarity(G)

    if is_planar:
        print("\nThe graph is planar.")
    else:
        print("\nThe graph is NOT planar.")
        print_counterexample_nodes(certificate)
        print_counterexample_edges(certificate)
        save_counterexample_graphml(
            certificate,
            "tube_analysis/underground_planarity_counterexample.graphml",
        )
        print_witness_segments(certificate)
