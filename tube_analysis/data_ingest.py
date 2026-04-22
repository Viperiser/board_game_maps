from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations
from typing import Iterable, Optional

from pathlib import Path
import pandas as pd
import networkx as nx

# ----------------------------
# Data Integrity
# ----------------------------


def normalise_name_for_matching(name: str) -> str:
    """
    Normalise a station name for fuzzy comparison.

    This is deliberately aggressive for matching purposes.
    It is NOT intended to replace your canonical station names.

    Examples:
        "King’s Cross St. Pancras" -> "kings cross st pancras"
        "Earl's Court"             -> "earls court"
        "  Bank-Monument "         -> "bank monument"
    """
    if pd.isna(name):
        return ""

    text = str(name).strip()

    # Unicode normalisation, then lowercase.
    text = unicodedata.normalize("NFKC", text).lower()

    # Replace common punctuation variants with spaces or nothing.
    text = text.replace("&", " and ")
    text = text.replace("’", "'")
    text = text.replace("`", "'")

    # Remove apostrophes entirely so Earl's and Earls match.
    text = text.replace("'", "")

    # Replace non-alphanumeric characters with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_near_matches(
    names: Iterable[str],
    threshold: float = 0.88,
    min_length: int = 4,
    include_exact_normalised_matches: bool = True,
    return_dataframe: bool = True,
) -> pd.DataFrame | list[dict]:
    """
    Find suspiciously similar station names.

    Parameters
    ----------
    names:
        Iterable of raw names to compare.
    threshold:
        Minimum similarity ratio for fuzzy matches, using SequenceMatcher.
        0.88 is a sensible starting point.
    min_length:
        Ignore very short strings after normalisation, because they create noise.
    include_exact_normalised_matches:
        If True, include cases where raw strings differ but normalised forms
        are identical. These are often the most useful cases.
    return_dataframe:
        If True, return a pandas DataFrame. Otherwise return a list of dicts.

    Returns
    -------
    pd.DataFrame or list[dict]
        Candidate near-matches, sorted from most suspicious to least.
    """
    unique_names = sorted({str(n).strip() for n in names if pd.notna(n)})
    normalised = {name: normalise_name_for_matching(name) for name in unique_names}

    results: list[dict] = []

    for a, b in combinations(unique_names, 2):
        na = normalised[a]
        nb = normalised[b]

        if len(na) < min_length or len(nb) < min_length:
            continue

        raw_ratio = SequenceMatcher(None, a, b).ratio()
        norm_ratio = SequenceMatcher(None, na, nb).ratio()

        exact_norm_match = (na == nb) and (a != b)

        if exact_norm_match and include_exact_normalised_matches:
            match_type = "exact_normalised_match"
        elif norm_ratio >= threshold:
            match_type = "fuzzy_normalised_match"
        else:
            continue

        results.append(
            {
                "name_1": a,
                "name_2": b,
                "normalised_1": na,
                "normalised_2": nb,
                "raw_similarity": round(raw_ratio, 4),
                "normalised_similarity": round(norm_ratio, 4),
                "match_type": match_type,
            }
        )

    results.sort(
        key=lambda row: (
            row["match_type"] != "exact_normalised_match",
            -row["normalised_similarity"],
            -row["raw_similarity"],
            row["name_1"],
            row["name_2"],
        )
    )

    if return_dataframe:
        return pd.DataFrame(results)

    return results


def find_exact_normalised_collisions(names: Iterable[str]) -> pd.DataFrame:
    """
    Return groups of raw names that collapse to the same normalised form.
    """
    unique_names = sorted({str(n).strip() for n in names if pd.notna(n)})

    rows = []
    for name in unique_names:
        rows.append(
            {
                "raw_name": name,
                "normalised_name": normalise_name_for_matching(name),
            }
        )

    df = pd.DataFrame(rows)

    grouped = (
        df.groupby("normalised_name")["raw_name"]
        .agg(lambda x: sorted(set(x)))
        .reset_index()
    )

    grouped["n_raw_names"] = grouped["raw_name"].apply(len)

    grouped = grouped[grouped["n_raw_names"] > 1].copy()
    grouped = grouped.sort_values(
        ["n_raw_names", "normalised_name"], ascending=[False, True]
    ).reset_index(drop=True)

    return grouped


# ----------------------------
# Alias handling
# ----------------------------


class UnionFind:
    """
    Minimal union-find / disjoint-set structure for station alias groups.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_station_alias_map(
    station_names: Iterable[str],
    equivalent_pairs: Optional[Iterable[tuple[str, str]]] = None,
    canonical_name_joiner: str = "-",
) -> dict[str, str]:
    """
    Build a mapping from raw station name -> canonical station name.

    If equivalent_pairs contains ("Bank", "Monument"), both will map to
    "Bank-Monument" by default.

    If you provide chains such as:
        [("Bank", "Monument"), ("Monument", "Cannon Street")]
    then all three will collapse into one canonical node.

    Parameters
    ----------
    station_names:
        All station names appearing in the CSV.
    equivalent_pairs:
        Pairs of names to treat as the same station.
    canonical_name_joiner:
        String used to join names in a merged station label.

    Returns
    -------
    dict[str, str]
        Mapping from original station name to canonical station name.
    """
    uf = UnionFind()

    station_names = list(station_names)
    for name in station_names:
        uf.find(name)

    for a, b in equivalent_pairs or []:
        uf.union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for name in station_names:
        groups[uf.find(name)].append(name)

    canonical_by_root: dict[str, str] = {}
    for root, members in groups.items():
        canonical_by_root[root] = canonical_name_joiner.join(sorted(set(members)))

    alias_map: dict[str, str] = {}
    for name in station_names:
        alias_map[name] = canonical_by_root[uf.find(name)]

    return alias_map


# ----------------------------
# Edge record
# ----------------------------


@dataclass
class EdgeRecord:
    station_a: str
    station_b: str
    lines: set[str]
    supporting_lines: list[str]

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def to_dict(self) -> dict:
        return {
            "station_a": self.station_a,
            "station_b": self.station_b,
            "lines": tuple(sorted(self.lines)),
            "lines_str": ";".join(sorted(self.lines)),
            "n_lines": self.n_lines,
            "supporting_lines": tuple(self.supporting_lines),
        }


# ----------------------------
# Main builder
# ----------------------------


def build_underground_graph(
    df: pd.DataFrame,
    equivalent_pairs: Optional[Iterable[tuple[str, str]]] = None,
    drop_self_loops: bool = True,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Build a simple undirected NetworkX graph from station sequence data.

    Expected columns in df:
        - uid
        - position
        - name
        - line

    Behaviour:
        - Stations are canonicalised first using equivalent_pairs.
        - Within each line, rows are sorted by position.
        - Consecutive stations are converted into edges.
        - Repeated edges across lines are collapsed into one edge with
          a multi-line metadata attribute.

    Parameters
    ----------
    df:
        Input station sequence table.
    equivalent_pairs:
        Pairs of station names to treat as the same station, e.g.
        [("Bank", "Monument")].
    drop_self_loops:
        If aliasing causes consecutive stations to collapse to the same
        canonical node, drop that self-loop edge.

    Returns
    -------
    G : nx.Graph
        Graph with node names as canonical station names.
    edges_df : pd.DataFrame
        One row per unique adjacency, with line metadata.
    stations_df : pd.DataFrame
        Original rows plus canonical station names.
    """
    required_cols = {"uid", "position", "name", "line"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    working = df.copy()

    # Make sure ordering works sensibly.
    working["position"] = pd.to_numeric(working["position"], errors="raise")

    # Canonicalise station names.
    alias_map = build_station_alias_map(
        station_names=working["name"].dropna().unique(),
        equivalent_pairs=equivalent_pairs,
    )
    working["canonical_name"] = working["name"].map(alias_map)

    # Build raw consecutive edges by line.
    edge_records: dict[tuple[str, str], EdgeRecord] = {}

    for line_name, group in working.groupby("line", sort=False):
        group = group.sort_values("position").reset_index(drop=True)

        stations = group["canonical_name"].tolist()
        for i in range(len(stations) - 1):
            a = stations[i]
            b = stations[i + 1]

            if pd.isna(a) or pd.isna(b):
                continue

            # Undirected simple graph edge key.
            edge_key = tuple(sorted((a, b)))

            if drop_self_loops and edge_key[0] == edge_key[1]:
                continue

            if edge_key not in edge_records:
                edge_records[edge_key] = EdgeRecord(
                    station_a=edge_key[0],
                    station_b=edge_key[1],
                    lines=set(),
                    supporting_lines=[],
                )

            edge_records[edge_key].lines.add(line_name)
            edge_records[edge_key].supporting_lines.append(line_name)

    # Convert edge records to DataFrame.
    edges_df = (
        pd.DataFrame([record.to_dict() for record in edge_records.values()])
        .sort_values(["station_a", "station_b"])
        .reset_index(drop=True)
    )

    # Build NetworkX graph.
    G = nx.Graph()

    # Add nodes from canonical names.
    for canonical_name in sorted(working["canonical_name"].dropna().unique()):
        original_names = sorted(
            working.loc[working["canonical_name"] == canonical_name, "name"]
            .dropna()
            .unique()
            .tolist()
        )
        serving_lines = sorted(
            working.loc[working["canonical_name"] == canonical_name, "line"]
            .dropna()
            .unique()
            .tolist()
        )

        G.add_node(
            canonical_name,
            original_names=tuple(original_names),
            merged_from=tuple(original_names),
            lines=tuple(serving_lines),
            n_lines=len(serving_lines),
        )

    # Add edges with metadata.
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["station_a"],
            row["station_b"],
            lines=row["lines"],
            lines_str=row["lines_str"],
            n_lines=row["n_lines"],
        )

    return G, edges_df, working


# ----------------------------
# Graph inspection tools
# ----------------------------


def print_all_edges(G: nx.Graph, sort_by: str = "stations") -> None:
    """
    Print every edge and its line metadata.

    sort_by:
        - "stations": alphabetical by node names
        - "n_lines": descending number of lines, then alphabetical
    """
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append(
            {
                "station_a": u,
                "station_b": v,
                "lines": data.get("lines", ()),
                "lines_str": data.get("lines_str", ""),
                "n_lines": data.get("n_lines", 0),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("[Graph has no edges]")
        return

    if sort_by == "n_lines":
        df = df.sort_values(
            ["n_lines", "station_a", "station_b"],
            ascending=[False, True, True],
        )
    else:
        df = df.sort_values(["station_a", "station_b"])

    for _, row in df.iterrows():
        print(f"{row['station_a']}  <->  {row['station_b']}   " f"[{row['lines_str']}]")


def edge_table(G: nx.Graph) -> pd.DataFrame:
    """
    Return the graph edges as a tidy DataFrame.
    """
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append(
            {
                "station_a": u,
                "station_b": v,
                "lines": data.get("lines", ()),
                "lines_str": data.get("lines_str", ""),
                "n_lines": data.get("n_lines", 0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(["station_a", "station_b"]).reset_index(drop=True)


def print_edges_for_station(G: nx.Graph, station: str) -> None:
    """
    Print all edges incident to one station.
    """
    if station not in G:
        print(f"[Station not found: {station}]")
        return

    neighbours = sorted(G.neighbors(station))
    if not neighbours:
        print(f"[Station has no neighbours: {station}]")
        return

    print(f"\nEdges incident to {station}:")
    for nbr in neighbours:
        data = G.get_edge_data(station, nbr, default={})
        print(f"  {station}  <->  {nbr}   " f"[{data.get('lines_str', '')}]")


def print_edges_for_line(G: nx.Graph, line_name: str) -> None:
    """
    Print all edges that include a given line in their metadata.
    """
    matches = []
    for u, v, data in G.edges(data=True):
        lines = set(data.get("lines", ()))
        if line_name in lines:
            matches.append((u, v, data.get("lines_str", "")))

    if not matches:
        print(f"[No edges found for line: {line_name}]")
        return

    matches.sort()
    print(f"\nEdges on line: {line_name}")
    for u, v, lines_str in matches:
        print(f"  {u}  <->  {v}   [{lines_str}]")


def print_subgraph_neighbourhood(G: nx.Graph, stations: list[str]) -> None:
    """
    Print all edges touching any station in the supplied list.
    Useful for checking a local patch of the graph.
    """
    seen = set()

    for station in stations:
        if station not in G:
            print(f"[Station not found: {station}]")
            continue

        for nbr in G.neighbors(station):
            edge = tuple(sorted((station, nbr)))
            if edge in seen:
                continue
            seen.add(edge)

    if not seen:
        print("[No matching local edges found]")
        return

    print("\nLocal edge neighbourhood:")
    for u, v in sorted(seen):
        data = G.get_edge_data(u, v, default={})
        print(f"  {u}  <->  {v}   [{data.get('lines_str', '')}]")


# ----------------------------
# Convenience wrapper for CSV
# ----------------------------


def build_underground_graph_from_csv(
    csv_path: str,
    equivalent_pairs: Optional[Iterable[tuple[str, str]]] = None,
    drop_self_loops: bool = True,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    return build_underground_graph(
        df=df,
        equivalent_pairs=equivalent_pairs,
        drop_self_loops=drop_self_loops,
    )


def export_canonical_graph_tables(
    G: nx.Graph,
    edges_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    output_dir: str | Path,
    stations_filename: str = "stations_canonical.csv",
    edges_filename: str = "edges_canonical.csv",
    aliases_filename: str = "station_aliases.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Export canonical station and edge tables.

    Outputs
    -------
    stations_canonical.csv
        One row per canonical station node.

    edges_canonical.csv
        One row per unique adjacency.

    station_aliases.csv
        One row per original source row, showing how raw station names map
        to canonical station names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Canonical station table
    # ----------------------------
    station_rows = []
    for node, data in G.nodes(data=True):
        original_names = tuple(sorted(data.get("original_names", (node,))))
        lines = tuple(sorted(data.get("lines", ())))

        station_rows.append(
            {
                "station_id": node,
                "display_name": node,
                "original_names": "|".join(original_names),
                "n_original_names": len(original_names),
                "lines": ";".join(lines),
                "n_lines": len(lines),
            }
        )

    stations_canonical_df = (
        pd.DataFrame(station_rows).sort_values("station_id").reset_index(drop=True)
    )

    # ----------------------------
    # Canonical edge table
    # ----------------------------
    edge_rows = []
    for u, v, data in G.edges(data=True):
        a, b = sorted((u, v))
        lines = tuple(sorted(data.get("lines", ())))

        edge_rows.append(
            {
                "edge_id": f"{a}__{b}",
                "station_a": a,
                "station_b": b,
                "lines": ";".join(lines),
                "n_lines": len(lines),
            }
        )

    edges_canonical_df = (
        pd.DataFrame(edge_rows)
        .sort_values(["station_a", "station_b"])
        .reset_index(drop=True)
    )

    # ----------------------------
    # Raw-to-canonical alias table
    # ----------------------------
    alias_cols = [
        c
        for c in ["uid", "line", "position", "name", "canonical_name"]
        if c in stations_df.columns
    ]
    aliases_df = (
        stations_df[alias_cols]
        .rename(columns={"name": "raw_name"})
        .sort_values(["line", "position", "raw_name"])
        .reset_index(drop=True)
    )

    # ----------------------------
    # Write files
    # ----------------------------
    stations_canonical_df.to_csv(output_dir / stations_filename, index=False)
    edges_canonical_df.to_csv(output_dir / edges_filename, index=False)
    aliases_df.to_csv(output_dir / aliases_filename, index=False)

    return stations_canonical_df, edges_canonical_df, aliases_df


def export_graphml(G: nx.Graph, filepath: str | Path) -> None:
    """
    Export the graph to GraphML.

    GraphML does not like tuples/sets, so convert those attributes to strings.
    """
    H = nx.Graph()

    for node, data in G.nodes(data=True):
        clean_data = {}
        for k, v in data.items():
            if isinstance(v, (tuple, list, set)):
                clean_data[k] = ";".join(map(str, v))
            else:
                clean_data[k] = v
        H.add_node(node, **clean_data)

    for u, v, data in G.edges(data=True):
        clean_data = {}
        for k, v2 in data.items():
            if isinstance(v2, (tuple, list, set)):
                clean_data[k] = ";".join(map(str, v2))
            else:
                clean_data[k] = v2
        H.add_edge(u, v, **clean_data)

    nx.write_graphml(H, filepath)


#### EXECUTION ########

FILE = "tube_analysis/20260422-Tube Lines Master.csv"

equivalent_pairs = [
    ("Bank", "Monument"),
    # add more if needed later
]

G, edges_df, stations_df = build_underground_graph_from_csv(
    FILE,
    equivalent_pairs=equivalent_pairs,
)

stations_canonical_df, edges_canonical_df, aliases_df = export_canonical_graph_tables(
    G=G,
    edges_df=edges_df,
    stations_df=stations_df,
    output_dir="canonical_output",
)

export_graphml(G, "tube_analysis/underground.graphml")
