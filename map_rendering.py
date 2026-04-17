import pandas as pd
import numpy as np
import networkx as nx


import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.patches import Rectangle


# ======================================================================
# Helpers
# ======================================================================


def detect_outer_face_id(master_embedding, primal_pos):
    """
    Pick the face with the largest absolute polygon area, based on the
    boundary nodes in the primal drawing.
    """
    best_face_id = None
    best_area = -1.0

    for face in master_embedding["faces"]:
        pts = [primal_pos[n] for n in face["boundary_nodes"]]
        area = abs(polygon_signed_area(pts))

        if area > best_area:
            best_area = area
            best_face_id = face["id"]

    return best_face_id


def place_bounded_face_node(face, primal_pos, shrink=0.88):
    """
    Place a bounded face node near the polygon centroid, shrunk slightly
    toward the average of boundary vertices to keep it away from edges.
    """
    pts = np.array([primal_pos[n] for n in face["boundary_nodes"]], dtype=float)

    centroid = np.array(polygon_centroid(pts), dtype=float)
    mean_pt = pts.mean(axis=0)

    p = shrink * centroid + (1 - shrink) * mean_pt
    return tuple(p)


def cubic_controls_for_primal(p0, p3, alpha1=1 / 3, alpha2=2 / 3):
    """
    Straight-line cubic controls.
    """
    p1 = (1 - alpha1) * p0 + alpha1 * p3
    p2 = (1 - alpha2) * p0 + alpha2 * p3
    return p1, p2


def make_outer_square(primal_pos, style):
    pts = np.array(list(primal_pos.values()), dtype=float)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)

    min_x, min_y = min_xy
    max_x, max_y = max_xy

    width = max_x - min_x
    height = max_y - min_y
    span = max(width, height)

    margin = style["outer_square_margin"] * span

    if style.get("outer_square_side") is None:
        side = span + 2 * margin
    else:
        side = style["outer_square_side"]

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    half = 0.5 * side

    return {
        "centre": (cx, cy),
        "side": side,
        "left": cx - half,
        "right": cx + half,
        "bottom": cy - half,
        "top": cy + half,
    }


def assign_outer_square_ports(
    outer_square, ordered_edge_ids, master_embedding, node_xy
):
    """
    Assign one attachment point on the square boundary to each outer-face edge.

    Each port is chosen by casting a ray from the square centre toward the
    corresponding border node, and intersecting that ray with the square.
    """
    ports = {}

    cx, cy = outer_square["centre"]
    left = outer_square["left"]
    right = outer_square["right"]
    bottom = outer_square["bottom"]
    top = outer_square["top"]

    centre = np.array([cx, cy], dtype=float)

    for edge_id in ordered_edge_ids:
        edge = master_embedding["edges"][edge_id]
        u = edge["u"]
        v = edge["v"]

        # Identify the border endpoint of this dual edge
        if master_embedding["nodes"][u]["type"] == "border":
            border_id = u
        else:
            border_id = v

        border_xy = np.array(node_xy[border_id], dtype=float)
        d = border_xy - centre

        dx, dy = d

        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            # Degenerate fallback
            ports[edge_id] = (right, cy)
            continue

        candidates = []

        # Intersections with vertical sides
        if abs(dx) > 1e-12:
            t_left = (left - cx) / dx
            y_left = cy + t_left * dy
            if t_left > 0 and bottom <= y_left <= top:
                candidates.append((t_left, (left, y_left)))

            t_right = (right - cx) / dx
            y_right = cy + t_right * dy
            if t_right > 0 and bottom <= y_right <= top:
                candidates.append((t_right, (right, y_right)))

        # Intersections with horizontal sides
        if abs(dy) > 1e-12:
            t_bottom = (bottom - cy) / dy
            x_bottom = cx + t_bottom * dx
            if t_bottom > 0 and left <= x_bottom <= right:
                candidates.append((t_bottom, (x_bottom, bottom)))

            t_top = (top - cy) / dy
            x_top = cx + t_top * dx
            if t_top > 0 and left <= x_top <= right:
                candidates.append((t_top, (x_top, top)))

        if not candidates:
            ports[edge_id] = (right, cy)
        else:
            # Nearest positive intersection along the ray
            _, pt = min(candidates, key=lambda x: x[0])
            ports[edge_id] = pt

    return ports


def cubic_controls_for_dual(
    p_face,
    p_border,
    primal_edge,
    primal_pos,
    idx,
    n,
    launch_strength=0.28,
    arrival_strength=0.18,
    tangent_strength=0.12,
):
    """
    Cubic controls for a bounded-face dual edge.

    c_face controls launch from the face node.
    c_border controls arrival at the border node, making it roughly
    perpendicular to the primal edge.
    """
    a, b = primal_edge
    pa = np.array(primal_pos[a], dtype=float)
    pb = np.array(primal_pos[b], dtype=float)

    edge_vec = pb - pa
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return cubic_controls_for_primal(p_face, p_border)

    edge_unit = edge_vec / edge_len
    n1 = np.array([-edge_unit[1], edge_unit[0]])
    n2 = -n1

    # Choose the normal pointing toward the face node
    if np.dot(p_face - p_border, n1) > np.dot(p_face - p_border, n2):
        inward = n1
    else:
        inward = n2

    span = np.linalg.norm(p_border - p_face)

    if n <= 1:
        fan = 0.0
    else:
        fan = (idx - 0.5 * (n - 1)) / max(0.5 * (n - 1), 1e-12)

    c_face = (
        p_face
        + arrival_strength * (p_border - p_face)
        + tangent_strength * span * fan * edge_unit
    )

    c_border = p_border + launch_strength * span * inward

    return c_face, c_border


def cubic_controls_for_outer_dual(
    p_face,
    p_border,
    primal_edge,
    primal_pos,
    idx,
    n,
    launch_strength=0.45,
    arrival_strength=0.30,
    tangent_strength=0.35,
    curve_base=0.08,
    distance_scale=1.25,
    distance_power=2.0,
    angle_scale=0.8,
):
    """
    Cubic controls for an outer-face dual edge.

    c_face controls launch from the outer node in a neat fan.
    c_border controls arrival at the border node, perpendicular to the
    outer side of the primal edge.
    """
    a, b = primal_edge
    pa = np.array(primal_pos[a], dtype=float)
    pb = np.array(primal_pos[b], dtype=float)

    edge_vec = pb - pa
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return cubic_controls_for_primal(p_face, p_border)

    edge_unit = edge_vec / edge_len
    n1 = np.array([-edge_unit[1], edge_unit[0]])
    n2 = -n1

    # outward normal: choose the one pointing toward the assigned square port
    to_outer = p_face - p_border

    if np.dot(to_outer, n1) > np.dot(to_outer, n2):
        outward = n1
    else:
        outward = n2

    pts = np.array(list(primal_pos.values()), dtype=float)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

    dist = np.linalg.norm(p_border - p_face)
    dynamic = dist / max(span, 1e-12)
    dynamic_term = dynamic**distance_power

    to_face = p_face - p_border
    to_face_norm = np.linalg.norm(to_face)

    if to_face_norm < 1e-12:
        alignment = 0.0
    else:
        to_face_unit = to_face / to_face_norm
        alignment = np.dot(to_face_unit, outward)

    awkwardness = 0.5 * (1 - alignment)

    outward_push = (
        span
        * (curve_base + distance_scale * dynamic_term)
        * (1 + angle_scale * awkwardness)
    )

    if n <= 1:
        fan = 0.0
    else:
        fan = (idx - 0.5 * (n - 1)) / max(0.5 * (n - 1), 1e-12)

    # Launch fan at outer node
    c_face = (
        p_face
        + launch_strength * (p_border - p_face)
        + tangent_strength * span * fan * edge_unit
    )

    # Perpendicular arrival at border from outside
    c_border = p_border + arrival_strength * outward_push * outward

    return c_face, c_border


def cubic_bezier_point(p0, p1, p2, p3, t=0.5):
    """
    Point on a cubic Bezier curve.
    """
    return (
        ((1 - t) ** 3) * p0
        + 3 * ((1 - t) ** 2) * t * p1
        + 3 * (1 - t) * (t**2) * p2
        + (t**3) * p3
    )


def polygon_signed_area(points):
    """
    Signed area of polygon given by ordered points.
    """
    pts = np.array(points, dtype=float)
    n = len(pts)
    area = 0.0

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return 0.5 * area


def polygon_centroid(points):
    """
    Polygon centroid. Falls back to mean point if area is tiny.
    """
    pts = np.array(points, dtype=float)
    n = len(pts)

    area2 = 0.0
    cx = 0.0
    cy = 0.0

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    if abs(area2) < 1e-12:
        mean_pt = pts.mean(axis=0)
        return tuple(mean_pt)

    cx /= 3.0 * area2
    cy /= 3.0 * area2
    return (cx, cy)


# ======================================================================
# Main functions
# ======================================================================


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


def build_master_embedding(primal_embedding):
    """
    Build a master embedding from a primal NetworkX PlanarEmbedding.

    The master embedding contains:
    - primal region nodes
    - border nodes, one per primal edge
    - face nodes, one per primal face

    And edges of two types:
    - primal: region <-> border
    - dual:   face   <-> border

    Returns a lightweight custom embedding structure:
    {
        "nodes": {...},
        "edges": {...},
        "rotation": {...},
        "faces": [...],
        "primal_edge_to_border": {...},
        "halfedge_to_face": {...},
    }
    """

    master = {
        "nodes": {},
        "edges": {},
        "rotation": {},
        "faces": [],
        "primal_edge_to_border": {},
        "halfedge_to_face": {},
    }

    edge_counter = 0

    def add_node(node_id, **attrs):
        master["nodes"][node_id] = attrs
        master["rotation"][node_id] = []

    def add_edge(u, v, edge_type, **attrs):
        nonlocal edge_counter
        edge_id = f"e{edge_counter}"
        edge_counter += 1
        master["edges"][edge_id] = {
            "u": u,
            "v": v,
            "type": edge_type,
            **attrs,
        }
        return edge_id

    def canonical_edge(u, v):
        return tuple(sorted((u, v), key=str))

    # ------------------------------------------------------------------
    # 1. Add primal region nodes
    # ------------------------------------------------------------------
    for v in primal_embedding.nodes():
        add_node(v, type="region", label=str(v))

    # ------------------------------------------------------------------
    # 2. Extract primal faces from half-edges
    # ------------------------------------------------------------------
    seen_half_edges = set()
    face_counter = 0

    for u, v in primal_embedding.edges():
        if (u, v) in seen_half_edges:
            continue

        boundary_nodes = list(primal_embedding.traverse_face(u, v, seen_half_edges))
        half_edges = [
            (boundary_nodes[i], boundary_nodes[(i + 1) % len(boundary_nodes)])
            for i in range(len(boundary_nodes))
        ]

        face_id = f"face_{face_counter}"
        face_counter += 1

        # Crude label for now
        if len(set(boundary_nodes)) == len(primal_embedding.nodes()):
            label = "Outer"
        else:
            label = "-".join(str(x) for x in boundary_nodes) + " Nexus"

        master["faces"].append(
            {
                "id": face_id,
                "label": label,
                "boundary_nodes": boundary_nodes,
                "half_edges": half_edges,
            }
        )

        add_node(face_id, type="face", label=label)

        for he in half_edges:
            master["halfedge_to_face"][he] = face_id

    # ------------------------------------------------------------------
    # 3. Add border nodes, one per primal undirected edge
    # ------------------------------------------------------------------
    seen_undirected = set()

    for u, v in primal_embedding.edges():
        uv = canonical_edge(u, v)
        if uv in seen_undirected:
            continue
        seen_undirected.add(uv)

        a, b = uv
        border_id = f"{a}-{b} Border"

        add_node(
            border_id,
            type="border",
            label=border_id,
            primal_edge=uv,
        )

        master["primal_edge_to_border"][uv] = border_id

    # ------------------------------------------------------------------
    # 4. Add primal edges: region <-> border
    # ------------------------------------------------------------------
    # Also record which master edge is the primal incidence for each
    # directed primal half-edge endpoint occurrence.
    primal_incidence_edge = {}

    seen_undirected.clear()

    for u, v in primal_embedding.edges():
        uv = canonical_edge(u, v)
        if uv in seen_undirected:
            continue
        seen_undirected.add(uv)

        border_id = master["primal_edge_to_border"][uv]

        e1 = add_edge(u, border_id, "primal", primal_edge=uv, endpoint=u)
        e2 = add_edge(v, border_id, "primal", primal_edge=uv, endpoint=v)

        primal_incidence_edge[(u, uv)] = e1
        primal_incidence_edge[(v, uv)] = e2

    # ------------------------------------------------------------------
    # 5. Add dual edges: face <-> border
    # One per half-edge occurrence on each face boundary
    # ------------------------------------------------------------------
    face_border_edge = {}

    for face in master["faces"]:
        face_id = face["id"]

        for he in face["half_edges"]:
            uv = canonical_edge(*he)
            border_id = master["primal_edge_to_border"][uv]

            e = add_edge(
                face_id,
                border_id,
                "dual",
                primal_halfedge=he,
                primal_edge=uv,
                face=face_id,
            )

            face_border_edge[(face_id, he)] = e

    # ------------------------------------------------------------------
    # 6. Build cyclic order around each region node
    # Inherit order from primal embedding neighbours
    # ------------------------------------------------------------------
    for v in primal_embedding.nodes():
        nbrs = list(primal_embedding.neighbors_cw_order(v))
        rotation = []

        for nbr in nbrs:
            uv = canonical_edge(v, nbr)
            rotation.append(primal_incidence_edge[(v, uv)])

        master["rotation"][v] = rotation

    # ------------------------------------------------------------------
    # 7. Build cyclic order around each face node
    # Follow face boundary order exactly, including repeats
    # ------------------------------------------------------------------
    for face in master["faces"]:
        face_id = face["id"]
        rotation = []

        for he in face["half_edges"]:
            rotation.append(face_border_edge[(face_id, he)])

        master["rotation"][face_id] = rotation

    # ------------------------------------------------------------------
    # 8. Build cyclic order around each border node
    # [region A, face right of (A,B), region B, face right of (B,A)]
    # ------------------------------------------------------------------
    for uv, border_id in master["primal_edge_to_border"].items():
        a, b = uv

        face_ab = master["halfedge_to_face"][(a, b)]
        face_ba = master["halfedge_to_face"][(b, a)]

        e_a = primal_incidence_edge[(a, uv)]
        e_b = primal_incidence_edge[(b, uv)]

        # Need the exact dual-edge IDs for these two face incidences
        e_face_ab = None
        e_face_ba = None

        for edge_id in master["rotation"][face_ab]:
            edge = master["edges"][edge_id]
            if edge["type"] == "dual" and edge["primal_halfedge"] == (a, b):
                if edge["u"] == face_ab or edge["v"] == face_ab:
                    e_face_ab = edge_id
                    break

        for edge_id in master["rotation"][face_ba]:
            edge = master["edges"][edge_id]
            if edge["type"] == "dual" and edge["primal_halfedge"] == (b, a):
                if edge["u"] == face_ba or edge["v"] == face_ba:
                    e_face_ba = edge_id
                    break

        master["rotation"][border_id] = [e_a, e_face_ab, e_b, e_face_ba]

    return master


def get_visual_data(master_embedding, primal_pos, style=None):
    """
    Generate visual data for a master embedding.

    Parameters
    ----------
    master_embedding : dict
        Output from build_master_embedding(...)

    primal_pos : dict
        Mapping from primal region node -> (x, y), typically from
        nx.combinatorial_embedding_to_pos(primal_embedding)

    style : dict or None
        Settings dict. If None, defaults are used.

    Returns
    -------
    visual_data : dict
        {
            "nodes": [
                {"id": ..., "xy": (..., ...), "type": ..., "label": ...},
                ...
            ],
            "edge_paths": [
                {
                    "id": ...,
                    "u": ...,
                    "v": ...,
                    "type": ...,
                    "points": [p0, p1, p2],   # quadratic Bezier control points
                    "midpoint": (..., ...),
                },
                ...
            ],
            "node_labels": [
                {"text": ..., "xy": (..., ...), "node_id": ..., "type": ...},
                ...
            ],
            "face_labels": [
                {"text": ..., "xy": (..., ...), "node_id": ..., "type": ...},
                ...
            ],
        }
    """
    if style is None:
        style = {}

    style = {
        "border_t": 0.5,
        "face_centroid_shrink": 0.88,
        "outer_face_offset": 0.35,
        "outer_face_angle": 135,
        "dual_control_alpha": 0.55,
        "dual_control_beta": 0.18,
        "dual_max_fan_offset": 1.0,
        "primal_control_alpha": 0.5,
        "region_label_offset_y": 0.06,
        "show_node_labels": True,
        "show_face_labels": False,
        **style,
    }

    node_xy = {}

    # ------------------------------------------------------------
    # 1. Region nodes inherit primal positions directly
    # ------------------------------------------------------------
    for node_id, attrs in master_embedding["nodes"].items():
        if attrs["type"] == "region":
            node_xy[node_id] = tuple(primal_pos[node_id])

    # ------------------------------------------------------------
    # 2. Border nodes sit on primal-edge midpoints (or t-point)
    # ------------------------------------------------------------
    for node_id, attrs in master_embedding["nodes"].items():
        if attrs["type"] != "border":
            continue

        a, b = attrs["primal_edge"]
        pa = np.array(node_xy[a], dtype=float)
        pb = np.array(node_xy[b], dtype=float)

        t = style["border_t"]
        p = (1 - t) * pa + t * pb
        node_xy[node_id] = tuple(p)

    # ------------------------------------------------------------
    # 3. Face nodes
    #    - bounded faces: shrunk polygon centroid
    #    - outer face: placed outside the primal drawing
    # ------------------------------------------------------------
    outer_face_id = detect_outer_face_id(master_embedding, primal_pos)
    outer_square = make_outer_square(primal_pos, style)
    node_xy[outer_face_id] = outer_square["centre"]

    for face in master_embedding["faces"]:
        face_id = face["id"]

        if face_id != outer_face_id:
            node_xy[face_id] = place_bounded_face_node(
                face,
                primal_pos,
                shrink=style["face_centroid_shrink"],
            )
    # ------------------------------------------------------------
    # 4. Build node list
    # ------------------------------------------------------------
    nodes = []
    for node_id, attrs in master_embedding["nodes"].items():
        nodes.append(
            {
                "id": node_id,
                "xy": node_xy[node_id],
                "type": attrs["type"],
                "label": attrs.get("label", str(node_id)),
            }
        )

    # ------------------------------------------------------------
    # 5. Build edge paths
    #    Everything is cubic: [start, control1, control2, end]
    # ------------------------------------------------------------
    edge_paths = []

    # Precompute face-local ordering for dual edges
    face_dual_order = {}
    for node_id, attrs in master_embedding["nodes"].items():
        if attrs["type"] != "face":
            continue

        edge_ids = master_embedding["rotation"][node_id]
        dual_ids = [
            eid for eid in edge_ids if master_embedding["edges"][eid]["type"] == "dual"
        ]
        face_dual_order[node_id] = dual_ids

    outer_square = make_outer_square(primal_pos, style)
    node_xy[outer_face_id] = outer_square["centre"]
    outer_ports = assign_outer_square_ports(
        outer_square,
        face_dual_order[outer_face_id],
        master_embedding,
        node_xy,
    )
    for edge_id, edge in master_embedding["edges"].items():
        u = edge["u"]
        v = edge["v"]

        p0 = np.array(node_xy[u], dtype=float)
        p3 = np.array(node_xy[v], dtype=float)

        if edge["type"] == "primal":
            p1, p2 = cubic_controls_for_primal(
                p0,
                p3,
                alpha1=style["primal_control_alpha1"],
                alpha2=style["primal_control_alpha2"],
            )

        elif edge["type"] == "dual":
            if master_embedding["nodes"][u]["type"] == "face":
                face_id = u
                border_id = v
                p_face = p0
                p_border = p3
                face_at_start = True
            else:
                face_id = v
                border_id = u
                p_face = p3
                p_border = p0
                face_at_start = False

            ordered_dual_ids = face_dual_order[face_id]
            idx = ordered_dual_ids.index(edge_id)
            n = len(ordered_dual_ids)

            primal_edge = master_embedding["nodes"][border_id]["primal_edge"]

            if face_id == outer_face_id:
                p_outer_port = np.array(outer_ports[edge_id], dtype=float)

                if face_at_start:
                    p0 = p_outer_port
                    p_face = p0
                    p_border = p3
                else:
                    p3 = p_outer_port
                    p_face = p3
                    p_border = p0

                c_face, c_border = cubic_controls_for_outer_dual(
                    p_face,
                    p_border,
                    primal_edge,
                    primal_pos,
                    idx=idx,
                    n=n,
                    launch_strength=style["outer_launch_strength"],
                    arrival_strength=style["outer_arrival_strength"],
                    tangent_strength=style["outer_tangent_strength"],
                    curve_base=style["outer_curve_base"],
                    distance_scale=style["outer_curve_distance_scale"],
                    distance_power=style["outer_curve_distance_power"],
                    angle_scale=style["outer_curve_angle_scale"],
                )

            else:
                c_face, c_border = cubic_controls_for_dual(
                    p_face,
                    p_border,
                    primal_edge,
                    primal_pos,
                    idx=idx,
                    n=n,
                    launch_strength=style["dual_launch_strength"],
                    arrival_strength=style["dual_arrival_strength"],
                    tangent_strength=style["dual_tangent_strength"],
                )

            if face_at_start:
                p1, p2 = c_face, c_border
            else:
                p1, p2 = c_border, c_face
        else:
            p1, p2 = cubic_controls_for_primal(
                p0,
                p3,
                alpha1=style["primal_control_alpha1"],
                alpha2=style["primal_control_alpha2"],
            )

        midpoint = cubic_bezier_point(p0, p1, p2, p3, t=0.5)

        edge_paths.append(
            {
                "id": edge_id,
                "u": u,
                "v": v,
                "type": edge["type"],
                "points": [tuple(p0), tuple(p1), tuple(p2), tuple(p3)],
                "midpoint": tuple(midpoint),
            }
        )
    # ------------------------------------------------------------
    # 6. Labels
    # ------------------------------------------------------------
    node_labels = []
    face_labels = []

    if style["show_node_labels"]:
        for node_id, attrs in master_embedding["nodes"].items():
            if attrs["type"] != "region":
                continue

            x, y = node_xy[node_id]
            node_labels.append(
                {
                    "text": attrs.get("label", str(node_id)),
                    "xy": (x, y + style["region_label_offset_y"]),
                    "node_id": node_id,
                    "type": "region",
                }
            )

    if style["show_face_labels"]:
        for node_id, attrs in master_embedding["nodes"].items():
            if attrs["type"] != "face":
                continue

            face_labels.append(
                {
                    "text": attrs.get("label", str(node_id)),
                    "xy": node_xy[node_id],
                    "node_id": node_id,
                    "type": "face",
                }
            )

    return {
        "nodes": nodes,
        "edge_paths": edge_paths,
        "node_labels": node_labels,
        "face_labels": face_labels,
        "outer_square": outer_square,
    }


def draw_visual_data(visual_data, style=None):
    if style is None:
        style = {}

    # Defaults
    style = {
        "region_node_facecolor": "#ffffff",
        "region_node_edgecolor": "#444444",
        "border_node_facecolor": "#dddddd",
        "border_node_edgecolor": "#666666",
        "face_node_facecolor": "#ffcccc",
        "face_node_edgecolor": "#aa4444",
        "primal_edge_color": "#444444",
        "dual_edge_color": "#cc4444",
        "region_node_size": 300,
        "border_node_size": 120,
        "face_node_size": 200,
        "edge_width": 1.5,
        "font_family": "Open Sans",
        "font_size": 10,
        "label_color": "#222222",
        "label_bbox_alpha": 0.8,
        **style,
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # ------------------------------------------------------------
    # 1. Draw edges (cubic Bezier)
    # ------------------------------------------------------------
    for edge in visual_data["edge_paths"]:
        p0, p1, p2, p3 = edge["points"]

        verts = [p0, p1, p2, p3]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

        path = Path(verts, codes)

        color = (
            style["primal_edge_color"]
            if edge["type"] == "primal"
            else style["dual_edge_color"]
        )

        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=style["edge_width"],
            alpha=0.9,
        )
        ax.add_patch(patch)

    # ------------------------------------------------------------
    # 2. Draw nodes by type
    # ------------------------------------------------------------
    node_groups = {
        "region": [],
        "border": [],
        "face": [],
    }

    outer_face_id = None
    for node in visual_data["nodes"]:
        if node["type"] == "face" and node["label"] == "Outer":
            outer_face_id = node["id"]

    for node in visual_data["nodes"]:
        if node["id"] == outer_face_id:
            continue
        node_groups[node["type"]].append(node)

    def draw_nodes(nodes, facecolor, edgecolor, size):
        if not nodes:
            return
        xs = [float(n["xy"][0]) for n in nodes]
        ys = [float(n["xy"][1]) for n in nodes]

        ax.scatter(
            xs,
            ys,
            s=size,
            c=facecolor,
            edgecolors=edgecolor,
            linewidths=style["edge_width"],
            zorder=3,
        )

    draw_nodes(
        node_groups["region"],
        style["region_node_facecolor"],
        style["region_node_edgecolor"],
        style["region_node_size"],
    )

    draw_nodes(
        node_groups["border"],
        style["border_node_facecolor"],
        style["border_node_edgecolor"],
        style["border_node_size"],
    )

    draw_nodes(
        node_groups["face"],
        style["face_node_facecolor"],
        style["face_node_edgecolor"],
        style["face_node_size"],
    )

    # ------------------------------------------------------------
    # 3. Draw labels
    # ------------------------------------------------------------
    for label in visual_data.get("node_labels", []):
        x, y = label["xy"]

        ax.text(
            float(x),
            float(y),
            label["text"],
            fontsize=style["font_size"],
            family=style["font_family"],
            color=style["label_color"],
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=style["label_bbox_alpha"],
                edgecolor="none",
            ),
            zorder=4,
        )

    for label in visual_data.get("face_labels", []):
        x, y = label["xy"]

        ax.text(
            float(x),
            float(y),
            label["text"],
            fontsize=style["font_size"],
            family=style["font_family"],
            color="#aa0000",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=style["label_bbox_alpha"],
                edgecolor="none",
            ),
            zorder=4,
        )
    # ------------------------------------------------------------
    # 4. Draw outer square if present
    # ------------------------------------------------------------

    outer_square = visual_data.get("outer_square")

    if outer_square is not None:
        left = outer_square["left"]
        bottom = outer_square["bottom"]
        side = outer_square["side"]

        rect = Rectangle(
            (left, bottom),
            side,
            side,
            facecolor="none",
            edgecolor=style.get("outer_square_color", style["dual_edge_color"]),
            linewidth=style.get("outer_square_width", style["edge_width"]),
            linestyle=style.get("outer_square_linestyle", "-"),
            zorder=2,
        )
        ax.add_patch(rect)

    # ------------------------------------------------------------
    # 5. Final formatting
    # ------------------------------------------------------------
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    return fig, ax


VIS_STYLE = {
    # Border nodes sit at this fraction along the primal edge
    "border_t": 0.5,
    # Face-node placement
    "face_centroid_shrink": 0.88,  # pull bounded-face centroids slightly inward
    # Labels
    "region_label_offset_y": 0.3,
    "show_node_labels": True,
    "show_face_labels": False,
    # Outer face curves and margin
    "outer_curve_base": 0.00,  # minimum outward bend even for nearby/easy edges
    "outer_curve_distance_scale": 0.5,  # how much distance matters
    "outer_curve_distance_power": 1,  # how nonlinear the distance effect is
    "outer_curve_angle_scale": 0,  # how much extra bend you add when the outer node is at an awkward angle
    "primal_control_alpha1": 1.0 / 3.0,
    "primal_control_alpha2": 2.0 / 3.0,
    "dual_launch_strength": 0.28,
    "dual_arrival_strength": 0.18,
    "dual_tangent_strength": 0.12,
    "outer_launch_strength": 0.45,
    "outer_arrival_strength": 0.30,
    "outer_tangent_strength": 0.05,
    "outer_square_margin": 0.1,
    "outer_square_side": None,  # optional fixed size; None = derive from primal bbox
}

VIS_STYLE.update(
    {
        # Colours
        "region_node_facecolor": "#ffffff",
        "region_node_edgecolor": "#444444",
        "border_node_facecolor": "#dddddd",
        "border_node_edgecolor": "#666666",
        "face_node_facecolor": "#ffcccc",
        "face_node_edgecolor": "#aa4444",
        "primal_edge_color": "#444444",
        "dual_edge_color": "#cc4444",
        # Sizes
        "region_node_size": 300,
        "border_node_size": 0,
        "face_node_size": 0,
        "edge_width": 1.5,
        # Labels
        "font_family": "Open Sans",
        "font_size": 10,
        "label_color": "#222222",
        "label_bbox_alpha": 0,
        "outer_square_color": "#cc4444",
        "outer_square_width": 1.5,
        "outer_square_linestyle": "-",
    }
)

labels, matrix = read_adjacency_matrix_from_excel("20260416-Test with bridge.xlsx")
print("Labels:", labels)
print("Matrix:\n", matrix)

G = graph_from_adjacency_matrix(labels, matrix)

is_planar, primal_embedding = get_planar_embedding(G)

master_embedding = build_master_embedding(primal_embedding)

seed_pos = nx.combinatorial_embedding_to_pos(primal_embedding)

primal_pos = nx.spring_layout(
    G,
    pos=seed_pos,
    fixed=None,
    iterations=200,
    k=1.2,
    seed=1,
)

visual_data = get_visual_data(master_embedding, primal_pos, style=VIS_STYLE)
print("Visual data:", visual_data)

fig, ax = draw_visual_data(visual_data, style=VIS_STYLE)
plt.show()
