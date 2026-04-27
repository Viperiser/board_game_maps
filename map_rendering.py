import pandas as pd
import numpy as np
import networkx as nx
import math
from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgba

import textwrap

import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "path"
mpl.rcParams["pdf.fonttype"] = 42

# ======================================================================
# Helpers
# ======================================================================


def wrap_label(text, width):
    """
    Wrap label text without splitting words.
    """
    if width is None or width <= 0:
        return text

    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _embedding_faces(embedding):
    """Return all faces of a PlanarEmbedding as lists of nodes."""
    counted_half_edges = set()
    faces = []

    for v in embedding:
        for w in embedding.neighbors_cw_order(v):
            if (v, w) not in counted_half_edges:
                face = embedding.traverse_face(v, w, counted_half_edges)
                faces.append(face)

    return faces


def _outer_face_size_score(embedding):
    faces = _embedding_faces(embedding)
    return -max(len(face) for face in faces)


def choose_outer_face_id(master_embedding):
    """
    Choose the outer face by longest boundary walk.

    Bridges are intentionally double-counted because they appear twice
    in the face boundary walk.
    """
    if not master_embedding["faces"]:
        raise ValueError("master_embedding has no faces")

    outer_face = max(
        master_embedding["faces"],
        key=lambda face: len(face["half_edges"]),
    )

    return outer_face["id"]


def cyclic_orders_match(expected, actual):
    """
    Return True if actual is a cyclic rotation of expected.
    """
    if len(expected) != len(actual):
        return False
    if not expected:
        return True

    n = len(expected)
    for shift in range(n):
        if all(expected[i] == actual[(i + shift) % n] for i in range(n)):
            return True
    return False


def geometric_neighbor_order(node, G, pos):
    """
    Return neighbours of node in clockwise geometric order.
    """
    x0, y0 = pos[node]
    nbrs = list(G.neighbors(node))

    def angle(nbr):
        x, y = pos[nbr]
        return math.atan2(y - y0, x - x0)

    # atan2 gives CCW from -pi to pi; reverse for CW
    return sorted(nbrs, key=angle, reverse=True)


def embedding_is_preserved_locally(node, G, primal_embedding, pos):
    """
    Check that the cyclic neighbour order around node matches the
    combinatorial embedding, up to rotation.
    """
    expected = list(primal_embedding.neighbors_cw_order(node))
    actual = geometric_neighbor_order(node, G, pos)
    return cyclic_orders_match(expected, actual)


def move_preserves_embedding(node, G, primal_embedding, pos):
    """
    Check moved node and its neighbours.
    """
    to_check = {node, *G.neighbors(node)}
    for v in to_check:
        if not embedding_is_preserved_locally(v, G, primal_embedding, pos):
            return False
    return True


def tutte_embedding_from_outer_face(G, outer_face_nodes, radius=1.0):
    """
    Compute a Tutte-style barycentric embedding of planar graph G.

    Parameters
    ----------
    G : nx.Graph
        The primal graph.
    outer_face_nodes : list
        Outer-face nodes in cyclic order.
    radius : float
        Radius of the circle on which to place the outer-face nodes.

    Returns
    -------
    pos : dict
        Mapping node -> np.array([x, y], dtype=float)
    """
    outer = list(outer_face_nodes)

    if len(outer) != len(set(outer)):
        raise ValueError("outer_face_nodes contains duplicates")

    missing = [v for v in outer if v not in G]
    if missing:
        raise ValueError(f"outer_face_nodes not all in G: {missing}")

    pos = {}

    # Place boundary nodes equally spaced on a circle
    n_outer = len(outer)
    for i, v in enumerate(outer):
        theta = 2 * math.pi * i / n_outer
        pos[v] = np.array(
            [radius * math.cos(theta), radius * math.sin(theta)],
            dtype=float,
        )

    interior = [v for v in G.nodes() if v not in pos]
    if not interior:
        return pos

    index = {v: i for i, v in enumerate(interior)}
    n_int = len(interior)

    A = np.zeros((n_int, n_int), dtype=float)
    bx = np.zeros(n_int, dtype=float)
    by = np.zeros(n_int, dtype=float)

    for v in interior:
        i = index[v]
        nbrs = list(G.neighbors(v))
        deg = len(nbrs)

        if deg == 0:
            raise ValueError(f"Interior node {v!r} has degree 0")

        A[i, i] = 1.0

        for u in nbrs:
            w = 1.0 / deg

            if u in index:
                j = index[u]
                A[i, j] -= w
            else:
                bx[i] += w * pos[u][0]
                by[i] += w * pos[u][1]

    x = np.linalg.solve(A, bx)
    y = np.linalg.solve(A, by)

    for v in interior:
        i = index[v]
        pos[v] = np.array([x[i], y[i]], dtype=float)

    return pos


def point_in_polygon(point, polygon):
    """
    Ray-casting point-in-polygon test.
    polygon: sequence of (x, y)
    point: (x, y)
    """
    x, y = point
    pts = np.asarray(polygon, dtype=float)
    inside = False
    n = len(pts)

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]

        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside

    return inside


def node_angle_penalty(G, pos):
    """
    Penalise uneven angular spacing of incident edges around each node.
    Lower is better.
    """
    penalty = 0.0

    for u in G.nodes():
        neighbours = list(G.neighbors(u))
        d = len(neighbours)

        if d < 2:
            continue

        p = np.asarray(pos[u], dtype=float)

        angles = []
        for v in neighbours:
            q = np.asarray(pos[v], dtype=float)
            dx, dy = q - p
            ang = math.atan2(dy, dx)
            angles.append(ang)

        angles.sort()

        gaps = []
        for i in range(d):
            a1 = angles[i]
            a2 = angles[(i + 1) % d]
            gap = a2 - a1
            if i == d - 1:
                gap += 2 * math.pi
            gaps.append(gap)

        target = 2 * math.pi / d
        penalty += sum((gap - target) ** 2 for gap in gaps)

    return float(penalty)


def outer_face_roundness_penalty(pos, outer_face_nodes, concavity_weight=5.0):
    """
    Penalise outer-face shapes that are uneven in radius and that contain
    concave inward dents.

    Lower is better.
    """
    pts = np.array([pos[n] for n in outer_face_nodes], dtype=float)

    # --------------------------------------------------
    # 1. Existing radial-variance term
    # --------------------------------------------------
    centre = pts.mean(axis=0)
    radii = np.linalg.norm(pts - centre, axis=1)
    radius_var = float(np.var(radii))

    # --------------------------------------------------
    # 2. New concavity penalty
    #    Penalise turns that go the "wrong way" around the boundary
    # --------------------------------------------------
    area2 = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1

    orientation = 1.0 if area2 >= 0 else -1.0

    concavity_penalty = 0.0
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p = pts[i]
        p_next = pts[(i + 1) % n]

        v1 = p - p_prev
        v2 = p_next - p

        cross_z = v1[0] * v2[1] - v1[1] * v2[0]

        # If the signed turn disagrees with the polygon orientation,
        # this is a concave kink. Penalise its magnitude.
        bad_turn = -orientation * cross_z
        if bad_turn > 0:
            concavity_penalty += bad_turn

    return radius_var + concavity_weight * concavity_penalty


def extract_faces_from_embedding(primal_embedding):
    """
    Return list of faces, each as an ordered list of nodes.
    Deterministic ordering.
    """
    seen_half_edges = set()
    faces = []

    for u in sorted(primal_embedding.nodes(), key=str):
        for v in primal_embedding.neighbors_cw_order(u):
            if (u, v) in seen_half_edges:
                continue
            face = list(primal_embedding.traverse_face(u, v, seen_half_edges))
            faces.append(face)

    return faces


def segments_intersect(p1, p2, q1, q2, eps=1e-12):
    """
    Proper segment intersection test, ignoring shared endpoints.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    # Proper crossing
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    # Collinear edge cases
    if abs(o1) <= eps and on_segment(p1, p2, q1):
        return True
    if abs(o2) <= eps and on_segment(p1, p2, q2):
        return True
    if abs(o3) <= eps and on_segment(q1, q2, p1):
        return True
    if abs(o4) <= eps and on_segment(q1, q2, p2):
        return True

    return False


def layout_has_crossing(G, pos):
    """
    True if any two non-adjacent edges cross.
    """
    edges = list(G.edges())

    for i, (a, b) in enumerate(edges):
        p1 = pos[a]
        p2 = pos[b]

        for j in range(i + 1, len(edges)):
            c, d = edges[j]

            # Ignore edges sharing a vertex
            if len({a, b, c, d}) < 4:
                continue

            q1 = pos[c]
            q2 = pos[d]

            if segments_intersect(p1, p2, q1, q2):
                return True

    return False


def polygon_min_angle(points):
    """
    Minimum angle at vertices of a polygon (in radians).
    Used as a proxy for 'skinniness'.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)

    if n < 3:
        return 0.0

    best = float("inf")

    for i in range(n):
        a = pts[i - 1]
        b = pts[i]
        c = pts[(i + 1) % n]

        u = a - b
        v = c - b

        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)

        if nu < 1e-12 or nv < 1e-12:
            continue

        cosang = np.dot(u, v) / (nu * nv)
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.acos(cosang)

        best = min(best, ang)

    if best == float("inf"):
        return 0.0

    return best


def layout_score(G, pos, faces, weights, outer_face_nodes=None):
    """
    Higher is better.
    """

    nodes = list(G.nodes())
    edges = list(G.edges())

    # --------------------------------------------------
    # 1. Node spread: reward separation
    # --------------------------------------------------
    min_node_dist = float("inf")
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            d = np.linalg.norm(np.asarray(pos[u]) - np.asarray(pos[v]))
            min_node_dist = min(min_node_dist, d)

    if min_node_dist == float("inf"):
        min_node_dist = 0.0

    # --------------------------------------------------
    # 2. Edge uniformity: penalise high variance
    # --------------------------------------------------
    edge_lengths = [
        np.linalg.norm(np.asarray(pos[u]) - np.asarray(pos[v])) for u, v in edges
    ]
    if edge_lengths:
        edge_var = float(np.var(edge_lengths))
    else:
        edge_var = 0.0

    # --------------------------------------------------
    # 3. Face area: reward larger bounded faces
    # --------------------------------------------------
    face_areas = []
    for face in faces:
        pts = [pos[n] for n in face]
        area = abs(polygon_signed_area(pts))
        face_areas.append(area)

    if face_areas:
        max_area = max(face_areas)
        bounded_areas = [a for a in face_areas if a < max_area - 1e-12]
        mean_bounded_area = np.mean(bounded_areas) if bounded_areas else 0.0
    else:
        mean_bounded_area = 0.0

    # --------------------------------------------------
    # 4. Angle penalty: penalise uneven edge fans at nodes
    # --------------------------------------------------
    angle_penalty = node_angle_penalty(G, pos)

    outer_roundness_penalty = 0.0
    if outer_face_nodes is not None and len(outer_face_nodes) >= 3:
        outer_roundness_penalty = outer_face_roundness_penalty(
            pos, outer_face_nodes, concavity_weight=weights.get("outer_concavity", 20.0)
        )

    score = (
        weights["node_spread"] * min_node_dist
        - weights["edge_uniformity"] * edge_var
        + weights["face_area"] * mean_bounded_area
        - weights["angle_penalty"] * angle_penalty
        - weights["outer_roundness"] * outer_roundness_penalty
    )

    return float(score)


def point_to_polygon_boundary_distance(point, polygon):
    """
    Minimum distance from point to any polygon edge.
    """
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)

    return min(
        point_to_segment_distance(point, pts[i], pts[(i + 1) % n]) for i in range(n)
    )


def interior_point_of_polygon(points, grid_size=25):
    """
    Find a point inside a polygon.

    Strategy:
    - use centroid if it lies inside
    - otherwise search a regular grid over the bounding box and choose
      the interior point farthest from the polygon boundary
    """
    pts = np.asarray(points, dtype=float)

    # First try centroid
    c = polygon_centroid(pts)
    if point_in_polygon(c, pts):
        return tuple(c)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    best_point = None
    best_score = -1.0

    xs = np.linspace(min_x, max_x, grid_size)
    ys = np.linspace(min_y, max_y, grid_size)

    for x in xs:
        for y in ys:
            p = (x, y)
            if point_in_polygon(p, pts):
                score = point_to_polygon_boundary_distance(p, pts)
                if score > best_score:
                    best_score = score
                    best_point = p

    if best_point is not None:
        return best_point

    # Very degenerate fallback
    return tuple(pts.mean(axis=0))


def point_to_segment_distance(point, a, b):
    """
    Euclidean distance from point to line segment ab.
    """
    p = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    denom = np.dot(ab, ab)

    if denom < 1e-12:
        return np.linalg.norm(p - a)

    t = np.dot(p - a, ab) / denom
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def place_bounded_face_node(face, primal_pos, shrink=0.88):
    """
    Place a bounded face node at a robust interior point.

    For convex faces, this is usually close to the centroid.
    For concave faces, this avoids placing the face node outside
    the polygon.
    """
    pts = np.array([primal_pos[n] for n in face["boundary_nodes"]], dtype=float)

    p = np.array(interior_point_of_polygon(pts), dtype=float)

    # Optional slight pull toward vertex mean, but only keep it if still inside
    mean_pt = pts.mean(axis=0)
    candidate = shrink * p + (1 - shrink) * mean_pt

    if point_in_polygon(candidate, pts):
        return tuple(candidate)

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


def outer_dual_outward_normal(
    edge, master_embedding, primal_pos, node_xy, outer_face_id
):
    """
    Return the outward unit normal for a specific outer dual edge.

    For a normal outer edge, choose the normal pointing away from the
    adjacent interior face.

    For a bridge, where both sides are outer, use the stored halfedge_sign
    so the two outer incidences get opposite normals.
    """
    a, b = edge["primal_edge"]
    pa = np.array(primal_pos[a], dtype=float)
    pb = np.array(primal_pos[b], dtype=float)

    edge_vec = pb - pa
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return np.array([1.0, 0.0], dtype=float)

    edge_unit = edge_vec / edge_len
    n1 = np.array([-edge_unit[1], edge_unit[0]])
    n2 = -n1

    he = edge["primal_halfedge"]
    opp_he = (he[1], he[0])
    opp_face_id = master_embedding["halfedge_to_face"][opp_he]

    # If the opposite face is not outer, point away from it
    if opp_face_id != outer_face_id:
        p_border = np.array(
            node_xy[master_embedding["primal_edge_to_border"][edge["primal_edge"]]],
            dtype=float,
        )
        p_inner = np.array(node_xy[opp_face_id], dtype=float)

        if np.dot(p_border - p_inner, n1) > np.dot(p_border - p_inner, n2):
            return n1
        else:
            return n2

    # Bridge case: both sides are outer, so use halfedge sign
    if edge["halfedge_sign"] == +1:
        return n1
    else:
        return n2


def assign_outer_square_ports(
    outer_square, ordered_edge_ids, master_embedding, primal_pos, node_xy, outer_face_id
):
    """
    Assign one square-boundary attachment point per outer dual edge.

    Each port is found by casting a ray from the corresponding border node
    outward along that incidence's outward normal.
    """
    ports = {}

    left = outer_square["left"]
    right = outer_square["right"]
    bottom = outer_square["bottom"]
    top = outer_square["top"]

    for edge_id in ordered_edge_ids:
        edge = master_embedding["edges"][edge_id]

        u = edge["u"]
        v = edge["v"]

        if master_embedding["nodes"][u]["type"] == "border":
            border_id = u
        else:
            border_id = v

        p_border = np.array(node_xy[border_id], dtype=float)
        outward = outer_dual_outward_normal(
            edge, master_embedding, primal_pos, node_xy, outer_face_id
        )

        bx, by = p_border
        dx, dy = outward

        candidates = []

        if abs(dx) > 1e-12:
            t_left = (left - bx) / dx
            y_left = by + t_left * dy
            if t_left > 0 and bottom <= y_left <= top:
                candidates.append((t_left, (left, y_left)))

            t_right = (right - bx) / dx
            y_right = by + t_right * dy
            if t_right > 0 and bottom <= y_right <= top:
                candidates.append((t_right, (right, y_right)))

        if abs(dy) > 1e-12:
            t_bottom = (bottom - by) / dy
            x_bottom = bx + t_bottom * dx
            if t_bottom > 0 and left <= x_bottom <= right:
                candidates.append((t_bottom, (x_bottom, bottom)))

            t_top = (top - by) / dy
            x_top = bx + t_top * dx
            if t_top > 0 and left <= x_top <= right:
                candidates.append((t_top, (x_top, top)))
        if not candidates:
            ports[edge_id] = tuple(p_border)
        else:
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
    outward,
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


def _generate_orderings(G, n_trials, seed):
    rng = random.Random(seed)

    nodes = list(G.nodes())

    # deterministic base heuristics
    deg = dict(G.degree())
    try:
        core = nx.core_number(G)
    except:
        core = {v: 0 for v in G.nodes()}

    orderings = []

    # 1. original order
    orderings.append(nodes)

    # 2. low degree first (more "boundary-like")
    orderings.append(sorted(nodes, key=lambda v: (deg[v], str(v))))

    # 3. high degree first
    orderings.append(sorted(nodes, key=lambda v: (-deg[v], str(v))))

    # 4. low core first
    orderings.append(sorted(nodes, key=lambda v: (core[v], str(v))))

    # 5+. random but deterministic
    for i in range(max(0, n_trials - len(orderings))):
        shuffled = nodes[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)

    return orderings[:n_trials]


def normalise_positions(pos, target_span):
    """
    Rescale positions so the largest layout dimension equals target_span.

    Returns a new dict. Does not mutate input.
    """
    pts = np.array(list(pos.values()), dtype=float)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)

    centre = 0.5 * (min_xy + max_xy)
    span_xy = max_xy - min_xy
    span = max(span_xy[0], span_xy[1], 1e-12)

    scale = target_span / span

    return {
        node: tuple((np.asarray(xy, dtype=float) - centre) * scale)
        for node, xy in pos.items()
    }


def apply_dynamic_visual_scale(style, n_nodes):
    """
    Fill derived visual-size entries from base values and node count.
    """
    style = dict(style)

    reference = style["reference_node_count"]
    scale = math.sqrt(reference / max(n_nodes, 1))

    scale = max(style["visual_scale_min"], min(style["visual_scale_max"], scale))

    style["region_node_size"] = style["base_region_node_size"] * scale**2
    style["border_node_size"] = style["base_border_node_size"] * scale**2
    style["face_node_size"] = style["base_face_node_size"] * scale**2

    style["font_size"] = style["base_font_size"] * scale
    style["edge_width"] = style["base_edge_width"] * scale

    # This is still in data units, but now data units are normalised.
    style["region_label_offset_y"] = style["base_region_label_offset_y"] * scale

    return style


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


def get_planar_embedding(
    G,
    require_planar=True,
    n_trials=1,
    seed=42,
    score_fn=None,
):
    """
    Check planarity and return a PlanarEmbedding if planar.
    Optionally try multiple node orderings and pick the best embedding.
    """

    if score_fn is None:
        score_fn = _outer_face_size_score

    best_emb = None
    best_score = None
    is_planar_global = False

    orderings = _generate_orderings(G, n_trials, seed)

    for node_order in orderings:
        H = nx.Graph()
        H.add_nodes_from(node_order)
        H.add_edges_from(G.edges())

        is_planar, emb = nx.check_planarity(H)

        if not is_planar:
            continue

        is_planar_global = True

        score = score_fn(emb)

        if (
            best_emb is None or score < best_score
        ):  # Lower (more negative) is better for this score
            best_emb = emb
            best_score = score

    if require_planar and not is_planar_global:
        raise ValueError("Graph is not planar")

    return is_planar_global, best_emb


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

            if he == uv:
                halfedge_sign = +1
            elif he == (uv[1], uv[0]):
                halfedge_sign = -1
            else:
                raise ValueError(f"Half-edge {he} does not match canonical edge {uv}")

            e = add_edge(
                face_id,
                border_id,
                "dual",
                primal_halfedge=he,
                primal_edge=uv,
                face=face_id,
                halfedge_sign=halfedge_sign,
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


def refine_primal_layout(
    G,
    primal_embedding,
    initial_pos,
    iterations=3000,
    step_scale=0.08,
    temperature_start=0.05,
    temperature_end=0.001,
    fixed_nodes=None,
    weights=None,
    seed=42,
    outer_face_nodes=None,
):
    """
    Optional refinement pass for a planar primal layout.

    Starts from an existing planar layout and makes small random moves
    that try to improve spacing and face quality, while rejecting any
    move that creates an edge crossing.

    Parameters
    ----------
    G : nx.Graph
        The primal graph.

    primal_embedding : nx.PlanarEmbedding
        Embedding used to determine face boundaries.

    initial_pos : dict
        Mapping node -> (x, y), typically from
        nx.combinatorial_embedding_to_pos(...)

    iterations : int
        Number of refinement steps.

    step_scale : float
        Typical size of proposed moves, as a fraction of layout span.

    temperature_start, temperature_end : float
        Simulated annealing schedule. Early on, slightly worse moves can
        be accepted. Later, behaviour becomes greedier.

    fixed_nodes : iterable or None
        Nodes that should not move. Often you may want to pin outer-face
        boundary nodes.

    weights : dict or None
        Weights for score terms.

    seed : int
        Random seed.

    Returns
    -------
    pos : dict
        Refined positions.
    """

    if fixed_nodes is None:
        fixed_nodes = set()
    else:
        fixed_nodes = set(fixed_nodes)

    rng = random.Random(seed)

    pos = {k: np.array(v, dtype=float).copy() for k, v in initial_pos.items()}

    nodes = list(G.nodes())
    movable_nodes = [n for n in nodes if n not in fixed_nodes]
    if not movable_nodes:
        return {k: tuple(v) for k, v in pos.items()}

    faces = extract_faces_from_embedding(primal_embedding)

    pts = np.array(list(pos.values()), dtype=float)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1e-9)
    step = step_scale * span

    current_score = layout_score(
        G, pos, faces, weights=weights, outer_face_nodes=outer_face_nodes
    )

    for t in range(iterations):
        alpha = t / max(iterations - 1, 1)
        temperature = (1 - alpha) * temperature_start + alpha * temperature_end

        node = rng.choice(movable_nodes)
        old_xy = pos[node].copy()

        dx = rng.uniform(-step, step)
        dy = rng.uniform(-step, step)
        pos[node] = old_xy + np.array([dx, dy], dtype=float)

        # Reject if the move creates a crossing
        if layout_has_crossing(G, pos):
            pos[node] = old_xy
            continue

        # Reject if the move changes the embedding
        if not move_preserves_embedding(node, G, primal_embedding, pos):
            pos[node] = old_xy
            continue

        new_score = layout_score(
            G, pos, faces, weights=weights, outer_face_nodes=outer_face_nodes
        )

        delta = new_score - current_score

        if delta >= 0:
            current_score = new_score
        else:
            # Annealing acceptance
            accept_prob = math.exp(delta / max(temperature, 1e-12))
            if rng.random() < accept_prob:
                current_score = new_score
            else:
                pos[node] = old_xy

    print("FINAL SCORE:", current_score)
    print("FINAL POS:")
    for k in sorted(pos, key=str):
        print(k, tuple(round(x, 6) for x in pos[k]))

    return {k: tuple(v) for k, v in pos.items()}


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
    outer_face_id = choose_outer_face_id(master_embedding)
    outer_square = make_outer_square(primal_pos, style)

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
        if node_id == outer_face_id:
            continue

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

    node_xy[outer_face_id] = outer_square["centre"]

    outer_ports = assign_outer_square_ports(
        outer_square,
        face_dual_order[outer_face_id],
        master_embedding,
        primal_pos,
        node_xy,
        outer_face_id,
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

                outward = outer_dual_outward_normal(
                    edge, master_embedding, primal_pos, node_xy, outer_face_id
                )

                c_face, c_border = cubic_controls_for_outer_dual(
                    p_face,
                    p_border,
                    outward,
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
                    "text": wrap_label(
                        attrs.get("label", str(node_id)),
                        (
                            style["label_wrap_width"]
                            if style["label_wrap_enabled"]
                            else None
                        ),
                    ),
                    "xy": (x, y),
                    "node_id": node_id,
                    "type": "region",
                }
            )

    if style["show_face_labels"]:
        for node_id, attrs in master_embedding["nodes"].items():
            if attrs["type"] != "face":
                continue
            if node_id == outer_face_id:
                continue

            face_labels.append(
                {
                    "text": wrap_label(
                        attrs.get("label", str(node_id)),
                        (
                            style["label_wrap_width"]
                            if style["label_wrap_enabled"]
                            else None
                        ),
                    ),
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

    style = {**VIS_STYLE, **style}

    fig, ax = plt.subplots(
        figsize=style["figure_size"],
        dpi=style.get("figure_dpi", 200),
    )
    # ------------------------------------------------------------
    # 1. Draw edges (cubic Bezier)
    # ------------------------------------------------------------
    for edge in visual_data["edge_paths"]:
        if edge["type"] == "primal" and not style["draw_primal_edges"]:
            continue
        if edge["type"] == "dual" and not style["draw_dual_edges"]:
            continue

        p0, p1, p2, p3 = edge["points"]

        verts = [p0, p1, p2, p3]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        path = MplPath(verts, codes)

        edgecolor = (
            style["primal_edge_color"]
            if edge["type"] == "primal"
            else style["dual_edge_color"]
        )

        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=edgecolor,
            lw=style["edge_width"],
            alpha=style["edge_alpha"],
            capstyle=style["edge_capstyle"],
            joinstyle=style["edge_joinstyle"],
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

    node_linewidth = (
        style["edge_width"]
        if style["node_linewidth"] is None
        else style["node_linewidth"]
    )

    def draw_nodes(nodes, facecolor, edgecolor, size):
        if not nodes:
            return

        xs = [float(n["xy"][0]) for n in nodes]
        ys = [float(n["xy"][1]) for n in nodes]

        edge_rgba = to_rgba(edgecolor, style["edge_alpha"])

        ax.scatter(
            xs,
            ys,
            s=size,
            c=facecolor,
            edgecolors=edge_rgba,
            linewidths=node_linewidth,
            zorder=style["node_zorder"],
        )

    if style["draw_primal_edges"]:
        draw_nodes(
            node_groups["region"],
            style["region_node_facecolor"],
            style["primal_edge_color"],
            style["region_node_size"],
        )

    draw_nodes(
        node_groups["border"],
        style["border_node_facecolor"],
        style["border_node_color"],
        style["border_node_size"],
    )

    draw_nodes(
        node_groups["face"],
        style["face_node_facecolor"],
        style["dual_edge_color"],
        style["face_node_size"],
    )

    # ------------------------------------------------------------
    # 3. Draw labels
    # ------------------------------------------------------------
    if style["show_node_labels"]:
        for label in visual_data.get("node_labels", []):
            x, y = label["xy"]
            y = y + style["region_label_offset_y"]

            ax.text(
                float(x),
                float(y),
                label["text"],
                fontsize=style["font_size"],
                family=style["font_family"],
                fontweight=style["label_fontweight"],
                alpha=style["label_alpha"],
                color=style["label_color"],
                ha=style["label_ha"],
                va=style["label_va"],
                bbox=dict(
                    boxstyle=style["label_bbox_boxstyle"],
                    facecolor=style["label_bbox_facecolor"],
                    alpha=style["label_bbox_alpha"],
                    edgecolor=style["label_bbox_edgecolor"],
                ),
                zorder=style["label_zorder"],
                linespacing=style["label_linespacing"],
            )

    if style["show_face_labels"]:
        for label in visual_data.get("face_labels", []):
            x, y = label["xy"]

            ax.text(
                float(x),
                float(y),
                label["text"],
                fontsize=style["font_size"],
                family=style["font_family"],
                color=style["face_label_color"],
                ha=style["label_ha"],
                va=style["label_va"],
                bbox=dict(
                    boxstyle=style["label_bbox_boxstyle"],
                    facecolor=style["label_bbox_facecolor"],
                    alpha=style["label_bbox_alpha"],
                    edgecolor=style["label_bbox_edgecolor"],
                ),
                zorder=style["label_zorder"],
            )

    # ------------------------------------------------------------
    # 4. Final formatting
    # ------------------------------------------------------------
    ax.set_aspect(style["axis_aspect"])

    if not style["axis_visible"]:
        ax.axis("off")

    if style["tight_layout"]:
        plt.tight_layout()

    return fig, ax


def main(filename, weights, refine=True, use_tutte=True):
    labels, matrix = read_adjacency_matrix_from_excel(filename)
    print("Labels:", labels)
    print("Matrix:\n", matrix)

    G = graph_from_adjacency_matrix(labels, matrix)

    is_planar, primal_embedding = get_planar_embedding(G, n_trials=30, seed=42)
    if not is_planar:
        raise ValueError(f"Graph from {filename!r} is not planar")

    print("EMBEDDING SIGNATURE:")
    for u in sorted(primal_embedding.nodes(), key=str):
        print(u, list(primal_embedding.neighbors_cw_order(u)))

    master_embedding = build_master_embedding(primal_embedding)

    # Temporary layout only for identifying the outer face
    primal_pos_tmp = nx.combinatorial_embedding_to_pos(
        primal_embedding,
        fully_triangulate=False,
    )

    print("TEMP INITIAL POS:")
    for k in sorted(primal_pos_tmp, key=str):
        print(k, tuple(round(x, 6) for x in primal_pos_tmp[k]))

    outer_face_id = choose_outer_face_id(master_embedding)

    face_lookup = {face["id"]: face for face in master_embedding["faces"]}
    outer_face_nodes = list(face_lookup[outer_face_id]["boundary_nodes"])

    face = face_lookup[outer_face_id]
    print("INITIAL OUTER FACE ID:", outer_face_id)
    print("OUTER FACE BOUNDARY NODES:", face["boundary_nodes"])
    print("OUTER FACE HALF EDGES:")
    for he in face["half_edges"]:
        opp = (he[1], he[0])
        print(
            he,
            "face(he) =",
            master_embedding["halfedge_to_face"].get(he),
            "face(opp) =",
            master_embedding["halfedge_to_face"].get(opp),
        )

    if use_tutte:
        primal_pos = tutte_embedding_from_outer_face(
            G,
            outer_face_nodes,
            radius=1.0,
        )
        print("USING TUTTE INITIAL POS")
    else:
        primal_pos = primal_pos_tmp
        print("USING COMBINATORIAL EMBEDDING INITIAL POS")

    print("INITIAL POS:")
    for k in sorted(primal_pos, key=str):
        print(k, tuple(round(x, 6) for x in primal_pos[k]))

    if weights is None:
        weights = {
            "node_spread": 0.1,
            "edge_uniformity": 0.0,
            "face_area": 0.1,
            "angle_penalty": 1.0,
            "outer_roundness": 0.0,
            "outer_concavity": 100.0,
        }

    print("WEIGHTS IN MAIN:", weights)

    if refine:
        primal_pos = refine_primal_layout(
            G,
            primal_embedding,
            primal_pos,
            iterations=4000,
            step_scale=0.04,
            temperature_start=0.02,
            temperature_end=0.0005,
            fixed_nodes=None,
            weights=weights,
            outer_face_nodes=outer_face_nodes,
            seed=42,
        )

    print(
        "OUTER FACE ID AFTER REFINEMENT:",
        choose_outer_face_id(master_embedding),
    )

    primal_pos = normalise_positions(
        primal_pos,
        target_span=VIS_STYLE["target_layout_span"],
    )

    style = apply_dynamic_visual_scale(
        VIS_STYLE,
        n_nodes=G.number_of_nodes(),
    )

    visual_data = get_visual_data(master_embedding, primal_pos, style=VIS_STYLE)

    base_path = Path(filename)
    stem = base_path.stem

    figures_dir = Path("Figures")
    figures_dir.mkdir(exist_ok=True)

    figure_specs = [
        (
            "fig1",
            {
                "draw_primal_edges": True,
                "draw_dual_edges": False,
                "border_node_size": 0,
                "face_node_size": 0,
            },
            False,
        ),
        (
            "fig2",
            {
                "draw_primal_edges": True,
                "draw_dual_edges": False,
                "border_node_size": 0,
            },
            False,
        ),
        (
            "fig3",
            {
                "draw_primal_edges": True,
                "draw_dual_edges": False,
            },
            True,
        ),
        (
            "fig4",
            {
                "draw_primal_edges": True,
                "draw_dual_edges": True,
            },
            False,
        ),
        (
            "fig5",
            {
                "draw_primal_edges": True,
                "draw_dual_edges": True,
                "border_node_size": 0,
            },
            False,
        ),
        (
            "fig6",
            {
                "draw_primal_edges": False,
                "draw_dual_edges": True,
                "border_node_size": 0,
                "face_node_size": 0,
                "region_label_offset_y": 0,
            },
            False,
        ),
        (
            "fig7",
            {
                "draw_primal_edges": True,
                "draw_dual_edges": True,
                "border_node_size": 0,
                "face_node_size": 0,
            },
            False,
        ),
        (
            "fig8",
            {
                "draw_primal_edges": False,
                "draw_dual_edges": True,
                "border_node_size": 0,
                "region_node_size": 0,
                "region_label_offset_y": 0,
            },
            False,
        ),
    ]

    for suffix, layer_style, show_figure in figure_specs:
        fig, ax = draw_visual_data(
            visual_data,
            style={
                **style,
                **layer_style,
            },
        )

        output_path = figures_dir / f"{stem}-{suffix}.png"
        fig.savefig(output_path, dpi=style["figure_dpi"], bbox_inches="tight")
        print(f"Saved {output_path}")

        if show_figure:
            plt.show()
        else:
            plt.close(fig)


VIS_STYLE = {
    # ------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------
    "target_layout_span": 10.0,
    "reference_node_count": 10,
    "visual_scale_min": 0.35,
    "visual_scale_max": 1.25,
    "base_region_label_offset_y": 0.3,
    "base_region_node_size": 250,
    "base_border_node_size": 120,
    "base_face_node_size": 250,
    "base_font_size": 12,
    "base_edge_width": 2,
    # ------------------------------------------------------------
    # Geometry / construction parameters
    # ------------------------------------------------------------
    "border_t": 0.5,
    "face_centroid_shrink": 0.88,
    "region_label_offset_y": None,
    "show_node_labels": True,
    "show_face_labels": False,
    "outer_curve_base": 0.00,
    "outer_curve_distance_scale": 0.5,
    "outer_curve_distance_power": 1,
    "outer_curve_angle_scale": 0,
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
    # ------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------
    "draw_primal_edges": True,
    "draw_dual_edges": True,
    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    "figure_size": (8, 6),
    "figure_dpi": 200,
    "axis_aspect": "equal",
    "axis_visible": False,
    "tight_layout": True,
    # ------------------------------------------------------------
    # Edge styling
    # ------------------------------------------------------------
    "primal_edge_color": "#1B9E77",
    "dual_edge_color": "#D95F02",
    "border_node_color": "#7570B3",
    "edge_width": None,
    "edge_alpha": 0.7,
    "edge_capstyle": "round",
    "edge_joinstyle": "round",
    # ------------------------------------------------------------
    # Node styling
    # ------------------------------------------------------------
    "region_node_facecolor": "#CFECE4",
    "border_node_facecolor": "#D6D2CB",
    "face_node_facecolor": "#F6D5BF",
    "region_node_size": None,
    "border_node_size": None,
    "face_node_size": None,
    "node_linewidth": None,  # None means use edge_width
    "node_zorder": 3,
    # ------------------------------------------------------------
    # Label styling
    # ------------------------------------------------------------
    "font_family": "Open Sans",
    "font_size": None,
    "label_fontweight": "bold",
    "label_color": "#3A3A3A",
    "label_alpha": 0.85,
    "face_label_color": "#aa0000",
    "label_ha": "center",
    "label_va": "bottom",
    "label_zorder": 4,
    "label_bbox_boxstyle": "round,pad=0.2",
    "label_bbox_facecolor": "#ffffff",
    "label_bbox_alpha": 0,
    "label_bbox_edgecolor": "none",
    "label_wrap_enabled": True,
    "label_wrap_width": 12,
    "label_linespacing": 0.9,
}


#### Notes to future Nick
# PATH is the file with the adjacency matrix in - must be binary and symmetric
# WEIGHTS is a dict of weights for finding the best layout - these need tweaking for different matrices
# VIS_STYLE above is the style dict for the visualisation - tweak as desired, but it should be mostly fine as is for different matrices
# Finally note 'SCALE_PARAM' above the dictionary - this affects font and node sizes and should be tweaked for more or fewer nodes

# Run configuration
PATH = "raw_data/20260423-Low Variance.xlsx"
USE_TUTTE = True  # Whether to use Tutte embedding for initial layout, or just the combinatorial embedding layout. Tutte is often better but can be very slow for larger graphs.
# Tutte will only work if there are no 'bridges' / danglers in the outer face
# Otherwise outer face nodes get repeated and it breaks
REFINE = False  # Whether to run the optional refinement pass after the initial layout. This can improve spacing and face quality, but is also quite slow, especially for larger graphs. If using REFINE=True, you may want to tweak the WEIGHTS below to get better results - the current values are just what worked well for the Hammer of the Scots graph.
WEIGHTS = {
    "node_spread": 0.6,  # Rewards spreading out nodes
    "edge_uniformity": 0,  # Rewards edges of similar length
    "face_area": 0.0,  # Rewards faces having similar area
    "angle_penalty": 0.0,  # Rewards angles that are nicely spread around their nodes
    "outer_roundness": 1,  # Rewards outer face nodes being placed in a more circular arrangement, rather than all bunched up on one side
    "outer_concavity": 100.0,  # Penalises outer face nodes being placed in a concave arrangement, which can lead to weird dual edges that loop around the outside of the drawing
}

if __name__ == "__main__":
    main(PATH, WEIGHTS, REFINE, USE_TUTTE)
