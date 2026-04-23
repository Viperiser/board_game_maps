import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

import numpy as np


def _make_straight_edge(p0, p3):
    """
    Return a cubic-Bezier edge specification that renders as a straight line.
    """
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    c1 = p0 + (p3 - p0) / 3.0
    c2 = p0 + 2.0 * (p3 - p0) / 3.0

    return [
        tuple(p0),
        tuple(c1),
        tuple(c2),
        tuple(p3),
    ]


def _unit_perp(vec):
    """
    Return a unit perpendicular vector to vec.
    Falls back safely if vec is tiny.
    """
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return np.array([0.0, 1.0], dtype=float)
    u = vec / n
    return np.array([-u[1], u[0]], dtype=float)


def _make_cubic_edge(p0, p3, bend=0.0, along=0.28):
    """
    Build a cubic Bezier from p0 to p3.

    Parameters
    ----------
    p0, p3 : array-like
        Start and end points.
    bend : float
        Signed perpendicular offset applied to both control points.
        Positive and negative values bow the curve in opposite directions.
    along : float
        How far the control points sit along the straight-line segment.
    """
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    delta = p3 - p0
    perp = _unit_perp(delta)

    c1 = p0 + along * delta + bend * perp
    c2 = p0 + (1.0 - along) * delta + bend * perp

    return [
        tuple(p0),
        tuple(c1),
        tuple(c2),
        tuple(p3),
    ]


def make_k5_visual_data(radius=3.0):
    """
    Create visual_data for K5 using straight edges.
    """
    labels = ["A", "B", "C", "D", "E"]

    angles = [math.pi / 2 - 2 * math.pi * i / 5 for i in range(5)]
    pos = {
        label: np.array(
            [radius * math.cos(theta), radius * math.sin(theta)], dtype=float
        )
        for label, theta in zip(labels, angles)
    }

    nodes = []
    node_labels = []
    for label in labels:
        xy = tuple(pos[label])
        nodes.append(
            {
                "id": label,
                "label": label,
                "type": "region",
                "xy": xy,
            }
        )
        node_labels.append(
            {
                "id": f"label_{label}",
                "text": label,
                "xy": xy,
            }
        )

    edges = []
    edge_id = 0

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            u = labels[i]
            v = labels[j]
            edges.append(
                {
                    "id": f"e{edge_id}",
                    "u": u,
                    "v": v,
                    "type": "primal",
                    "points": _make_straight_edge(pos[u], pos[v]),
                }
            )
            edge_id += 1

    return {
        "nodes": nodes,
        "edge_paths": edges,
        "node_labels": node_labels,
        "face_labels": [],
        "outer_square": None,
    }


def make_k33_visual_data(x_gap=4.0, y_levels=(2.0, 0.0, -2.0)):
    """
    Create visual_data for K3,3 using straight edges.
    """
    left = ["U1", "U2", "U3"]
    right = ["V1", "V2", "V3"]

    pos = {}
    for label, y in zip(left, y_levels):
        pos[label] = np.array([-x_gap / 2, y], dtype=float)
    for label, y in zip(right, y_levels):
        pos[label] = np.array([+x_gap / 2, y], dtype=float)

    nodes = []
    node_labels = []
    for label in left + right:
        xy = tuple(pos[label])
        nodes.append(
            {
                "id": label,
                "label": label,
                "type": "region",
                "xy": xy,
            }
        )
        node_labels.append(
            {
                "id": f"label_{label}",
                "text": label,
                "xy": xy,
            }
        )

    edges = []
    edge_id = 0

    for u in left:
        for v in right:
            edges.append(
                {
                    "id": f"e{edge_id}",
                    "u": u,
                    "v": v,
                    "type": "primal",
                    "points": _make_straight_edge(pos[u], pos[v]),
                }
            )
            edge_id += 1

    return {
        "nodes": nodes,
        "edge_paths": edges,
        "node_labels": node_labels,
        "face_labels": [],
        "outer_square": None,
    }


def save_nonplanar_illustrations(style=None, out_dir="Figures", close_figs=True):
    """
    Build and save K5 and K3,3 using draw_visual_data().
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagrams = {
        "K5": make_k5_visual_data(),
        "K3_3": make_k33_visual_data(),
    }

    for name, visual_data in diagrams.items():
        fig, ax = draw_visual_data(visual_data, style=style)

        out_path = out_dir / f"{name}.png"
        fig.savefig(
            out_path,
            dpi=(style or {}).get("figure_dpi", VIS_STYLE.get("figure_dpi", 200)),
            bbox_inches="tight",
        )

        if close_figs:
            plt.close(fig)

    print(f"Saved {len(diagrams)} figures to {out_dir.resolve()}")


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
            facecolor=style["outer_square_facecolor"],
            edgecolor=style["outer_square_color"],
            linewidth=style["outer_square_width"],
            linestyle=style["outer_square_linestyle"],
            zorder=style["outer_square_zorder"],
        )
        ax.add_patch(rect)

    # ------------------------------------------------------------
    # 5. Final formatting
    # ------------------------------------------------------------
    ax.set_aspect(style["axis_aspect"])

    if not style["axis_visible"]:
        ax.axis("off")

    if style["tight_layout"]:
        plt.tight_layout()

    return fig, ax


def make_square_triangle_visual_data(triangle_inside=True):
    """
    Create visual_data for the 'square and triangle' graph
    with consistent overall scale in both embeddings.
    """

    # Fixed square (identical in both cases)
    pos = {
        "A": np.array([0.0, 2.0], dtype=float),
        "B": np.array([2.0, 2.0], dtype=float),
        "C": np.array([2.0, 0.0], dtype=float),
        "D": np.array([0.0, 0.0], dtype=float),
    }

    # Place E symmetrically relative to square centre (1,1)
    if triangle_inside:
        pos["E"] = np.array([1.0, 1.2], dtype=float)  # inside
    else:
        pos["E"] = np.array([1.0, 2.8], dtype=float)  # outside but same overall extent

    edge_list = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),
        ("A", "E"),
        ("B", "E"),
    ]

    nodes = []
    node_labels = []
    for label, xy in pos.items():
        xy_t = tuple(xy)
        nodes.append(
            {
                "id": label,
                "label": label,
                "type": "region",
                "xy": xy_t,
            }
        )
        node_labels.append(
            {
                "id": f"label_{label}",
                "text": label,
                "xy": xy_t,
            }
        )

    edges = []
    for i, (u, v) in enumerate(edge_list):
        edges.append(
            {
                "id": f"e{i}",
                "u": u,
                "v": v,
                "type": "primal",
                "points": _make_straight_edge(pos[u], pos[v]),
            }
        )

    return {
        "nodes": nodes,
        "edge_paths": edges,
        "node_labels": node_labels,
        "face_labels": [],
        "outer_square": None,
    }


def save_square_triangle_embeddings(style=None, out_dir="Figures", close_figs=True):
    """
    Save two embeddings of the square-triangle graph:
    - triangle inside the square
    - triangle outside the square
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagrams = {
        "square_triangle_inside": make_square_triangle_visual_data(
            triangle_inside=True
        ),
        "square_triangle_outside": make_square_triangle_visual_data(
            triangle_inside=False
        ),
    }

    for name, visual_data in diagrams.items():
        fig, ax = draw_visual_data(visual_data, style=style)

        out_path = out_dir / f"{name}.png"
        fig.savefig(
            out_path,
            dpi=(style or {}).get("figure_dpi", VIS_STYLE.get("figure_dpi", 200)),
            bbox_inches="tight",
        )

        if close_figs:
            plt.close(fig)

    print(f"Saved {len(diagrams)} figures to {out_dir.resolve()}")


SCALE_PARAM = 1.0  # Global scaling factor for figure elements - tweak as needed for different graphs

VIS_STYLE = {
    # ------------------------------------------------------------
    # Geometry / construction parameters
    # ------------------------------------------------------------
    "border_t": 0.5,
    "face_centroid_shrink": 0.88,
    "region_label_offset_y": 0.2 * SCALE_PARAM,
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
    "primal_edge_color": "#A45A6F",
    "dual_edge_color": "#4A6FA4",
    "edge_width": 2,
    "edge_alpha": 0.7,
    "edge_capstyle": "round",
    "edge_joinstyle": "round",
    # ------------------------------------------------------------
    # Node styling
    # ------------------------------------------------------------
    "region_node_facecolor": "#E7C8CF",
    "region_node_edgecolor": "#A45A6F",
    "border_node_facecolor": "#D6D2CB",
    "border_node_edgecolor": "#6B665E",
    "face_node_facecolor": "#C9D6EA",
    "face_node_edgecolor": "#4A6FA4",
    "region_node_size": 300 * SCALE_PARAM,
    "border_node_size": 150 * SCALE_PARAM,
    "face_node_size": 300 * SCALE_PARAM,
    "node_linewidth": None,  # None means use edge_width
    "node_zorder": 3,
    # ------------------------------------------------------------
    # Label styling
    # ------------------------------------------------------------
    "font_family": "Open Sans",
    "font_size": 20 * SCALE_PARAM,
    "label_fontweight": "bold",
    "label_color": "#3A3A3A",
    "label_alpha": 0.85,
    "face_label_color": "#aa0000",
    "label_ha": "center",
    "label_va": "center_baseline",
    "label_zorder": 4,
    "label_bbox_boxstyle": "round,pad=0.2",
    "label_bbox_facecolor": "#ffffff",
    "label_bbox_alpha": 0,
    "label_bbox_edgecolor": "none",
    # ------------------------------------------------------------
    # Outer square styling
    # ------------------------------------------------------------
    "outer_square_color": "#6A8BB8",
    "outer_square_width": 2,
    "outer_square_linestyle": "-",
    "outer_square_facecolor": "none",
    "outer_square_zorder": 2,
}

# save_nonplanar_illustrations(
#     style={
#         "show_face_labels": False,
#         "draw_dual_edges": True,
#         "show_node_labels": True,
#     }
# )

save_square_triangle_embeddings()
