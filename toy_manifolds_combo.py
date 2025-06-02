"""
Generate toy manifolds for testing homeomorphisms with figures.
"""

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
import math
import os
import argparse
from src_toy_manifolds import (
    swiss_roll,
    dog_ear,
    ribbon,
    cylinder,
    split,
    hole,
    pinch,
    collapse,
)

# Configure matplotlib for figures
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.2,
        "axes.grid": False,
        "figure.figsize": (12, 7),
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    }
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate initial 2D grid points
def generate_grid_points(n, x_range=(-1, 1), y_range=(-1, 1)):
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    distance_to_diag = np.abs(points[:, 0] - points[:, 1])
    points *= 1 - distance_to_diag[:, np.newaxis] * 1e-8
    return points


# Calculate triangle area - vectorized
def triangle_area(vertices):
    # For 2D triangles
    if vertices.shape[-1] == 2:
        # Handle both individual triangles and arrays of triangles
        if vertices.ndim == 2:  # Single triangle
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]
            return 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
        else:  # Multiple triangles
            # Extract coordinates for all triangles at once
            x1, y1 = vertices[:, 0, 0], vertices[:, 0, 1]
            x2, y2 = vertices[:, 1, 0], vertices[:, 1, 1]
            x3, y3 = vertices[:, 2, 0], vertices[:, 2, 1]
            return 0.5 * np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

    # For 3D triangles
    elif vertices.shape[-1] == 3:
        if vertices.ndim == 2:  # Single triangle
            # Get two edges of the triangle as vectors
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            # Calculate the cross product and its magnitude
            cross = np.cross(v1, v2)
            return 0.5 * np.linalg.norm(cross)
        else:  # Multiple triangles
            # Calculate edges for all triangles at once
            v1 = vertices[:, 1] - vertices[:, 0]  # First edge for all triangles
            v2 = vertices[:, 2] - vertices[:, 0]  # Second edge for all triangles
            # Calculate cross products for all triangles
            cross = np.cross(v1, v2)
            # Calculate magnitudes of cross products
            return 0.5 * np.linalg.norm(cross, axis=1)
    else:
        raise ValueError("Vertices must be either 2D or 3D")


# Color generation based on complex number representation
def get_complex_colors(points):
    z = points[:, 0] + 1j * points[:, 1]
    phase = np.angle(z)
    magnitude = np.abs(z)

    h = (phase + np.pi) / (2 * np.pi)
    s = np.full_like(h, 0.9)
    v = ((2.0 / np.pi) * np.arctan(magnitude) + 1) / 2.0
    v = (v - v.min() + 0.1) / v.max()

    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = np.zeros((len(points), 3))
    masks = [i == k for k in range(6)]
    rgb[masks[0]] = np.column_stack([v, t, p])[masks[0]]
    rgb[masks[1]] = np.column_stack([q, v, p])[masks[1]]
    rgb[masks[2]] = np.column_stack([p, v, t])[masks[2]]
    rgb[masks[3]] = np.column_stack([p, q, v])[masks[3]]
    rgb[masks[4]] = np.column_stack([t, p, v])[masks[4]]
    rgb[masks[5]] = np.column_stack([v, p, q])[masks[5]]

    return rgb


# Main plotting function
def create_visualization(
    output_dir="figures",
    output_name="fig_points_manifolds_combo",
    area_ratio_threshold=6,
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate points and triangulation
    points_np = generate_grid_points(20)
    tri = Delaunay(points_np)
    points = torch.tensor(points_np, dtype=torch.float32, device=device)

    # Calculate original triangle areas
    orig_triangles = points_np[tri.simplices]
    orig_areas = triangle_area(orig_triangles)

    # Calculate colors
    point_colors = get_complex_colors(points_np)
    triangle_colors = np.mean(point_colors[tri.simplices], axis=1)

    # Initialize figure - 4x2 grid for the 8 transformations (no identity)
    fig = plt.figure(figsize=(12, 8))

    # Define concise, non-LaTeX equations for each transformation
    transform_equations = {
        "F1 (Swiss Roll)": "h(t) = [u(1-t)/2 + t(1+u)cos(3πtu)/2, y, t(1+u)sin(3πtu)/2]   u=(x+1)/2",
        "F2 (Dog Ear)": "h(t) = [x,y,0] if y≤1-x; else rotate([x,y,0]-c, θ(t))+c   c=[0.5,0.5,0]",
        "F3 (Ribbon)": "h(t) = [x, ty·cos(y'sin(tπ/2))+(1-t)y, ysin(y'sin(tπ/2))+0.5t²]   y'=0.75tπ(y+1)+π/4",
        "F4 (Cylinder)": "h(t) = [(1-t)x+tcos(xπ), (1-t)y+tsin(xπ), ty]",
        "F5 (Split)": "h(t) = [x, 0.5t+y(1-0.5t)] if y≥0; [x, -0.5t+y(1-0.5t)] if y<0",
        "F6 (Hole)": "h(t) = [x+0.5t(1-|x|)x/(r+0.01), y+0.5t(1-|y|)y/(r+0.01)]   r=√(x²+y²)",
        "F7 (Pinch)": "h(t) = [x, y|x|^(2t)]",
        "F8 (Collapse)": "h(t) = [x, (1-t)y]",
    }

    # Transformation configurations with simple labels
    transforms = [
        ("F1 (Swiss Roll)", swiss_roll, True, (0, 0)),
        ("F2 (Dog Ear)", dog_ear, True, (0, 1)),
        ("F3 (Ribbon)", ribbon, True, (0, 2)),
        ("F4 (Cylinder)", cylinder, True, (0, 3)),
        ("F5 (Split)", split, False, (1, 0)),
        ("F6 (Hole)", hole, False, (1, 1)),
        ("F7 (Pinch)", pinch, False, (1, 2)),
        ("F8 (Collapse)", collapse, False, (1, 3)),
    ]

    alpha = 0.7  # Slightly higher alpha for better visibility

    # Plot transformations in a 2x4 grid (no identity)
    for name, func, is_3d, pos in transforms:
        transformed = func(points).cpu().numpy()
        ax = fig.add_subplot(
            2, 4, pos[0] * 4 + pos[1] + 1, projection="3d" if is_3d else None
        )

        if is_3d:
            vertices = transformed[tri.simplices]
            trans_areas = triangle_area(vertices)
            area_ratios = trans_areas / orig_areas
            valid_mask = area_ratios <= area_ratio_threshold

            if np.any(valid_mask):
                valid_vertices = vertices[valid_mask]
                valid_colors = triangle_colors[valid_mask]
                poly3d = Poly3DCollection(
                    valid_vertices,
                    facecolors=valid_colors,
                    alpha=alpha,
                    edgecolors="black",
                    linewidths=0.2,
                )
                ax.add_collection3d(poly3d)

            ax.view_init(elev=30, azim=45)
            ax.set(
                xlim=(-1.1, 1.1),
                ylim=(-1.1, 1.1),
                zlim=(-1.1, 1.1),
                xticks=[],
                yticks=[],
                zticks=[],
            )
            # Make panes transparent for cleaner look
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("lightgray")
            ax.grid(False)
        else:
            vertices_2d = transformed[tri.simplices][:, :, :2]
            trans_areas = triangle_area(vertices_2d)
            area_ratios = trans_areas / orig_areas
            valid_mask = area_ratios <= area_ratio_threshold

            if np.any(valid_mask):
                valid_vertices = vertices_2d[valid_mask]
                valid_colors = triangle_colors[valid_mask]
                poly2d = PolyCollection(
                    valid_vertices,
                    facecolors=valid_colors,
                    alpha=alpha,
                    edgecolors="black",
                    linewidths=0.2,
                )
                ax.add_collection(poly2d)
            ax.set(
                aspect="equal",
                xlim=(-1.1, 1.1),
                ylim=(-1.1, 1.1),
                xticks=[],
                yticks=[],
            )

        # Add transformation name as title
        ax.set_title(name, fontsize=12, fontweight="bold", pad=5)

    # Add main figure title
    fig.suptitle(
        "Homotopy Transformations (t = 1)", fontsize=16, fontweight="bold", y=0.98
    )

    # Create external legend figure with equations
    legend_height = 2.5  # inches
    legend_fig = plt.figure(figsize=(12, legend_height))

    # Create a single axis that covers the whole figure for the equation table
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")  # Hide axis

    # Create a table for equations - 4 columns, 2 rows
    cell_text = []
    cell_colors = []

    # Row 1: F1-F4
    row1 = [
        f"F1 (Swiss Roll):\n{transform_equations['F1 (Swiss Roll)']}",
        f"F2 (Dog Ear):\n{transform_equations['F2 (Dog Ear)']}",
        f"F3 (Ribbon):\n{transform_equations['F3 (Ribbon)']}",
        f"F4 (Cylinder):\n{transform_equations['F4 (Cylinder)']}",
    ]
    cell_text.append(row1)
    cell_colors.append(["#f8f8f8", "#f8f8f8", "#f8f8f8", "#f8f8f8"])

    # Row 2: F5-F8
    row2 = [
        f"F5 (Split):\n{transform_equations['F5 (Split)']}",
        f"F6 (Hole):\n{transform_equations['F6 (Hole)']}",
        f"F7 (Pinch):\n{transform_equations['F7 (Pinch)']}",
        f"F8 (Collapse):\n{transform_equations['F8 (Collapse)']}",
    ]
    cell_text.append(row2)
    cell_colors.append(["#f8f8f8", "#f8f8f8", "#f8f8f8", "#f8f8f8"])

    # Create and customize table
    table = legend_ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for key, cell in table.get_celld().items():
        cell.set_text_props(
            fontproperties=plt.rcParams["font.monospace"][0], linespacing=1.5
        )
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        cell.set_height(0.45)  # Adjust cell height

    # Save equations legend separately
    legend_path = os.path.join(output_dir, f"{output_name}_equations")
    legend_fig.savefig(f"{legend_path}.pdf", bbox_inches="tight", dpi=300)
    legend_fig.savefig(f"{legend_path}.png", bbox_inches="tight", dpi=300)
    plt.close(legend_fig)

    # Save main figure without equations
    for fmt in ("pdf", "png"):
        output_path = os.path.join(output_dir, f"{output_name}.{fmt}")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"Figure created successfully in {output_dir}!")
    print(f"Equation legend created separately as {output_name}_equations.pdf/png")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate toy manifold visualizations with equations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures (default: figures)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fig_points_manifolds_combo",
        help="Base filename for output figures (default: fig_points_manifolds_combo)",
    )
    parser.add_argument(
        "--area-ratio",
        type=float,
        default=6.0,
        help="Maximum area ratio threshold for triangle filtering (default: 6.0)",
    )
    args = parser.parse_args()

    create_visualization(
        output_dir=args.output_dir,
        output_name=args.output_name,
        area_ratio_threshold=args.area_ratio,
    )
