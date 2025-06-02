"""
Generate toy manifolds for testing homeomorphisms. To quickly draw many triangles
with different colors you have to add them as a collection all at once which is not
commonly done in matplotlib examples online.

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

# Set publication-quality matplotlib parameters
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.grid": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # "text.usetex": True,  # LaTeX rendering disabled
        # big title font
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
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


# Function to create single transformation plot
def create_single_plot(
    fname,
    name,
    func,
    is_3d,
    points,
    tri,
    area_ratio_threshold,
    orig_areas,
    triangle_colors,
    output_dir,
):
    """
    Create an individual plot for a single transformation.

    Args:
        name: Name of the transformation
        func: Transformation function
        is_3d: Whether the transformation produces 3D output
        points: Input points tensor
        tri: Delaunay triangulation
        area_ratio_threshold: Maximum area ratio for filtering triangles
        orig_areas: Original triangle areas
        triangle_colors: Triangle colors
        output_dir: Output directory
    """
    # Create figure
    fig = plt.figure(figsize=(7, 6))
    alpha = 0.6

    # Apply transformation
    transformed = func(points).cpu().numpy()

    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        vertices = transformed[tri.simplices]

        # Calculate transformed triangle areas
        trans_areas = triangle_area(vertices)

        # Calculate area ratios and create mask
        area_ratios = trans_areas / orig_areas
        valid_mask = area_ratios <= area_ratio_threshold

        # Apply mask to filter triangles
        if np.any(valid_mask):
            valid_vertices = vertices[valid_mask]
            valid_colors = triangle_colors[valid_mask]

            poly3d = Poly3DCollection(
                valid_vertices,
                alpha=alpha,
                facecolors=valid_colors,
                edgecolors="black",
                linewidths=0.1,
            )
            ax.add_collection3d(poly3d)

        ax.view_init(elev=30, azim=45)
        ax.set(
            xlim=(-1.1, 1.1),
            ylim=(-1.1, 1.1),
            zlim=(-1.1, 1.1),
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1],
            zticks=[-1, 0, 1],
        )
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("lightgray")
        ax.grid(True, alpha=0.2)
    else:
        ax = fig.add_subplot(111)
        # Use PolyCollection for 2D
        vertices_2d = transformed[tri.simplices][
            :, :, :2
        ]  # Drop z-coordinate if present

        # Calculate transformed triangle areas
        trans_areas = triangle_area(vertices_2d)

        # Calculate area ratios and create mask
        area_ratios = trans_areas / orig_areas
        valid_mask = area_ratios <= area_ratio_threshold

        # Apply mask to filter triangles
        if np.any(valid_mask):
            valid_vertices = vertices_2d[valid_mask]
            valid_colors = triangle_colors[valid_mask]

            poly2d = PolyCollection(
                valid_vertices,
                facecolors=valid_colors,
                alpha=alpha,
                edgecolors="black",
                linewidths=0.1,
            )
            ax.add_collection(poly2d)

        ax.set(
            aspect="equal",
            xlim=(-1.1, 1.1),
            ylim=(-1.1, 1.1),
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1],
        )
        ax.grid(True, alpha=0.2)

    # Set title to the transformation name
    ax.set_title(name, fontsize=30, pad=10)

    # Adjust layout
    plt.tight_layout()

    # Save outputs
    for fmt in ("pdf", "png"):
        output_path = os.path.join(output_dir, f"{fname}.{fmt}")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    plt.close(fig)


# Main plotting function
def create_visualization(
    output_dir="figures", output_name="fig_points_manifolds", area_ratio_threshold=6
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create individual plots directory
    individual_plots_dir = os.path.join(output_dir, "individual_manifold_plots")
    os.makedirs(individual_plots_dir, exist_ok=True)

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

    # Transformation configurations with simple labels
    transforms = [
        ("Original", r"$F_0$ (Original Domain)", lambda p: p, False, None),
        ("Swiss_Roll", r"$F_1$ (SR - Swiss Roll)", swiss_roll, True, None),
        ("Dog_Ear", r"$F_2$ (DE - Dog Ear)", dog_ear, True, None),
        ("Ribbon", r"$F_3$ (RB - Ribbon)", ribbon, True, None),
        ("Cylinder", r"$F_4$ (CY - Cylinder)", cylinder, True, None),
        ("Split", r"$F_5$ (SP - Split)", split, False, None),
        ("Hole", r"$F_6$ (HL - Hole)", hole, False, None),
        ("Pinch", r"$F_7$ (PN - Pinch)", pinch, False, None),
        ("Collapse", r"$F_8$ (CL - Collapse)", collapse, False, None),
    ]

    # Create individual plots for each transformation
    for fname, name, func, is_3d, _ in transforms:
        create_single_plot(
            fname,
            name,
            func,
            is_3d,
            points,
            tri,
            area_ratio_threshold,
            orig_areas,
            triangle_colors,
            individual_plots_dir,
        )

    print(f"Individual plots created successfully in {individual_plots_dir}!")

    # Initialize figure for combined plot
    fig = plt.figure(figsize=(15, 6))

    alpha = 0.6

    # Plot original domain with PolyCollection
    ax_orig = plt.subplot2grid((2, 6), (0, 0), colspan=2, rowspan=2)
    poly_orig = PolyCollection(
        orig_triangles,
        facecolors=triangle_colors,
        alpha=alpha,
        edgecolors="black",  # Add edges for better visibility
        linewidths=0.4,
    )
    ax_orig.add_collection(poly_orig)
    ax_orig.set(
        title=r"$F_0$ (Original Domain)",
        aspect="equal",
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        xticks=[-1, 0, 1],
        yticks=[-1, 0, 1],
    )
    ax_orig.grid(True, linestyle="--", alpha=0.3)

    # Transformation configurations with position for combined plot
    transforms_combined = [
        (r"$F_1$ (SR - Swiss Roll)", swiss_roll, True, (0, 2)),
        (r"$F_2$ (DE - Dog Ear)", dog_ear, True, (0, 3)),
        (r"$F_3$ (RB - Ribbon)", ribbon, True, (0, 4)),
        (r"$F_4$ (CY - Cylinder)", cylinder, True, (0, 5)),
        (r"$F_5$ (SP - Split)", split, False, (1, 2)),
        (r"$F_6$ (HL - Hole)", hole, False, (1, 3)),
        (r"$F_7$ (PN - Pinch)", pinch, False, (1, 4)),
        (r"$F_8$ (CL - Collapse)", collapse, False, (1, 5)),
    ]

    # Plot transformations
    for idx, (name, func, is_3d, pos) in enumerate(transforms_combined):
        transformed = func(points).cpu().numpy()
        ax = plt.subplot2grid((2, 6), pos, projection="3d" if is_3d else None)

        if is_3d:
            vertices = transformed[tri.simplices]

            # Calculate transformed triangle areas vectorized
            trans_areas = triangle_area(vertices)

            # Calculate area ratios and create mask
            area_ratios = trans_areas / orig_areas
            valid_mask = area_ratios <= area_ratio_threshold

            # Apply mask to filter triangles
            if np.any(valid_mask):
                valid_vertices = vertices[valid_mask]
                valid_colors = triangle_colors[valid_mask]

                poly3d = Poly3DCollection(
                    valid_vertices,
                    alpha=alpha,
                    facecolors=valid_colors,
                    edgecolors="black",  # Add edges for better visibility
                    linewidths=0.1,
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
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("lightgray")
            ax.grid(False)
        else:
            # Use PolyCollection for 2D
            vertices_2d = transformed[tri.simplices][:, :, :2]  # Drop z-coordinate

            # Calculate transformed triangle areas vectorized
            trans_areas = triangle_area(vertices_2d)

            # Calculate area ratios and create mask
            area_ratios = trans_areas / orig_areas
            valid_mask = area_ratios <= area_ratio_threshold

            # Apply mask to filter triangles
            if np.any(valid_mask):
                valid_vertices = vertices_2d[valid_mask]
                valid_colors = triangle_colors[valid_mask]

                poly2d = PolyCollection(
                    valid_vertices,
                    facecolors=valid_colors,
                    alpha=alpha,
                    edgecolors="black",  # Add edges for better visibility
                    linewidths=0.1,
                )
                ax.add_collection(poly2d)

            ax.set(
                aspect="equal", xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[], yticks=[]
            )

        ax.set_title(name)

    # Final adjustments
    fig.suptitle("Homotopy Transformations (t = 1)", fontsize=20, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save outputs
    for fmt in ("pdf", "png"):
        output_path = os.path.join(output_dir, f"{output_name}.{fmt}")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Combined figure created successfully in {output_dir}!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate toy manifold visualizations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures (default: figures)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fig_points_manifolds",
        help="Base filename for output figures (default: fig_points_manifolds)",
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
