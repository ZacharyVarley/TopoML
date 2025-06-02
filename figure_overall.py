"""
Generate an overall figure showing the parameter transformation and Cahn-Hilliard simulation
process for the Swiss Roll transformation with improved layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import os
import argparse
import torch
import glob
from typing import Dict, List

# Set publication-quality parameters for matplotlib
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)


def get_latest_simulation_file(output_dir: str) -> str:
    """
    Get the most recent simulation file based on timestamp in the filename.

    Args:
        output_dir: Directory containing simulation output files

    Returns:
        Path to the most recent simulation file
    """
    files = glob.glob(f"{output_dir}/*.npy")
    if not files:
        raise FileNotFoundError(f"No simulation files found in {output_dir}")

    # Sort by timestamp (filenames are in format "path/YYYYMMDDHHMMSS.npy")
    sorted_files = sorted(
        files, key=lambda f: os.path.basename(f).split(".")[0], reverse=True
    )
    # remove files with "_" in the name
    sorted_files = [f for f in sorted_files if "_" not in os.path.basename(f)]
    return sorted_files[0]


def create_clean_figure(
    simulation_data: Dict,
    output_dir: str = "figures",
    output_name: str = "figure_overall_clean",
) -> None:
    """
    Create a clean publication-quality figure showing the parameter transformation
    and simulation process with optimal layout and no connecting elements.

    Args:
        simulation_data: Dictionary containing simulation data
        selected_indices: List of indices for points to highlight
        output_dir: Directory to save output figure
        output_name: Base name for output figure file
    """
    # Extract data from simulation dictionary
    original_params = simulation_data["original_params"]
    transformed_params_norm = simulation_data["transformed_params_norm"]
    simulation_results = simulation_data["simulation_results"]
    config = simulation_data["config"]
    grid_size = config["grid_size"]

    # Create device for torch operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create figure with better proportions
    fig = plt.figure(figsize=(18, 6))

    # Define a simple 2x6 grid
    # We'll position the plots using subplot2grid for precise control

    # Create a top title for the entire figure
    title_ax = fig.add_axes([0, 0.95, 1, 0.05])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.6,
        "Cahn-Hilliard Simulations Under Swiss Roll Parameter Transformation",
        fontsize=16,
        fontweight="bold",
        ha="center",
    )
    # title_ax.text(
    #     0.5,
    #     0.2,
    #     "Mapping between parameter space, transformed space, and resulting microstructures",
    #     fontsize=12,
    #     style="italic",
    #     ha="center",
    # )

    # Plot 1: Original uniform sampling (Parameter Space) - spans 2 rows and 2 columns
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2, rowspan=2)
    original_points = transformed_params_norm["Identity"][:, :2]

    # Plot all points with better visibility
    scatter_all = ax1.scatter(
        original_points[:, 0],
        original_points[:, 1],
        s=5,
        alpha=0.4,
        color="royalblue",
        edgecolor=None,
    )

    # # Highlight the selected points with bigger markers
    # scatter_selected = ax1.scatter(
    #     original_points[selected_indices, 0],
    #     original_points[selected_indices, 1],
    #     s=80,
    #     color="red",
    #     edgecolor="black",
    #     linewidth=1.5,
    #     zorder=10,
    # )

    selected_indices = [0, 1, 2, 3]  # Example indices for selected points

    # Add labels for selected points
    for i, idx in enumerate(selected_indices):
        ax1.annotate(
            f"{i+1}",
            (original_points[idx, 0], original_points[idx, 1]),
            xytext=(0, 0),
            textcoords="offset points",
            color="white",
            fontweight="bold",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle", facecolor="black", alpha=0.9, pad=0.2),
        )

    ax1.set_title("Parameter Space", fontsize=14, pad=15)
    ax1.set_xlabel(r"x", fontsize=12)
    ax1.set_ylabel(r"y", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect("equal")
    ax1.set_facecolor("#f8f8f8")

    # Plot 2: Swiss Roll transformation in 3D - spans 2 rows and 2 columns
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2, rowspan=2, projection="3d")

    # Get original points as tensor
    original_points_tensor = torch.tensor(
        original_points, dtype=torch.float32, device=device
    )

    # Get transformed points
    from src_toy_manifolds import swiss_roll

    swiss_roll_points_tensor = swiss_roll(original_points_tensor)
    swiss_roll_points = swiss_roll_points_tensor.cpu().numpy()

    # Normalize the transformed points
    sr_points_nrm_x = (swiss_roll_points[:, 0] * 0.5 + 0.5) * 0.4 + 0.3
    sr_points_nrm_y = (swiss_roll_points[:, 1] * 0.5 + 0.5) * 0.6 + 0.7
    sr_points_nrm_z = (swiss_roll_points[:, 2] * 0.5 + 0.5) * 0.6 + 0.7

    # Plot all transformed points with better visibility
    ax2.scatter(
        sr_points_nrm_x,
        sr_points_nrm_y,
        sr_points_nrm_z,
        s=5,
        alpha=0.4,
        color="royalblue",
        edgecolor=None,
    )

    # Highlight selected points with larger markers
    ax2.scatter(
        sr_points_nrm_x[selected_indices],
        sr_points_nrm_y[selected_indices],
        sr_points_nrm_z[selected_indices],
        s=80,
        color="red",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
    )

    # Add labels for selected points
    for i, idx in enumerate(selected_indices):
        ax2.text(
            # swiss_roll_points[idx, 0],  # c_init from 0.3 to 0.7
            # swiss_roll_points[idx, 1],  # alpha from 0.7 to 1.3
            # swiss_roll_points[idx, 2],  # beta from 0.7 to 1.3
            sr_points_nrm_x[idx],
            sr_points_nrm_y[idx],
            sr_points_nrm_z[idx],
            f"{i+1}",
            color="white",
            fontweight="bold",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle", facecolor="black", alpha=0.9, pad=0.2),
        )

    ax2.set_title("Swiss Roll Transformation", fontsize=14, pad=15)
    ax2.set_xlabel(r"$c_{\mathrm{init}}$", fontsize=12)
    ax2.set_ylabel(r"$\alpha$", fontsize=12)
    ax2.set_zlabel(r"$\beta$", fontsize=12)
    ax2.view_init(elev=20, azim=60)
    # ax2.set_xlim(-1.1, 1.1)
    # ax2.set_ylim(-1.1, 1.1)
    # ax2.set_zlim(-1.1, 1.1)
    ax2.grid(False)
    for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("lightgray")
    ax2.set_facecolor("#f8f8f8")

    # Set up consistent colormap for all simulation visualizations
    vmin = 0.0
    vmax = 1.0
    cmap = "viridis"

    # Get simulation results for Swiss Roll
    sim_results_swiss_roll = simulation_results["Swiss_Roll"]

    # Create a title for the microstructures section
    sim_title_ax = fig.add_axes([0.7, 0.9, 0.3, 0.02])
    sim_title_ax.axis("off")
    sim_title_ax.text(0.25, 2.0, "Resulting Microstructures", fontsize=14, ha="center")

    # Create the four microstructure plots one by one
    # Define positions for the 2x2 grid in the right section
    sim_pos = [
        (0, 4),  # Top-left
        (0, 5),  # Top-right
        (1, 4),  # Bottom-left
        (1, 5),  # Bottom-right
    ]

    # Plot each microstructure
    for i, (pos, idx) in enumerate(zip(sim_pos, selected_indices)):
        row, col = pos
        ax = plt.subplot2grid((2, 6), (row, col), colspan=1, rowspan=1)
        result = sim_results_swiss_roll[idx].reshape(grid_size, grid_size)

        im = ax.imshow(
            result,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="none",
        )

        # Get the original parameter values
        c_init = sr_points_nrm_x[idx]
        alpha = sr_points_nrm_y[idx]
        beta = sr_points_nrm_z[idx]

        # Add point number in top-left corner
        ax.text(
            0.05,
            0.95,
            f"Point {i+1}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.3),
        )

        # Add parameter information at bottom
        param_text = f"({alpha:.2f}, {beta:.2f}, {c_init:.2f})"
        ax.text(
            0.5,
            0.02,
            param_text,
            transform=ax.transAxes,
            fontsize=12,
            va="bottom",
            ha="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.3),
        )

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a border to the simulation images
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.75)

    # Add colorbar for the simulation results
    cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.6])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("Phase Concentration", fontsize=12)

    # Adjust overall spacing
    plt.subplots_adjust(
        wspace=0.1, hspace=0.1, left=0.05, right=0.9, top=0.90, bottom=0.08
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save figure in both PDF and PNG formats
    for ext in ["pdf", "png"]:
        output_path = os.path.join(output_dir, f"{output_name}.{ext}")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    print(
        f"Figure saved to {output_dir}/{output_name}.pdf and {output_dir}/{output_name}.png"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a clean figure showing the parameter transformation and Cahn-Hilliard simulation process",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to the simulation results file. If not provided, uses the most recent file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory for saving output figure",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="figure_overall_clean",
        help="Base name for output figure file",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to generate the clean overall figure."""
    # Parse command-line arguments
    args = parse_arguments()

    # Determine the input file
    input_file = args.input_file
    if input_file is None:
        try:
            input_file = get_latest_simulation_file("ch_manifold_results")
            print(f"Using most recent simulation file: {input_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify an input file with --input_file")
            return

    # Load the simulation data
    print(f"Loading data from: {input_file}")
    simulation_data = np.load(input_file, allow_pickle=True).item()

    # Create the clean figure
    create_clean_figure(
        simulation_data=simulation_data,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )

    print("Done!")


if __name__ == "__main__":
    main()
