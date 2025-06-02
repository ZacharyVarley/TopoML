"""
Visualize Cahn-Hilliard simulation results and metrics with publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.cm import ScalarMappable
import argparse
import glob
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import torch

# Use publication-quality settings for matplotlib
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)

# Define metric name mapping for better labels
METRIC_LABELS = {
    "MLE_original": r"$\hat{d}_{\text{original}}$",
    "MLE_transformed": r"$\hat{d}_{\text{transformed}}$",
    "B0_original": r"$\beta_0^{\text{original}}$",
    "B0_transformed": r"$\beta_0^{\text{transformed}}$",
    "Forward_Lipschitz": r"$\text{Lip}(f)$",
    "Forward_Lipschitz_mean": r"$\text{Mean Lip}(f)$",
    "Inverse_Lipschitz": r"$\text{Lip}(f^{-1})$",
    "Inverse_Lipschitz_mean": r"$\text{Mean Lip}(f^{-1})$",
    "LJD_domain": r"$\text{LJD}_{\text{domain}}$",
    "LJD_codomain": r"$\text{LJD}_{\text{codomain}}$",
    "LJD_combined": r"$\text{LJD}_{\text{combined}}$",
    "MST_error_input": r"$\text{MST}_{\text{input}}$",
    "MST_error_output": r"$\text{MST}_{\text{output}}$",
    "MST_error_combined": r"$\text{MST}_{\text{combined}}$",
    "TopoAE": r"$\text{TopoAE}$",
    "RTD": r"$\text{RTD}$",
}

# Define transformation name mapping to match fig_points_manifolds.py
TRANSFORM_LABELS = {
    "Identity": r"$F_0$",
    "Swiss_Roll": r"$F_1$",
    "Dog_Ear": r"$F_2$",
    "Ribbon": r"$F_3$",
    "Cylinder": r"$F_4$",
    "Split": r"$F_5$",
    "Hole": r"$F_6$",
    "Pinch": r"$F_7$",
    "Collapse": r"$F_8$",
}


def get_latest_metrics_file(output_dir: str) -> str:
    """
    Get the most recent metrics file based on timestamp in the filename.

    Args:
        output_dir: Directory containing metrics output files

    Returns:
        Path to the most recent metrics file
    """
    files = glob.glob(f"{output_dir}/*_metrics_*.npy")
    if not files:
        raise FileNotFoundError(f"No metrics files found in {output_dir}")

    # Sort by timestamp and metrics number (both parts of the filename)
    sorted_files = sorted(
        files, key=lambda f: os.path.basename(f).split(".")[0], reverse=True
    )
    return sorted_files[0]


def load_metrics_and_simulation_data(metrics_file: str) -> Tuple[Dict, Dict]:
    """
    Load both the metrics data and the original simulation data.

    Args:
        metrics_file: Path to the metrics file

    Returns:
        Tuple of (metrics_data, simulation_data)
    """
    # Load metrics data
    metrics_data = np.load(metrics_file, allow_pickle=True).item()

    # Get path to the original simulation data
    source_data_path = metrics_data.get("source_data_path")
    if not source_data_path or not os.path.exists(source_data_path):
        raise FileNotFoundError(f"Source data file not found: {source_data_path}")

    # Load simulation data
    simulation_data = np.load(source_data_path, allow_pickle=True).item()

    return metrics_data, simulation_data


def get_metric_range(
    metrics: Dict,
    transform_names: List[str],
    metric_name: str,
    percentile: float = 99.0,
) -> Tuple[float, float]:
    """
    Compute consistent color range for a given metric across all transformations.

    Args:
        metrics: Dictionary of metrics data
        transform_names: List of transformation names
        metric_name: Name of the metric
        percentile: Percentile to use for upper bound (to avoid outliers)

    Returns:
        Tuple of (min_val, max_val) for the colorbar range
    """
    all_values = []
    per_point_key = f"{metric_name}_per_point"

    for transform_name in transform_names:
        if transform_name in metrics:
            transform_metrics = metrics[transform_name]
            if per_point_key in transform_metrics:
                values = transform_metrics[per_point_key]
                all_values.append(values)

    if not all_values:
        return (0, 1)  # Default range if no values

    # Combine all values and compute range
    all_values = np.concatenate(all_values)
    min_val = np.min(all_values)

    # Use percentile for max value to avoid outlier influence
    max_val = np.percentile(all_values, percentile)

    # Add a small buffer to the range
    range_buffer = 0.05 * (max_val - min_val)
    return (min_val - range_buffer, max_val + range_buffer)


def needs_log_scale(metric_name: str, values: np.ndarray) -> bool:
    """
    Determine if a metric should be displayed on a log scale.

    Args:
        metric_name: Name of the metric
        values: Array of metric values

    Returns:
        Boolean indicating whether to use log scale
    """
    # Define metrics that typically need log scale
    log_scale_metrics = {"Inverse_Lipschitz", "Forward_Lipschitz"}

    # Also check data range - if it spans more than 2 orders of magnitude, use log scale
    if values.max() > 0 and values.min() > 0:
        data_range = values.max() / values.min()
        if data_range > 100 or metric_name in log_scale_metrics:
            return True

    return False


def plot_simulation_grid(
    fig: plt.Figure,
    sim_results: np.ndarray,
    indices: List[int],
    transform_names: List[str],
    grid_size: int,
    save_path: str,
    cmap: str = "viridis",
) -> None:
    """
    Plot a clean grid of simulation results with proper aspect ratio.

    Args:
        fig: Matplotlib figure
        sim_results: Dictionary of simulation results arrays
        indices: List of indices to plot
        transform_names: List of transformation names
        grid_size: Size of each simulation grid
        save_path: Path to save the figure
        cmap: Colormap for the plot
    """
    # Determine layout based on number of transformations and samples
    n_transforms = len(transform_names)
    n_samples = min(
        len(indices), 6
    )  # Limit to 6 samples per transformation for clarity

    if n_samples == 0:
        print("No samples to display")
        return

    # Create figure with proper aspect ratio
    n_cols = min(n_samples, 3)
    n_rows = n_transforms * ((n_samples + n_cols - 1) // n_cols)

    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2.2 / n_cols))

    # Determine global color scale for consistent coloring
    vmin, vmax = np.inf, -np.inf
    for transform_name in transform_names:
        if transform_name in sim_results:
            data = sim_results[transform_name]
            vmin = min(vmin, np.min(data))
            vmax = max(vmax, np.max(data))

    # Create subplots
    for t_idx, transform_name in enumerate(transform_names):
        if transform_name not in sim_results:
            continue

        transform_data = sim_results[transform_name]

        for i in range(min(n_samples, len(indices))):
            idx = indices[i]
            ax_idx = t_idx * n_cols + i + 1
            if ax_idx > n_rows * n_cols:
                break

            ax = fig.add_subplot(n_rows, n_cols, ax_idx)

            # Get the simulation result for this index and reshape
            result = transform_data[idx].reshape(grid_size, grid_size)

            # Plot with proper colormap and common scale
            im = ax.imshow(result, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

            if i == 0:
                # Only add transformation name to first sample in each row
                # Use the transformed name from the mapping
                display_name = TRANSFORM_LABELS.get(transform_name, transform_name)
                ax.set_title(f"{display_name}")

            # Clean up the axes
            ax.set_xticks([])
            ax.set_yticks([])

            # Add sample index as text in the corner
            ax.text(
                0.05,
                0.95,
                f"Sample {idx}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    # Add a single colorbar for all plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Concentration")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved simulation visualization to {save_path}")


def plot_metric_comparison(
    metrics: Dict,
    transform_names: List[str],
    metric_name: str,
    params: np.ndarray,
    output_path: str,
    log_scale: bool = False,
    vmin: float = None,
    vmax: float = None,
) -> None:
    """
    Create a publication-quality comparison of a metric across transformations.

    Args:
        metrics: Dictionary of metrics data
        transform_names: List of transformation names to visualize
        metric_name: Name of the metric to visualize
        params: Original parameter values (for scatter plot coloring)
        output_path: Path to save the figure
        log_scale: Whether to use log scale for colormap
        vmin: Minimum value for colorbar (optional)
        vmax: Maximum value for colorbar (optional)
    """
    n_transforms = len(transform_names)
    n_cols = min(3, n_transforms)
    n_rows = (n_transforms + n_cols - 1) // n_cols

    # Create figure with better aspect ratio
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3))

    per_point_key = f"{metric_name}_per_point"

    # Collect all values to determine global range if not provided
    if vmin is None or vmax is None:
        all_values = []
        for transform_name in transform_names:
            if transform_name in metrics:
                if per_point_key in metrics[transform_name]:
                    values = metrics[transform_name][per_point_key]
                    all_values.append(values)

        if all_values:
            all_values = np.concatenate(all_values)
            if vmin is None:
                # vmin = np.nanmin(all_values)
                vmin = np.nanpercentile(all_values, 1)
            if vmax is None:
                # Use 99th percentile to avoid extreme outliers affecting the scale
                vmax = np.nanpercentile(all_values, 99)

    # Choose colormap and normalization
    cmap = plt.cm.viridis
    norm = (
        LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
        if log_scale
        else Normalize(vmin=vmin, vmax=vmax)
    )

    # Create the plot grid
    grid = GridSpec(n_rows, n_cols, figure=fig)

    # Get proper metric label from mapping
    metric_label = METRIC_LABELS.get(metric_name, metric_name)

    # Plot each transformation
    for i, transform_name in enumerate(transform_names):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(grid[row, col])

        if (
            transform_name not in metrics
            or per_point_key not in metrics[transform_name]
        ):
            ax.text(
                0.5,
                0.5,
                f"No data for {transform_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(TRANSFORM_LABELS.get(transform_name, transform_name))
            continue

        # Get metric values
        values = metrics[transform_name][per_point_key]

        # Create scatter plot
        sc = ax.scatter(
            params[:, 0],  # First parameter
            params[:, 1],  # Second parameter
            c=values,
            cmap=cmap,
            norm=norm,
            s=50,
            alpha=0.8,
            edgecolors="none",
        )

        # Add labels and clean up
        ax.set_title(TRANSFORM_LABELS.get(transform_name, transform_name))

        # Only add x-axis label for bottom row, y-axis label for leftmost column
        if row == n_rows - 1:
            ax.set_xlabel(r"Parameter $c_0$")
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(r"Parameter $\alpha$")
        else:
            ax.set_yticklabels([])

        # Turn off the ticks as they all have the same range [-1, 1]
        ax.set_xticks([])
        ax.set_yticks([])

        # equal aspect ratio
        ax.set_aspect("equal")

        # Add metrics statistics as text
        avg_value = metrics[transform_name].get(metric_name, np.nan)
        ax.text(
            0.05,
            0.95,
            f"Mean: {avg_value:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(metric_label)

    # Add overall title with proper LaTeX formatting
    fig.suptitle(f"{metric_label} across transformations", y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {metric_name} comparison to {output_path}")


def plot_metric_comparison_with_mst(
    metrics: Dict,
    transform_names: List[str],
    metric_name: str,
    params: np.ndarray,
    output_path: str,
    log_scale: bool = False,
    vmin: float = None,
    vmax: float = None,
) -> None:
    """
    Create a publication-quality comparison of a metric across transformations
    with MST edges visualized.

    Args:
        metrics: Dictionary of metrics data
        transform_names: List of transformation names to visualize
        metric_name: Name of the metric to visualize
        params: Original parameter values (for scatter plot coloring)
        output_path: Path to save the figure
        log_scale: Whether to use log scale for colormap
        vmin: Minimum value for colorbar (optional)
        vmax: Maximum value for colorbar (optional)
    """
    n_transforms = len(transform_names)
    n_cols = min(3, n_transforms)
    n_rows = (n_transforms + n_cols - 1) // n_cols

    # Create figure with better aspect ratio
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3))

    per_point_key = f"{metric_name}_per_point"

    # Collect all values to determine global range if not provided
    if vmin is None or vmax is None:
        all_values = []
        for transform_name in transform_names:
            if transform_name in metrics:
                if per_point_key in metrics[transform_name]:
                    values = metrics[transform_name][per_point_key]
                    all_values.append(values)

        if all_values:
            all_values = np.concatenate(all_values)
            if vmin is None:
                # vmin = np.nanmin(all_values)
                vmin = np.nanpercentile(all_values, 1)
            if vmax is None:
                # Use 95th percentile to avoid extreme outliers affecting the scale
                vmax = np.nanpercentile(all_values, 99)

    # Choose colormap and normalization
    cmap = plt.cm.viridis
    norm = (
        LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
        if log_scale
        else Normalize(vmin=vmin, vmax=vmax)
    )

    # Create the plot grid
    grid = GridSpec(n_rows, n_cols, figure=fig)

    # Get proper metric label from mapping
    metric_label = METRIC_LABELS.get(metric_name, metric_name)

    # Determine if we're working with MST error metrics
    is_mst_metric = "MST_error" in metric_name and "combined" not in metric_name

    # Import necessary modules for line collection
    from matplotlib.collections import LineCollection

    # Plot each transformation
    for i, transform_name in enumerate(transform_names):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(grid[row, col])

        if (
            transform_name not in metrics
            or per_point_key not in metrics[transform_name]
        ):
            ax.text(
                0.5,
                0.5,
                f"No data for {transform_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(TRANSFORM_LABELS.get(transform_name, transform_name))
            continue

        # Get metric values
        values = metrics[transform_name][per_point_key]

        # Create scatter plot
        sc = ax.scatter(
            params[:, 0],  # First parameter
            params[:, 1],  # Second parameter
            c="black",  # Use black for points
            zorder=10,  # Make points appear above edges
            s=5,
        )

        # Draw MST edges if we have the data and it's an MST metric
        if is_mst_metric:
            # Choose appropriate MST data based on the metric type
            if "MST_error_in" in metric_name:
                mst_key = "MST_input_vertex_pairs"
                mst_error_key = "MST_error_in_per_point"
            elif "MST_error_ot" in metric_name:
                mst_key = "MST_output_vertex_pairs"
                mst_error_key = "MST_error_ot_per_point"

            # Check if we have the required MST data
            if (
                mst_key in metrics[transform_name]
                and mst_error_key in metrics[transform_name]
            ):
                # Get MST vertex pairs and points
                mst_vertex_pairs = metrics[transform_name][mst_key]
                edge_errors = metrics[transform_name][mst_error_key]

                # We need to map from high dimensional space to 2D space for visualization
                # We'll use the parameter space for this
                edge_segments = []
                edge_colors = []

                for edge_idx, (i, j) in enumerate(mst_vertex_pairs):
                    edge_segments.append(
                        [
                            [params[i, 0], params[i, 1]],
                            [params[j, 0], params[j, 1]],
                        ]
                    )
                    edge_colors.append(edge_errors[edge_idx])

                # Create a line collection for efficient edge plotting
                line_segments = LineCollection(
                    edge_segments,
                    cmap=cmap,
                    norm=norm,
                    linewidth=2,
                    alpha=0.8,
                    zorder=5,  # Draw under points
                )
                line_segments.set_array(np.array(edge_colors))
                ax.add_collection(line_segments)

        # Add labels and clean up
        ax.set_title(TRANSFORM_LABELS.get(transform_name, transform_name))

        # Only add x-axis label for bottom row, y-axis label for leftmost column
        if row == n_rows - 1:
            ax.set_xlabel(r"Parameter $c_0$")
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(r"Parameter $\alpha$")
        else:
            ax.set_yticklabels([])

        # Turn off the ticks as they all have the same range [-1, 1]
        ax.set_xticks([])
        ax.set_yticks([])

        # equal aspect ratio
        ax.set_aspect("equal")

        # Add metrics statistics as text
        avg_value = metrics[transform_name].get(metric_name, np.nan)
        ax.text(
            0.05,
            0.95,
            f"Mean: {avg_value:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            zorder=15,  # Ensure text is above everything
        )

    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(metric_label)

    # Add overall title with proper LaTeX formatting
    title = f"{metric_label} across transformations"
    if is_mst_metric:
        title += " (with MST edges)"
    fig.suptitle(title, y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {metric_name} comparison with MST visualization to {output_path}")


def plot_average_metrics(
    metrics: Dict,
    transform_names: List[str],
    metric_names: List[str],
    output_path: str,
) -> None:
    """
    Create a comparison bar chart of average metrics across transformations.

    Args:
        metrics: Dictionary of metrics data
        transform_names: List of transformation names
        metric_names: List of metrics to visualize
        output_path: Path to save the figure
    """
    n_metrics = len(metric_names)

    # Create figure with proper aspect ratio
    fig, axes = plt.subplots(n_metrics, 1, figsize=(8, n_metrics * 2.5), sharex=True)

    # If only one metric, wrap in a list
    if n_metrics == 1:
        axes = [axes]

    # Define colors for bars
    colors = plt.cm.tab10(np.linspace(0, 1, len(transform_names)))

    # Get proper transformation labels
    display_transform_names = [
        TRANSFORM_LABELS.get(name, name) for name in transform_names
    ]

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]

        # Extract values
        values = []
        for transform_name in transform_names:
            if transform_name in metrics:
                value = metrics[transform_name].get(metric_name, np.nan)
                values.append(value)
            else:
                values.append(np.nan)

        # Create bar chart
        bars = ax.bar(display_transform_names, values, color=colors, alpha=0.8)

        # Add values on top of bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height * 1.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                    fontsize=8,
                )

        # Add labels with proper LaTeX formatting
        metric_label = METRIC_LABELS.get(metric_name, metric_name)
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by Transformation")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Check if we need log scale
        if needs_log_scale(metric_name, np.array(values)):
            ax.set_yscale("log")

    # Set x-axis labels for the bottom plot
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Transformation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved average metrics comparison to {output_path}")


def plot_metric_histograms(
    metrics: Dict,
    transform_names: List[str],
    metric_name: str,
    output_path: str,
    log_scale: bool = False,
) -> None:
    """
    Create histograms of metric distributions across transformations.

    Args:
        metrics: Dictionary of metrics data
        transform_names: List of transformation names
        metric_name: Name of the metric to visualize
        output_path: Path to save the figure
        log_scale: Whether to use log scale for x-axis
    """
    n_transforms = len(transform_names)
    n_cols = min(3, n_transforms)
    n_rows = (n_transforms + n_cols - 1) // n_cols

    # Create figure with proper aspect ratio
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 2.5))

    per_point_key = f"{metric_name}_per_point"

    # Get proper metric label from mapping
    metric_label = METRIC_LABELS.get(metric_name, metric_name)

    # Compute global min/max for consistent bins
    all_values = []
    for transform_name in transform_names:
        if transform_name in metrics and per_point_key in metrics[transform_name]:
            all_values.append(metrics[transform_name][per_point_key])

    if not all_values:
        print(f"No histogram data for {metric_name}")
        return

    all_values = np.concatenate(all_values)
    global_min = np.min(all_values)
    global_max = np.percentile(
        all_values, 99
    )  # 99th percentile to avoid extreme outliers

    # Create bins
    if log_scale and global_min > 0:
        bins = np.logspace(np.log10(max(global_min, 1e-5)), np.log10(global_max), 30)
    else:
        bins = np.linspace(global_min, global_max, 30)

    # Plot each transformation
    for i, transform_name in enumerate(transform_names):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        if (
            transform_name not in metrics
            or per_point_key not in metrics[transform_name]
        ):
            ax.text(0.5, 0.5, f"No data", ha="center", va="center")
            ax.set_title(TRANSFORM_LABELS.get(transform_name, transform_name))
            continue

        values = metrics[transform_name][per_point_key]

        # Create histogram
        ax.hist(
            values,
            bins=bins,
            alpha=0.7,
            color="royalblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Add mean line
        mean_value = metrics[transform_name].get(metric_name, np.nan)
        if not np.isnan(mean_value):
            ax.axvline(
                mean_value,
                color="crimson",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {mean_value:.3f}",
            )
            ax.legend(fontsize=8)

        # Set scale if needed
        if log_scale and global_min > 0:
            ax.set_xscale("log")

        # Clean up and labels
        ax.set_title(TRANSFORM_LABELS.get(transform_name, transform_name))

        # Only add x-axis label for bottom row, y-axis label for leftmost column
        if row == n_rows - 1:
            ax.set_xlabel(metric_label)
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_yticklabels([])

        # Make the plot cleaner by removing ticks
        ax.tick_params(axis="both", which="both", length=0)

    # Overall title
    fig.suptitle(f"{metric_label} Distribution", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {metric_name} histograms to {output_path}")


def create_publication_visualizations(
    metrics_data: Dict,
    simulation_data: Dict,
    output_dir: str,
    transform_names: List[str] = None,
    sample_indices: List[int] = None,
    metric_names: List[str] = None,
    save_format: str = "png",
) -> None:
    """
    Create publication-quality visualizations of metrics and simulation results.

    Args:
        metrics_data: Dictionary containing metrics data
        simulation_data: Dictionary containing simulation data
        output_dir: Directory to save output figures
        transform_names: List of transformation names to visualize (default: all)
        sample_indices: List of sample indices to visualize (default: first 6)
        metric_names: List of metrics to visualize (default: all per-point metrics)
        save_format: Format for saved figures (png, pdf, etc.)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get metrics data
    metrics = metrics_data["metrics"]

    # Get original configuration
    config = simulation_data["config"]
    grid_size = config.get("grid_size", 128)

    # Get simulation results
    sim_results = simulation_data["simulation_results"]

    # Get original parameters
    original_params = simulation_data["original_params"]
    transformed_params_norm = simulation_data["transformed_params_norm"]

    # Determine transformations to visualize
    if transform_names is None or len(transform_names) == 0:
        transform_names = list(metrics.keys())

    # Determine sample indices to visualize
    if sample_indices is None:
        # Use first samples by default, limited to 6
        n_samples = min(6, len(original_params))
        sample_indices = list(range(n_samples))

    # Determine metrics to visualize
    if metric_names is None:
        # Find all per-point metrics by checking for "_per_point" suffix
        all_metrics = set()
        for transform_name, transform_metrics in metrics.items():
            for metric_name in transform_metrics:
                if metric_name.endswith("_per_point"):
                    base_name = metric_name.replace("_per_point", "")
                    all_metrics.add(base_name)
        metric_names = sorted(list(all_metrics))

    # 1. Visualize simulation results (limited to a reasonable number of samples)
    fig = plt.figure()  # Temporary figure, will be closed in the function
    plot_simulation_grid(
        fig,
        sim_results,
        sample_indices[:6],  # Limit to 6 samples for clarity
        transform_names,
        grid_size,
        os.path.join(output_dir, f"CH_manifolds_sim_results.{save_format}"),
        cmap="viridis",
    )

    # 2. Create visualizations for each metric
    for metric_name in metric_names:
        # Check if we need log scale
        per_point_key = f"{metric_name}_per_point"
        test_values = []

        for transform_name in transform_names:
            if transform_name in metrics and per_point_key in metrics[transform_name]:
                test_values.append(metrics[transform_name][per_point_key])

        if not test_values:
            print(f"No data found for metric: {metric_name}")
            continue

        test_values = np.concatenate(test_values)
        use_log_scale = needs_log_scale(metric_name, test_values)

        # Calculate a consistent color range for all plots
        vmin, vmax = get_metric_range(metrics, transform_names, metric_name)

        # Parameter space comparisons colored by metric
        # For MST error metrics, use the special function that draws edges
        if "MST_error" in metric_name:
            plot_metric_comparison_with_mst(
                metrics,
                transform_names,
                metric_name,
                transformed_params_norm["Identity"],  # Original parameters
                os.path.join(
                    output_dir, f"CH_manifolds_{metric_name}_param_space.{save_format}"
                ),
                log_scale=use_log_scale,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            plot_metric_comparison(
                metrics,
                transform_names,
                metric_name,
                transformed_params_norm["Identity"],  # Original parameters
                os.path.join(
                    output_dir, f"CH_manifolds_{metric_name}_param_space.{save_format}"
                ),
                log_scale=use_log_scale,
                vmin=vmin,
                vmax=vmax,
            )

        # Histograms of metric distribution
        plot_metric_histograms(
            metrics,
            transform_names,
            metric_name,
            os.path.join(
                output_dir, f"CH_manifolds_{metric_name}_histograms.{save_format}"
            ),
            log_scale=use_log_scale,
        )

    # 3. Create bar chart comparing average metric values across transformations
    # Remove per-point metrics for the bar chart
    avg_metrics = [m for m in metric_names if not m.endswith("_per_point")]
    plot_average_metrics(
        metrics,
        transform_names,
        avg_metrics,
        os.path.join(output_dir, f"CH_manifolds_avg_metrics.{save_format}"),
    )

    print("All visualizations complete!")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Visualize Cahn-Hilliard simulation results and metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--metrics_file",
        type=str,
        default=None,
        help="Path to the metrics file. If not provided, uses the most recent file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory for saving figures",
    )

    parser.add_argument(
        "--transformations",
        type=str,
        nargs="+",
        default=None,
        help="List of transformations to visualize. Default: all transformations",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metrics to visualize. Default: all metrics with per-point values",
    )

    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="List of sample indices to visualize. Default: first 6 samples",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg", "jpg"],
        default="png",
        help="Format for saved figures",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to create publication-quality visualizations.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Determine the metrics file
    metrics_file = args.metrics_file
    if metrics_file is None:
        try:
            metrics_file = get_latest_metrics_file("ch_manifold_results")
            print(f"Using most recent metrics file: {metrics_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify a metrics file with --metrics_file")
            return

    # Load data
    print(f"Loading data from: {metrics_file}")
    try:
        metrics_data, simulation_data = load_metrics_and_simulation_data(metrics_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Create visualizations
    create_publication_visualizations(
        metrics_data=metrics_data,
        simulation_data=simulation_data,
        output_dir=args.output_dir,
        transform_names=args.transformations,
        sample_indices=args.samples,
        metric_names=args.metrics,
        save_format=args.format,
    )

    print("Visualization completed successfully!")


if __name__ == "__main__":
    main()
