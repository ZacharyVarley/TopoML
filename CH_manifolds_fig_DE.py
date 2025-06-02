"""
Visualize Signal-to-Noise Ratio (SNR) for Lipschitz constants from bootstrapped samples,
specifically for the Dog Ear transformation from Cahn-Hilliard simulations.
Creates a figure with imshow plots of SNR values for both forward and inverse Lipschitz constants
with a shared colorbar for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import argparse
import glob
import os
from typing import Dict, Tuple, List, Any, Optional

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
    "Forward_Lipschitz_SNR": r"$\text{SNR}_{\text{Lip}(f)}$",
    "Inverse_Lipschitz_SNR": r"$\text{SNR}_{\text{Lip}(f^{-1})}$",
}

# Define transformation name mapping
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


def load_metrics_data(metrics_file: str) -> Dict:
    """
    Load metrics data from file.

    Args:
        metrics_file: Path to the metrics file

    Returns:
        Dictionary containing metrics data
    """
    # Load metrics data
    metrics_data = np.load(metrics_file, allow_pickle=True).item()
    return metrics_data


def load_simulation_data(source_data_path: str) -> Dict:
    """
    Load simulation data from file.

    Args:
        source_data_path: Path to the simulation data file

    Returns:
        Dictionary containing simulation data
    """
    # Load simulation data
    simulation_data = np.load(source_data_path, allow_pickle=True).item()
    return simulation_data


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


def plot_lipschitz_variation(
    metrics: Dict[str, Any],
    transform_names: List[str],
    metric_name: str,
    original_params: np.ndarray,
    transformed_params_norm: Dict[str, np.ndarray],
    output_path_base: str,
) -> None:
    """
    Create a publication-quality visualization of Lipschitz constant variation
    for a specific metric (Forward or Inverse Lipschitz).

    Args:
        metrics: Dictionary of metrics data
        transform_names: List of transformation names to visualize (in order)
        metric_name: Name of the metric to visualize (Forward_Lipschitz or Inverse_Lipschitz)
        original_params: Original parameter values in parameter space
        transformed_params_norm: Dictionary of normalized transformed parameters (in [-1,1]² space)
        output_path_base: Base path for saving figures (without extension)
    """
    # Define grid layout: 3x3 for the 9 transformations
    n_rows, n_cols = 3, 3

    # Create figure with appropriate size for publication
    fig = plt.figure(figsize=(n_cols * 3.0, n_rows * 2.8))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.1)

    # Check if metric exists in any transformation
    per_point_key = f"{metric_name}_per_point"
    all_values = []

    # Collect all values to determine global color scale
    for transform_name in transform_names:
        if transform_name in metrics and per_point_key in metrics[transform_name]:
            values = metrics[transform_name][per_point_key]
            all_values.append(values)

    if not all_values:
        print(f"No {metric_name} data found for any transformation")
        return

    # Concatenate all values to determine global color scale
    all_values = np.concatenate(all_values)

    # Determine more appropriate range using percentiles to avoid outlier influence
    # These will work better for visualization than full range
    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)

    # Determine if we need log scale
    use_log_scale = needs_log_scale(metric_name, all_values)

    # Calculate a more useful colormap range for log scale
    if use_log_scale:
        # Round to nice powers of 10 for better colorbar ticks
        log_vmin = np.floor(np.log10(vmin))
        log_vmax = np.ceil(np.log10(vmax))
        vmin = 10**log_vmin
        vmax = 10**log_vmax

    # Get proper metric label
    metric_label = METRIC_LABELS.get(metric_name, metric_name)

    # Setup colormap and normalization globally
    # cmap = plt.cm.viridis
    cmap = plt.cm.inferno

    # Use log norm if appropriate
    norm = (
        LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
        if use_log_scale
        else Normalize(vmin=vmin, vmax=vmax)
    )

    # Get normalized uniform samples in [-1,1]² space
    uniform_samples_norm = transformed_params_norm["Identity"][:, :2]

    # Plot each transformation in row-by-row order
    for i, transform_name in enumerate(transform_names):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])

        # Check if we have data for this transformation
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

        # Get metric values for this transformation
        values = metrics[transform_name][per_point_key]

        # Create scatter plot using the original uniform points in [-1,1]² space
        scatter = ax.scatter(
            uniform_samples_norm[:, 0],  # x-coordinate in [-1,1]
            uniform_samples_norm[:, 1],  # y-coordinate in [-1,1]
            c=values,  # Color by metric value
            cmap=cmap,
            norm=norm,
            s=10,  # Slightly smaller point size
            alpha=0.8,
            edgecolors="none",
        )

        # Add transformation name as title
        transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)
        ax.set_title(transform_label, pad=4)

        # Set axis limits to show the full [-1,1]² range with a small margin
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])

        # Only show x-axis ticks and labels for bottom row
        if row_idx == n_rows - 1:
            ax.set_xlabel(r"$c_0$")
            ax.set_xticks([-1, 0, 1])
        else:
            ax.set_xticklabels([])

        # Only show y-axis ticks and labels for first column
        if col_idx == 0:
            ax.set_ylabel(r"$\alpha$")
            ax.set_yticks([-1, 0, 1])
        else:
            ax.set_yticklabels([])

        # Equal aspect ratio for square plot
        ax.set_aspect("equal")

        # Add grid for reference
        ax.grid(alpha=0.2, linestyle="--")

        # Add max value as text in corner
        max_value = np.max(values)
        stats_text = f"Max: {max_value:.2f}"

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # Add a single colorbar for all plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(metric_label)

    # Add more ticks for log scale colorbar
    if use_log_scale and (log_vmax - log_vmin) <= 5:
        cbar.set_ticks([10**i for i in range(int(log_vmin), int(log_vmax) + 1)])

    # Add overall title
    fig.suptitle(f"{metric_label} variation across transformations", y=0.98)

    # Tight layout, with room for the colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Save as both PDF and PNG
    plt.savefig(f"{output_path_base}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved {metric_name} variation figure to {output_path_base}.pdf and {output_path_base}.png"
    )


def plot_side_by_side_lipschitz(
    metrics: Dict[str, Any],
    transform_names: List[str],
    original_params: np.ndarray,
    transformed_params_norm: Dict[str, np.ndarray],
    output_path_base: str,
) -> None:
    """
    Create a publication-quality side-by-side visualization of both Forward and Inverse
    Lipschitz constant variation across transformations.

    Args:
        metrics: Dictionary of metrics data
        transform_names: List of transformation names to visualize (in order)
        original_params: Original parameter values in parameter space
        transformed_params_norm: Dictionary of normalized transformed parameters (in [-1,1]² space)
        output_path_base: Base path for saving figures (without extension)
    """
    # Define the metrics to plot side by side
    metric_names = ["Forward_Lipschitz", "Inverse_Lipschitz"]

    # Define grid layout: 3x6 (3 rows, 2 columns of 3 columns each)
    n_rows, n_cols = 3, 6

    # Create figure with appropriate size for publication
    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.8))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.1, hspace=0.1)

    # Map transformation names to abbreviated labels
    transform_abbrev = {
        "Identity": "IDENT",
        "Swiss_Roll": "SWR",
        "Dog_Ear": "DOGE",
        "Ribbon": "RBN",
        "Cylinder": "CYL",
        "Split": "SPLT",
        "Hole": "HOLE",
        "Pinch": "PNCH",
        "Collapse": "CLPS",
    }

    # For each metric (Forward/Inverse Lipschitz)
    for metric_idx, metric_name in enumerate(metric_names):
        # Check if metric exists in any transformation
        per_point_key = f"{metric_name}_per_point"
        all_values = []

        # Collect all values to determine global color scale
        for transform_name in transform_names:
            if transform_name in metrics and per_point_key in metrics[transform_name]:
                values = metrics[transform_name][per_point_key]
                all_values.append(values)

        if not all_values:
            print(f"No {metric_name} data found for any transformation")
            continue

        # Concatenate all values to determine global color scale
        all_values = np.concatenate(all_values)

        # Determine more appropriate range using percentiles to avoid outlier influence
        vmin = np.percentile(all_values, 5)
        vmax = np.percentile(all_values, 95)

        # Determine if we need log scale
        use_log_scale = needs_log_scale(metric_name, all_values)

        # Calculate a more useful colormap range for log scale
        if use_log_scale:
            # Round to nice powers of 10 for better colorbar ticks
            log_vmin = np.floor(np.log10(vmin))
            log_vmax = np.ceil(np.log10(vmax))
            vmin = 10**log_vmin
            vmax = 10**log_vmax

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Setup colormap and normalization globally
        cmap = plt.cm.inferno

        # Use log norm if appropriate
        norm = (
            LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
            if use_log_scale
            else Normalize(vmin=vmin, vmax=vmax)
        )

        # Get normalized uniform samples in [-1,1]² space
        uniform_samples_norm = transformed_params_norm["Identity"][:, :2]

        # Calculate grid offset for this metric (0 for Forward, 3 for Inverse)
        col_offset = metric_idx * 3

        # Add metric type as super title for this group
        ax_super = plt.subplot(gs[0, col_offset : col_offset + 3])
        ax_super.set_title(f"{metric_label}", pad=10)
        ax_super.axis("off")  # Hide this axes, it's just for the title

        # Plot each transformation in row-by-row order
        for i, transform_name in enumerate(transform_names):
            row_idx = i // 3
            col_idx = (i % 3) + col_offset

            # Create subplot in the proper position
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Check if we have data for this transformation
            if (
                transform_name not in metrics
                or per_point_key not in metrics[transform_name]
            ):
                ax.text(
                    0.5,
                    0.5,
                    f"No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)
                ax.set_title(transform_label, pad=4)
                continue

            # Get metric values for this transformation
            values = metrics[transform_name][per_point_key]

            # Create scatter plot using the original uniform points in [-1,1]² space
            scatter = ax.scatter(
                uniform_samples_norm[:, 0],  # x-coordinate in [-1,1]
                uniform_samples_norm[:, 1],  # y-coordinate in [-1,1]
                c=values,  # Color by metric value
                cmap=cmap,
                norm=norm,
                s=10,  # Smaller point size for combined plot
                alpha=0.8,
                edgecolors="none",
            )

            # Get the transformation name and abbreviation
            transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)
            abbrev = transform_abbrev.get(transform_name, "")

            # Add title with abbreviation
            ax.set_title(f"{transform_label} ({abbrev})", pad=4)

            # Set axis limits to show the full [-1,1]² range with a small margin
            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.05])

            # Remove all ticks by default
            ax.set_xticks([])
            ax.set_yticks([])

            # Only show x-axis ticks and labels for bottom row
            if row_idx == 2:  # Bottom row
                ax.set_xlabel(r"$c_0$")
                ax.set_xticks([-1, 0, 1])

            # Only show y-axis ticks and labels for first column of each section
            if col_idx == col_offset:  # First column of each section
                ax.set_ylabel(r"$\alpha$")
                ax.set_yticks([-1, 0, 1])

            # Equal aspect ratio for square plot
            ax.set_aspect("equal")

            # Add grid for reference
            ax.grid(alpha=0.2, linestyle="--")

            # Add max value as text in corner
            max_value = np.max(values)
            stats_text = f"Max: {max_value:.2f}"

            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=7,  # Smaller font for combined plot
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

        # Add colorbar for each metric section
        cbar_x_pos = 0.46 if metric_idx == 0 else 0.91
        cbar_ax = fig.add_axes([cbar_x_pos, 0.15, 0.01, 0.7])  # Thinner colorbar
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cbar.set_label(metric_label)

        # Add more ticks for log scale colorbar
        if use_log_scale and (log_vmax - log_vmin) <= 5:
            cbar.set_ticks([10**i for i in range(int(log_vmin), int(log_vmax) + 1)])

    # Add overall title
    fig.suptitle(f"Lipschitz constant variation across transformations", y=0.98)

    # # Tight layout with room for title and colorbars
    # plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Save as both PDF and PNG
    plt.savefig(f"{output_path_base}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved side-by-side Lipschitz variation figure to {output_path_base}.pdf and {output_path_base}.png"
    )


def plot_dog_ear_scatter(
    metrics: Dict[str, Any],
    transform_name: str,
    metric_names: List[str],
    transformed_params_norm: Dict[str, np.ndarray],
    output_path_base: str,
) -> None:
    """
    Create a basic scatter plot visualization of Dog Ear transformation for both
    Forward and Inverse Lipschitz constants side by side.

    Args:
        metrics: Dictionary of metrics data
        transform_name: Name of the transformation (should be "Dog_Ear")
        metric_names: List of metric names to visualize
        transformed_params_norm: Dictionary of normalized transformed parameters (in [-1,1]² space)
        output_path_base: Base path for saving figures (without extension)
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Get normalized uniform samples in [-1,1]² space
    uniform_samples_norm = transformed_params_norm["Identity"][:, :2]

    # Setup common parameters
    transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)

    # For each metric (Forward/Inverse Lipschitz)
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]

        # Check if metric exists for this transformation
        per_point_key = f"{metric_name}_per_point"

        # Skip if no data
        if (
            transform_name not in metrics
            or per_point_key not in metrics[transform_name]
        ):
            ax.text(
                0.5,
                0.5,
                f"No {metric_name} data for {transform_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Get metric values
        values = metrics[transform_name][per_point_key]

        # Determine color scale parameters
        vmin = np.percentile(values, 5)
        vmax = np.percentile(values, 95)

        # Determine if we need log scale
        use_log_scale = needs_log_scale(metric_name, values)

        # Calculate a more useful colormap range for log scale
        if use_log_scale:
            log_vmin = np.floor(np.log10(vmin))
            log_vmax = np.ceil(np.log10(vmax))
            vmin = 10**log_vmin
            vmax = 10**log_vmax

        # Setup colormap and normalization
        cmap = plt.cm.inferno

        # Use log norm if appropriate
        norm = (
            LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
            if use_log_scale
            else Normalize(vmin=vmin, vmax=vmax)
        )

        # Create scatter plot
        scatter = ax.scatter(
            uniform_samples_norm[:, 0],  # x-coordinate in [-1,1]
            uniform_samples_norm[:, 1],  # y-coordinate in [-1,1]
            c=values,  # Color by metric value
            cmap=cmap,
            norm=norm,
            s=20,  # Point size
            alpha=0.8,
            edgecolors="none",
        )

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Set title and labels
        ax.set_title(f"{transform_label}: {metric_label}", fontsize=12)
        ax.set_xlabel(r"$c_0$", fontsize=11)
        ax.set_ylabel(r"$\alpha$", fontsize=11)

        # Set axis limits
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])

        # Add grid for reference
        ax.grid(alpha=0.2, linestyle="--")

        # Set equal aspect ratio for square plot
        ax.set_aspect("equal")

        # turn off the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add stats text
        max_value = np.max(values)
        min_value = np.min(values)
        stats_text = f"Min: {min_value:.2f}\nMax: {max_value:.2f}"

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # Add overall title
    fig.suptitle(f"Dog Ear Transformation: Lipschitz Constants", fontsize=14, y=1.02)

    # Adjust layout but leave room for colorbars
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Add colorbars with proper sizing after tight_layout has been called
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        per_point_key = f"{metric_name}_per_point"

        # Skip if no data
        if (
            transform_name not in metrics
            or per_point_key not in metrics[transform_name]
        ):
            continue

        values = metrics[transform_name][per_point_key]
        vmin = np.percentile(values, 5)
        vmax = np.percentile(values, 95)
        use_log_scale = needs_log_scale(metric_name, values)

        if use_log_scale:
            log_vmin = np.floor(np.log10(vmin))
            log_vmax = np.ceil(np.log10(vmax))
            vmin = 10**log_vmin
            vmax = 10**log_vmax
            norm = LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Get the position and size of the subplot
        pos = ax.get_position()

        # Add properly sized colorbar
        cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=plt.cm.inferno), cax=cbar_ax)
        cbar.set_label(metric_label)

        # Add more ticks for log scale colorbar
        if use_log_scale and (log_vmax - log_vmin) <= 5:
            cbar.set_ticks([10**i for i in range(int(log_vmin), int(log_vmax) + 1)])

    # Save as both PDF and PNG
    plt.savefig(f"{output_path_base}_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}_scatter.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved Dog Ear scatter visualization to {output_path_base}_scatter.pdf and {output_path_base}_scatter.png"
    )


def plot_dog_ear_hexbin(
    metrics: Dict[str, Any],
    transform_name: str,
    metric_names: List[str],
    transformed_params_norm: Dict[str, np.ndarray],
    output_path_base: str,
    gridsize: int = 20,
) -> None:
    """
    Create a hexbin visualization of Dog Ear transformation for both
    Forward and Inverse Lipschitz constants side by side, addressing the overlapping points issue.

    Args:
        metrics: Dictionary of metrics data
        transform_name: Name of the transformation (should be "Dog_Ear")
        metric_names: List of metric names to visualize
        transformed_params_norm: Dictionary of normalized transformed parameters (in [-1,1]² space)
        output_path_base: Base path for saving figures (without extension)
        gridsize: Number of hexagons in the x-direction
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Get normalized uniform samples in [-1,1]² space
    uniform_samples_norm = transformed_params_norm["Identity"][:, :2]

    # Setup common parameters
    transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)

    # Store hexbin objects for later colorbar creation
    hexbin_objects = []

    # For each metric (Forward/Inverse Lipschitz)
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]

        # Check if metric exists for this transformation
        per_point_key = f"{metric_name}_per_point"

        # Skip if no data
        if (
            transform_name not in metrics
            or per_point_key not in metrics[transform_name]
        ):
            ax.text(
                0.5,
                0.5,
                f"No {metric_name} data for {transform_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            hexbin_objects.append(None)
            continue

        # Get metric values
        values = metrics[transform_name][per_point_key]

        # Determine if we need log scale
        use_log_scale = needs_log_scale(metric_name, values)

        # Setup colormap
        cmap = plt.cm.inferno

        # Create hexbin plot - This aggregates points into hexagonal bins
        if use_log_scale:
            # For log scale, we aggregate with 'mean' but apply log scaling to the color
            hexbin = ax.hexbin(
                uniform_samples_norm[:, 0],
                uniform_samples_norm[:, 1],
                C=values,
                gridsize=gridsize,
                cmap=cmap,
                reduce_C_function=np.mean,  # Use mean for aggregation
                mincnt=1,  # Only show hexbins with at least one point
                extent=[-1.05, 1.05, -1.05, 1.05],
            )
        else:
            # For linear scale, we can use regular hexbin
            hexbin = ax.hexbin(
                uniform_samples_norm[:, 0],
                uniform_samples_norm[:, 1],
                C=values,
                gridsize=gridsize,
                cmap=cmap,
                reduce_C_function=np.mean,
                mincnt=1,
                extent=[-1.05, 1.05, -1.05, 1.05],
            )

        # Store hexbin object for colorbar creation later
        hexbin_objects.append(hexbin)

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Set title and labels
        ax.set_title(f"{transform_label}: {metric_label} (Hexbin)", fontsize=12)
        ax.set_xlabel(r"$c_0$", fontsize=11)
        ax.set_ylabel(r"$\alpha$", fontsize=11)

        # Set axis limits
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])

        # Set equal aspect ratio for square plot
        ax.set_aspect("equal")

        # Add grid for reference
        ax.grid(alpha=0.2, linestyle="--")

        # Add stats text
        max_value = np.max(values)
        min_value = np.min(values)
        stats_text = f"Min: {min_value:.2f}\nMax: {max_value:.2f}"

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # Add overall title
    fig.suptitle(
        f"Dog Ear Transformation: Lipschitz Constants (Hexbin)", fontsize=14, y=1.02
    )

    # Adjust layout but leave room for colorbars
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Add colorbars with proper sizing after tight_layout has been called
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        hexbin = hexbin_objects[idx]

        # Skip if no data or no hexbin object
        if hexbin is None:
            continue

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Get the position and size of the subplot
        pos = ax.get_position()

        # Add properly sized colorbar
        cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(hexbin, cax=cbar_ax)
        cbar.set_label(f"Mean {metric_label}")

    # Save as both PDF and PNG
    plt.savefig(f"{output_path_base}_hexbin.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}_hexbin.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved Dog Ear hexbin visualization to {output_path_base}_hexbin.pdf and {output_path_base}_hexbin.png"
    )


def plot_dog_ear_imshow(
    metrics: Dict[str, Any],
    transform_name: str,
    metric_names: List[str],
    transformed_params_norm: Dict[str, np.ndarray],
    output_path_base: str,
    grid_size: int = 20,
) -> None:
    """
    Create a 2D histogram visualization using imshow of Dog Ear transformation for both
    Forward and Inverse Lipschitz constants side by side, using square bins.

    Args:
        metrics: Dictionary of metrics data
        transform_name: Name of the transformation (should be "Dog_Ear")
        metric_names: List of metric names to visualize
        transformed_params_norm: Dictionary of normalized transformed parameters (in [-1,1]² space)
        output_path_base: Base path for saving figures (without extension)
        grid_size: Number of bins in each dimension
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Get normalized uniform samples in [-1,1]² space
    uniform_samples_norm = transformed_params_norm["Identity"][:, :2]

    # Setup common parameters
    transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)

    # Define the bin edges for the histogram
    x_edges = np.linspace(-1.05, 1.05, grid_size + 1)
    y_edges = np.linspace(-1.05, 1.05, grid_size + 1)

    # Store imshow objects for later colorbar creation
    imshow_objects = []

    # For each metric (Forward/Inverse Lipschitz)
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]

        # Check if metric exists for this transformation
        per_point_key = f"{metric_name}_per_point"

        # Skip if no data
        if (
            transform_name not in metrics
            or per_point_key not in metrics[transform_name]
        ):
            ax.text(
                0.5,
                0.5,
                f"No {metric_name} data for {transform_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            imshow_objects.append(None)
            continue

        # Get metric values
        values = metrics[transform_name][per_point_key]

        # Create a 2D histogram
        H, _, _ = np.histogram2d(
            uniform_samples_norm[:, 0],
            uniform_samples_norm[:, 1],
            bins=[x_edges, y_edges],
            weights=values,
        )

        # Count points in each bin for averages
        counts, _, _ = np.histogram2d(
            uniform_samples_norm[:, 0],
            uniform_samples_norm[:, 1],
            bins=[x_edges, y_edges],
        )

        # Avoid division by zero
        mask = counts > 0

        # Calculate mean values per bin
        H_mean = np.zeros_like(H)
        H_mean[mask] = H[mask] / counts[mask]

        # Determine color scale parameters
        vmin = np.percentile(values, 1)
        vmax = np.percentile(values, 99)

        # Determine if we need log scale
        use_log_scale = needs_log_scale(metric_name, values)

        # Setup colormap and normalization
        cmap = plt.cm.inferno

        # Create the imshow plot
        if use_log_scale:
            # Use log norm for the imshow plot
            norm = LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
        else:
            # Use linear norm for the imshow plot
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot the 2D histogram using imshow with correct extent
        im = ax.imshow(
            H_mean.T,  # Transpose for correct orientation
            origin="lower",
            extent=[-1.05, 1.05, -1.05, 1.05],
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )

        # Store imshow object for later colorbar creation
        imshow_objects.append(im)

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Set title and labels
        ax.set_title(f"{transform_label}: {metric_label} (2D Histogram)", fontsize=12)
        ax.set_xlabel(r"$c_0$ (normalized)", fontsize=11)
        ax.set_ylabel(r"$\alpha$ (normalized)", fontsize=11)

        # Add grid for reference
        ax.grid(alpha=0.2, linestyle="--")

        # remove ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add stats text
        max_value = np.max(values)
        min_value = np.min(values)
        stats_text = f"Min: {min_value:.2f}\nMax: {max_value:.2f}"

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # Add overall title
    fig.suptitle(
        f"Dog Ear Transformation: Lipschitz Constants (2D Histogram)",
        fontsize=14,
        y=1.00,
    )

    # # Adjust layout but leave room for colorbars
    # plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Add colorbars with proper sizing after tight_layout has been called
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        im = imshow_objects[idx]

        # Skip if no data or no imshow object
        if im is None:
            continue

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Get the position and size of the subplot
        pos = ax.get_position()

        # Add properly sized colorbar
        cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f"Mean {metric_label}")

    # Save as both PDF and PNG
    plt.savefig(f"{output_path_base}_imshow.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}_imshow.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved Dog Ear 2D histogram visualization to {output_path_base}_imshow.pdf and {output_path_base}_imshow.png"
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Visualize Lipschitz constant variation for Cahn-Hilliard simulations",
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

    return parser.parse_args()


def main() -> None:
    """
    Main function to create publication-quality figures for
    Lipschitz constant variation visualization.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

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

    # Get metrics data and parameters
    metrics = metrics_data["metrics"]

    # Get transformed parameters
    transformed_params_norm = simulation_data.get("transformed_params_norm", {})

    # Create Dog Ear visualizations
    dog_ear_path_base = os.path.join(args.output_dir, "CH_manifolds_Dog_Ear")
    plot_dog_ear_scatter(
        metrics=metrics,
        transform_name="Dog_Ear",
        metric_names=["Forward_Lipschitz", "Inverse_Lipschitz"],
        transformed_params_norm=transformed_params_norm,
        output_path_base=dog_ear_path_base,
    )
    plot_dog_ear_hexbin(
        metrics=metrics,
        transform_name="Dog_Ear",
        metric_names=["Forward_Lipschitz", "Inverse_Lipschitz"],
        transformed_params_norm=transformed_params_norm,
        output_path_base=dog_ear_path_base,
    )
    plot_dog_ear_imshow(
        metrics=metrics,
        transform_name="Dog_Ear",
        metric_names=["Forward_Lipschitz", "Inverse_Lipschitz"],
        transformed_params_norm=transformed_params_norm,
        output_path_base=dog_ear_path_base,
    )

    print("All visualizations completed successfully!")


if __name__ == "__main__":
    main()
