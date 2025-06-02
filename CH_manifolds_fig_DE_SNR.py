"""
Visualize Signal-to-Noise Ratio (SNR) for Lipschitz constants from bootstrapped samples,
specifically for the Dog Ear transformation from Cahn-Hilliard simulations.
Creates a figure with imshow plots of SNR values for both forward and inverse Lipschitz constants
with a shared colorbar for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def collect_bootstrap_data(
    bootstrapped_metrics_list: List[Dict],
    transform_name: str,
    uniform_samples_norm: np.ndarray,
) -> Tuple[List, List, List, List]:
    """
    Collect Lipschitz values and their coordinates from all bootstrap samples.

    Args:
        bootstrapped_metrics_list: List of dictionaries containing bootstrapped metrics
        transform_name: Name of the transformation to collect data for
        uniform_samples_norm: Normalized uniform samples positions in [-1,1]² space

    Returns:
        Tuple of (forward_lip_all_x, forward_lip_all_y, forward_lip_all_values,
                  inverse_lip_all_x, inverse_lip_all_y, inverse_lip_all_values)
    """
    # Lists to collect all coordinate and value data across bootstraps
    forward_lip_all_x = []
    forward_lip_all_y = []
    forward_lip_all_values = []

    inverse_lip_all_x = []
    inverse_lip_all_y = []
    inverse_lip_all_values = []

    # Collect values from each bootstrap sample
    for bootstrap_metrics in bootstrapped_metrics_list:
        transform_metrics = bootstrap_metrics.get(transform_name, {})

        # Skip if no transform metrics
        if not transform_metrics:
            continue

        # Get per-point Lipschitz values for this bootstrap sample
        fwd_lip_pp = transform_metrics.get("Forward_Lipschitz_per_point", [])
        inv_lip_pp = transform_metrics.get("Inverse_Lipschitz_per_point", [])

        # Get subsample indices used for this bootstrap
        subsample_indices = transform_metrics.get("subsample_indices", None)

        print(
            f"Bootstrap sample {len(forward_lip_all_values) + 1}: "
            f"Forward Lipschitz points: {len(fwd_lip_pp)}, "
            f"Inverse Lipschitz points: {len(inv_lip_pp)}, "
            f"Has indices: {subsample_indices is not None}"
        )

        # Skip if no Lipschitz data
        if len(fwd_lip_pp) == 0 or len(inv_lip_pp) == 0:
            continue

        # If we have subsample indices, use them to get the correct coordinates
        if subsample_indices is not None:
            # Make sure lengths match
            if len(fwd_lip_pp) != len(subsample_indices) or len(inv_lip_pp) != len(
                subsample_indices
            ):
                print(
                    f"Warning: Mismatch in Lipschitz points ({len(fwd_lip_pp)}/{len(inv_lip_pp)}) and indices ({len(subsample_indices)})"
                )
                continue

            # For each point in this bootstrap sample
            for i in range(len(fwd_lip_pp)):
                # Get original index in the full dataset
                orig_idx = subsample_indices[i]

                # Get coordinates from the original uniform samples
                if orig_idx < len(uniform_samples_norm):
                    x, y = (
                        uniform_samples_norm[orig_idx, 0],
                        uniform_samples_norm[orig_idx, 1],
                    )

                    # Forward Lipschitz
                    if not np.isnan(fwd_lip_pp[i]):
                        forward_lip_all_x.append(x)
                        forward_lip_all_y.append(y)
                        forward_lip_all_values.append(fwd_lip_pp[i])

                    # Inverse Lipschitz
                    if not np.isnan(inv_lip_pp[i]):
                        inverse_lip_all_x.append(x)
                        inverse_lip_all_y.append(y)
                        inverse_lip_all_values.append(inv_lip_pp[i])
                else:
                    print(
                        f"Warning: Index {orig_idx} is out of bounds for uniform_samples_norm with shape {uniform_samples_norm.shape}"
                    )
        else:
            # Backward compatibility for data without indices
            # Make sure we have valid data that matches the number of points
            if len(fwd_lip_pp) == len(uniform_samples_norm) and len(inv_lip_pp) == len(
                uniform_samples_norm
            ):
                # For each point in this bootstrap sample
                for i in range(len(fwd_lip_pp)):
                    # Get coordinates directly
                    x, y = uniform_samples_norm[i, 0], uniform_samples_norm[i, 1]

                    # Forward Lipschitz
                    if not np.isnan(fwd_lip_pp[i]):
                        forward_lip_all_x.append(x)
                        forward_lip_all_y.append(y)
                        forward_lip_all_values.append(fwd_lip_pp[i])

                    # Inverse Lipschitz
                    if not np.isnan(inv_lip_pp[i]):
                        inverse_lip_all_x.append(x)
                        inverse_lip_all_y.append(y)
                        inverse_lip_all_values.append(inv_lip_pp[i])

    print(
        f"Collected {len(forward_lip_all_values)} forward Lipschitz values across all bootstraps"
    )
    print(
        f"Collected {len(inverse_lip_all_values)} inverse Lipschitz values across all bootstraps"
    )

    return (
        forward_lip_all_x,
        forward_lip_all_y,
        forward_lip_all_values,
        inverse_lip_all_x,
        inverse_lip_all_y,
        inverse_lip_all_values,
    )


def compute_binned_snr(
    bootstrapped_metrics_list: List[Dict],
    transform_name: str,
    uniform_samples_norm: np.ndarray,
    grid_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SNR values for bins by aggregating Lipschitz values from bootstrapped samples
    into spatial bins and then calculating SNR per bin.

    Args:
        bootstrapped_metrics_list: List of dictionaries containing bootstrapped metrics
        transform_name: Name of the transformation to calculate SNR for
        uniform_samples_norm: Normalized uniform samples positions in [-1,1]² space
        grid_size: Number of bins in each dimension

    Returns:
        Tuple of (forward_snr_grid, inverse_snr_grid) as 2D arrays
    """
    # Collect all Lipschitz values and coordinates from all bootstraps
    (
        forward_lip_all_x,
        forward_lip_all_y,
        forward_lip_all_values,
        inverse_lip_all_x,
        inverse_lip_all_y,
        inverse_lip_all_values,
    ) = collect_bootstrap_data(
        bootstrapped_metrics_list, transform_name, uniform_samples_norm
    )

    # Define bin edges
    x_edges = np.linspace(-1.05, 1.05, grid_size + 1)
    y_edges = np.linspace(-1.05, 1.05, grid_size + 1)

    # Initialize grids for results
    forward_mean_grid = np.zeros((grid_size, grid_size))
    forward_std_grid = np.zeros((grid_size, grid_size))
    forward_snr_grid = np.zeros((grid_size, grid_size))

    inverse_mean_grid = np.zeros((grid_size, grid_size))
    inverse_std_grid = np.zeros((grid_size, grid_size))
    inverse_snr_grid = np.zeros((grid_size, grid_size))

    # Fill grids with NaN for bins with no data
    forward_mean_grid.fill(np.nan)
    forward_std_grid.fill(np.nan)
    forward_snr_grid.fill(np.nan)

    inverse_mean_grid.fill(np.nan)
    inverse_std_grid.fill(np.nan)
    inverse_snr_grid.fill(np.nan)

    if forward_lip_all_values:
        # Process Forward Lipschitz bins
        # Digitize the x,y coordinates to bin indices
        x_indices = np.digitize(forward_lip_all_x, x_edges) - 1
        y_indices = np.digitize(forward_lip_all_y, y_edges) - 1

        # Filter out points outside the bins
        valid_indices = (
            (x_indices >= 0)
            & (x_indices < grid_size)
            & (y_indices >= 0)
            & (y_indices < grid_size)
        )

        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]
        values = np.array(forward_lip_all_values)[valid_indices]

        # Loop through bins and calculate stats
        bin_dict = {}
        for i, (x_idx, y_idx, val) in enumerate(zip(x_indices, y_indices, values)):
            bin_key = (x_idx, y_idx)
            if bin_key not in bin_dict:
                bin_dict[bin_key] = []
            bin_dict[bin_key].append(val)

        # Calculate statistics for each bin
        for (x_idx, y_idx), bin_values in bin_dict.items():
            if len(bin_values) > 1:  # Need at least 2 values to compute std dev
                mean_val = np.mean(bin_values)
                std_val = np.std(bin_values)
                # Compute SNR (mean/std) with small epsilon to avoid division by zero
                snr = mean_val / (std_val + 1e-9) if std_val > 1e-12 else np.nan

                forward_mean_grid[x_idx, y_idx] = mean_val
                forward_std_grid[x_idx, y_idx] = std_val
                forward_snr_grid[x_idx, y_idx] = snr

                # if there are fewer than 3 points in the bin, set SNR to NaN
                if len(bin_values) < 3:
                    forward_snr_grid[x_idx, y_idx] = np.nan
                    forward_mean_grid[x_idx, y_idx] = np.nan
                    forward_std_grid[x_idx, y_idx] = np.nan

    if inverse_lip_all_values:
        # Process Inverse Lipschitz bins
        # Digitize the x,y coordinates to bin indices
        x_indices = np.digitize(inverse_lip_all_x, x_edges) - 1
        y_indices = np.digitize(inverse_lip_all_y, y_edges) - 1

        # Filter out points outside the bins
        valid_indices = (
            (x_indices >= 0)
            & (x_indices < grid_size)
            & (y_indices >= 0)
            & (y_indices < grid_size)
        )

        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]
        values = np.array(inverse_lip_all_values)[valid_indices]

        # Loop through bins and calculate stats
        bin_dict = {}
        for i, (x_idx, y_idx, val) in enumerate(zip(x_indices, y_indices, values)):
            bin_key = (x_idx, y_idx)
            if bin_key not in bin_dict:
                bin_dict[bin_key] = []
            bin_dict[bin_key].append(val)

        # Calculate statistics for each bin
        for (x_idx, y_idx), bin_values in bin_dict.items():
            if len(bin_values) > 1:  # Need at least 2 values to compute std dev
                mean_val = np.mean(bin_values)
                std_val = np.std(bin_values)
                # Compute SNR (mean/std) with small epsilon to avoid division by zero
                snr = mean_val / (std_val + 1e-9) if std_val > 1e-12 else np.nan

                inverse_mean_grid[x_idx, y_idx] = mean_val
                inverse_std_grid[x_idx, y_idx] = std_val
                inverse_snr_grid[x_idx, y_idx] = snr

                # if there are fewer than 3 points in the bin, set SNR to NaN
                if len(bin_values) < 3:
                    inverse_snr_grid[x_idx, y_idx] = np.nan
                    inverse_mean_grid[x_idx, y_idx] = np.nan
                    inverse_std_grid[x_idx, y_idx] = np.nan

    return forward_snr_grid, inverse_snr_grid


def plot_dog_ear_snr_imshow(
    forward_snr_grid: np.ndarray,
    inverse_snr_grid: np.ndarray,
    output_path_base: str,
    transform_name: str = "Dog_Ear",
) -> None:
    """
    Create an imshow visualization of SNR values for both forward and inverse Lipschitz constants
    with a shared colorbar, using pre-computed binned SNR values.

    Args:
        forward_snr_grid: 2D array of SNR values for forward Lipschitz constant
        inverse_snr_grid: 2D array of SNR values for inverse Lipschitz constant
        output_path_base: Base path for saving the figure (without extension)
        transform_name: Name of the transformation (default is "Dog_Ear")
        grid_size: Number of bins in each dimension of the grid
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Setup common parameters
    transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)

    # Store imshow objects for later colorbar creation
    imshow_objects = []

    # Combine data to determine global color scale
    all_snrs = []
    for snr_grid in [forward_snr_grid, inverse_snr_grid]:
        valid_snrs = snr_grid[~np.isnan(snr_grid)]
        if len(valid_snrs) > 0:
            all_snrs.extend(valid_snrs)

    if not all_snrs:
        print("No valid SNR data to plot")
        return

    # Determine global color scale with percentile clipping to avoid outliers
    vmin = max(np.percentile(all_snrs, 5), 0)  # SNR should be positive
    vmax = np.percentile(all_snrs, 95)

    print(f"Color scale range: {vmin:.2f} to {vmax:.2f}")

    # Create a single normalization for both plots
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    # Metrics to plot
    metric_names = ["Forward_Lipschitz_SNR", "Inverse_Lipschitz_SNR"]
    snr_grid_list = [forward_snr_grid, inverse_snr_grid]

    # For each metric (Forward/Inverse Lipschitz SNR)
    for idx, (metric_name, snr_grid) in enumerate(zip(metric_names, snr_grid_list)):
        ax = axes[idx]

        if np.all(np.isnan(snr_grid)):
            ax.text(
                0.5,
                0.5,
                f"No {metric_name} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            imshow_objects.append(None)
            continue

        print(f"SNR grid shape: {snr_grid.shape}")
        print(f"Number of valid bins: {np.sum(~np.isnan(snr_grid))}")

        # Plot the grid directly using imshow
        im = ax.imshow(
            snr_grid.T,  # Transpose for correct orientation
            origin="lower",
            extent=[-1.05, 1.05, -1.05, 1.05],
            interpolation="nearest",
            cmap=cmap,
            norm=norm,  # Use the common normalization
            aspect="equal",
        )

        # Store imshow object for later colorbar creation
        imshow_objects.append(im)

        # Get proper metric label
        metric_label = METRIC_LABELS.get(metric_name, metric_name)

        # Set title and labels
        ax.set_title(f"{transform_label}: {metric_label}", fontsize=12)
        ax.set_xlabel(r"$x$", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"$y$", fontsize=11)

        # Set axis limits to show the full [-1,1]² range with a small margin
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])

        # Add grid for reference
        ax.grid(alpha=0.2, linestyle="--")

        # Add axis ticks
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

        # Add stats text
        valid_snrs = snr_grid[~np.isnan(snr_grid)]
        if len(valid_snrs) > 0:
            max_value = np.max(valid_snrs)
            min_value = np.min(valid_snrs)
            mean_value = np.mean(valid_snrs)
            stats_text = (
                f"Min: {min_value:.2f}\nMean: {mean_value:.2f}\nMax: {max_value:.2f}"
            )

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
        f"Signal-to-Noise Ratio of Lipschitz Constants (Bootstrapped)",
        fontsize=14,
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Add a single colorbar for both plots
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("Signal-to-Noise Ratio (SNR)")

    # Save as both PDF and PNG
    plt.savefig(f"{output_path_base}_snr.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}_snr.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved {transform_name} SNR visualization to {output_path_base}_snr.pdf and {output_path_base}_snr.png"
    )


def plot_inverse_snr_imshow(
    inverse_snr_grid: np.ndarray,
    output_path_base: str,
    transform_name: str = "Dog_Ear",
) -> None:
    """
    Create a standalone visualization of inverse SNR values (1/SNR).
    This plot just shows the inverse of the SNR for the inverse Lipschitz constant.

    Args:
        inverse_snr_grid: 2D array of SNR values for inverse Lipschitz constant
        output_path_base: Base path for saving the figure (without extension)
        transform_name: Name of the transformation (default is "Dog_Ear")
        grid_size: Number of bins in each dimension of the grid
    """
    # Create a figure for just the inverse Lipschitz
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Setup parameters
    transform_label = TRANSFORM_LABELS.get(transform_name, transform_name)

    if np.all(np.isnan(inverse_snr_grid)):
        ax.text(
            0.5,
            0.5,
            "No Inverse Lipschitz SNR data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.close()
        return

    # Calculate the inverse of SNR (1/SNR)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_snr_inv = np.divide(1.0, inverse_snr_grid)
        inverse_snr_inv[np.isinf(inverse_snr_inv)] = np.nan

    # Determine color scale with percentile clipping
    valid_inv = inverse_snr_inv[~np.isnan(inverse_snr_inv)]
    if len(valid_inv) == 0:
        print("No valid inverse SNR data to plot")
        plt.close()
        return

    vmin = np.percentile(valid_inv, 5)
    vmax = np.percentile(valid_inv, 95)

    # Make sure vmin is positive for log scale if needed
    vmin = max(vmin, 1e-5)

    print(f"Inverse SNR (1/SNR) range: {vmin:.4f} to {vmax:.4f}")

    # Create normalization and colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = (
        plt.cm.plasma
    )  # Using plasma for inverse SNR to distinguish from the SNR plots

    # Plot the grid using imshow
    im = ax.imshow(
        inverse_snr_inv.T,  # Transpose for correct orientation
        origin="lower",
        extent=[-1.05, 1.05, -1.05, 1.05],
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )

    # Get proper metric label
    metric_label = r"$1/\text{SNR}_{\text{Lip}(f^{-1})}$"

    # Set title and labels
    ax.set_title(f"{transform_label} (DE): {metric_label}", fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)

    # Set axis limits
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])

    # Add grid for reference
    ax.grid(alpha=0.2, linestyle="--")

    # Add axis ticks
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    # Add stats text for the inverse of SNR
    if len(valid_inv) > 0:
        max_value = np.nanmax(inverse_snr_inv)
        min_value = np.nanmin(inverse_snr_inv)
        mean_value = np.nanmean(inverse_snr_inv)
        stats_text = (
            f"Min: {min_value:.4f}\nMean: {mean_value:.4f}\nMax: {max_value:.4f}"
        )

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # Add overall title
    plt.suptitle(
        f"Inverse of Signal-to-Noise Ratio for Inverse Lipschitz",
        fontsize=14,
        y=0.98,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(f"{metric_label} (Higher values indicate more uncertainty)")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save as both PDF and PNG
    plt.savefig(f"{output_path_base}_inverse_snr_inv.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}_inverse_snr_inv.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved {transform_name} inverse SNR visualization to {output_path_base}_inverse_snr_inv.pdf and {output_path_base}_inverse_snr_inv.png"
    )


def plot_comparison_inverse_snr(
    swiss_roll_snr_grid: np.ndarray,
    dog_ear_snr_grid: np.ndarray,
    output_path_base: str,
) -> None:
    """
    Create a side-by-side comparison of inverse SNR values (1/SNR) for
    Swiss Roll (well-behaved) and Dog Ear transformations.

    Args:
        swiss_roll_snr_grid: 2D array of SNR values for Swiss Roll inverse Lipschitz constant
        dog_ear_snr_grid: 2D array of SNR values for Dog Ear inverse Lipschitz constant
        output_path_base: Base path for saving the figure (without extension)
        grid_size: Number of bins in each dimension of the grid
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Setup transformation labels
    swiss_roll_label = TRANSFORM_LABELS.get("Swiss_Roll", "Swiss Roll")
    dog_ear_label = TRANSFORM_LABELS.get("Dog_Ear", "Dog Ear")

    # Transform names for the plot
    transform_names = ["Swiss_Roll", "Dog_Ear"]
    transform_labels = [swiss_roll_label, dog_ear_label]
    snr_grids = [swiss_roll_snr_grid, dog_ear_snr_grid]

    # Check if we have valid data for both transformations
    valid_count = 0
    all_inv_snr_values = []

    for idx, snr_grid in enumerate(snr_grids):
        if snr_grid is None or np.all(np.isnan(snr_grid)):
            axes[idx].text(
                0.5,
                0.5,
                f"No Inverse Lipschitz SNR data for {transform_names[idx]}",
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )
        else:
            # Calculate the inverse of SNR (1/SNR)
            with np.errstate(divide="ignore", invalid="ignore"):
                inverse_snr_inv = np.divide(1.0, snr_grid)
                inverse_snr_inv[np.isinf(inverse_snr_inv)] = np.nan

            # Collect valid values for color scale calculation
            valid_values = inverse_snr_inv[~np.isnan(inverse_snr_inv)]
            if len(valid_values) > 0:
                all_inv_snr_values.extend(valid_values)
                valid_count += 1

    if valid_count == 0:
        print("No valid inverse SNR data to plot for either transformation")
        plt.close()
        return

    # Determine global color scale with percentile clipping
    vmin = np.percentile(all_inv_snr_values, 5)
    vmax = np.percentile(all_inv_snr_values, 95)

    # Make sure vmin is positive
    vmin = max(vmin, 1e-5)

    print(f"Comparison inverse SNR (1/SNR) range: {vmin:.4f} to {vmax:.4f}")

    # Create a single normalization for both plots
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma  # Using plasma for inverse SNR

    # For each transformation
    for idx, (transform_name, transform_label, snr_grid) in enumerate(
        zip(transform_names, transform_labels, snr_grids)
    ):
        ax = axes[idx]

        if snr_grid is None or np.all(np.isnan(snr_grid)):
            continue  # Skip if no data

        # Calculate the inverse of SNR (1/SNR)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse_snr_inv = np.divide(1.0, snr_grid)
            inverse_snr_inv[np.isinf(inverse_snr_inv)] = np.nan

        # Plot the grid using imshow
        im = ax.imshow(
            inverse_snr_inv.T,  # Transpose for correct orientation
            origin="lower",
            extent=[-1.05, 1.05, -1.05, 1.05],
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )

        # Set title and labels
        metric_label = r"$1/\text{SNR}_{\text{Lip}(f^{-1})}$"
        abbreviation = "Dog Ear" if transform_name == "Dog_Ear" else "Swiss Roll"
        ax.set_title(f"{abbreviation}: {metric_label}", fontsize=12)
        ax.set_xlabel(r"$x$", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"$y$", fontsize=11)

        # Set axis limits
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])

        # Add grid for reference
        ax.grid(alpha=0.2, linestyle="--")

        # Add axis ticks
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

        # Add stats text for the inverse of SNR
        valid_inv = inverse_snr_inv[~np.isnan(inverse_snr_inv)]
        if len(valid_inv) > 0:
            max_value = np.nanmax(inverse_snr_inv)
            min_value = np.nanmin(inverse_snr_inv)
            mean_value = np.nanmean(inverse_snr_inv)
            stats_text = (
                f"Min: {min_value:.4f}\nMean: {mean_value:.4f}\nMax: {max_value:.4f}"
            )

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
        f"Inverse of SNR for Inverse Lipschitz (Higher = More Uncertainty)",
        fontsize=14,
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Add a single colorbar for both plots
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(r"$1/\text{SNR}_{\text{Lip}(f^{-1})}$ (Uncertainty)")

    # Save as both PDF and PNG
    output_path = f"{output_path_base}_comparison_inverse_snr"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")

    plt.close()

    print(
        f"Saved comparison inverse SNR visualization to {output_path}.pdf and {output_path}.png"
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Visualize SNR of Lipschitz constants from bootstrapped samples for Dog Ear transformation",
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
        "--transform",
        type=str,
        default="Dog_Ear",
        help="Transformation to visualize",
    )

    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate comparison plots between well-behaved (Swiss_Roll) and non-trivial (Dog_Ear) transformations",
    )

    parser.add_argument(
        "--well_behaved",
        type=str,
        default="Swiss_Roll",
        help="Well-behaved transformation to use in comparison (default: Swiss_Roll)",
    )

    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="Number of bins in each dimension for SNR computation",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to create a publication-quality figure for
    SNR visualization of Lipschitz constants from bootstrapped samples.
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

    # Load metrics data
    print(f"Loading data from: {metrics_file}")
    try:
        metrics_data = load_metrics_data(metrics_file)
    except Exception as e:
        print(f"Error loading metrics data: {e}")
        return

    # Get source data path (if available)
    source_data_path = metrics_data.get("source_data_path")
    if not source_data_path or not os.path.exists(source_data_path):
        print(f"Source data file not found: {source_data_path}")
        return

    # Load simulation data
    print(f"Loading simulation data from: {source_data_path}")
    try:
        simulation_data = load_simulation_data(source_data_path)
    except Exception as e:
        print(f"Error loading simulation data: {e}")
        return

    # Get bootstrapped metrics
    bootstrapped_metrics_list = metrics_data.get("bootstrapped_metrics", [])

    if not bootstrapped_metrics_list:
        print("No bootstrapped metrics found in the data")
        return

    # Define transformations to process
    transform_name = args.transform  # Main transformation (Dog_Ear by default)
    well_behaved_name = args.well_behaved  # Well-behaved transformation for comparison

    # Get transformed parameters normalization
    transformed_params_norm = simulation_data.get("transformed_params_norm", {})

    # Check if we have the transformations in the data
    required_transforms = [transform_name, well_behaved_name]
    missing_transforms = []

    for t_name in required_transforms:
        if t_name not in transformed_params_norm:
            missing_transforms.append(t_name)

    if missing_transforms:
        print(f"Missing transformed parameters for: {', '.join(missing_transforms)}")
        if transform_name in missing_transforms:
            return

    # Get coordinates for uniform samples
    uniform_samples_norm = transformed_params_norm["Identity"][:, :2]

    # Compute binned SNR values for the main transformation
    print(f"Computing binned SNR values for {transform_name}...")
    grid_size = args.grid_size
    forward_snr_grid, inverse_snr_grid = compute_binned_snr(
        bootstrapped_metrics_list,
        transform_name,
        uniform_samples_norm,
        grid_size=grid_size,
    )

    if np.all(np.isnan(forward_snr_grid)) and np.all(np.isnan(inverse_snr_grid)):
        print(f"No valid SNR values could be computed for {transform_name}")
        return

    # Generate the output path
    output_path_base = os.path.join(args.output_dir, f"CH_manifolds_{transform_name}")

    # Create the visualization for the main transformation
    plot_dog_ear_snr_imshow(
        forward_snr_grid=forward_snr_grid,
        inverse_snr_grid=inverse_snr_grid,
        output_path_base=output_path_base,
        transform_name=transform_name,
    )

    # Create the inverse SNR visualization for the main transformation
    plot_inverse_snr_imshow(
        inverse_snr_grid=inverse_snr_grid,
        output_path_base=output_path_base,
        transform_name=transform_name,
    )

    # Compute Swiss Roll SNR only if it's available
    swiss_roll_inverse_snr_grid = None
    if well_behaved_name not in missing_transforms:
        print(f"Computing binned SNR values for {well_behaved_name}...")
        sr_forward_snr_grid, sr_inverse_snr_grid = compute_binned_snr(
            bootstrapped_metrics_list,
            well_behaved_name,
            uniform_samples_norm,
            grid_size=grid_size,
        )

        if not np.all(np.isnan(sr_inverse_snr_grid)):
            swiss_roll_inverse_snr_grid = sr_inverse_snr_grid

    # Create the comparison plot if Swiss Roll data is available
    if swiss_roll_inverse_snr_grid is not None:
        # Create a comparison path
        comparison_path_base = os.path.join(args.output_dir, "CH_manifolds_comparison")

        # Create the comparison visualization
        plot_comparison_inverse_snr(
            swiss_roll_snr_grid=swiss_roll_inverse_snr_grid,
            dog_ear_snr_grid=inverse_snr_grid,
            output_path_base=comparison_path_base,
        )
    else:
        print(f"Skipping comparison plot: No valid SNR data for {well_behaved_name}")

    print("SNR visualization completed successfully!")


if __name__ == "__main__":
    main()
