"""
Generate a visualization of Cahn-Hilliard simulations in a 3D parameter space grid.
This script creates a publication-ready figure showing a 3x3x3 grid of simulations
varying initial concentration, alpha, and beta parameters.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from src_CH import cahn_hilliard_simulation
from utils import get_device

# Default parameters
DEFAULT_CONFIG = {
    "device": None,  # Will auto-detect
    "output_dir": "figures",
    "grid_size": 128,  # Grid size for each simulation
    "nsteps": 1000,  # Simulation steps
    "noise_amplitude": 0.01,
    "seed": 42,
    "c_init_range": (0.3, 0.7),  # Initial concentration range
    "alpha_range": (0.7, 1.3),  # Alpha parameter range
    "beta_range": (0.7, 1.3),  # Beta parameter range
    "W_base": 1.0,  # Bulk free energy parameter
    "kappa_base": 1.5,  # Gradient energy parameter
    "M_base": 1.0,  # Mobility
    "dt_base": 0.5,  # Time step
    "grid_points": 3,  # Number of grid points in each parameter dimension
    # Parameter layout configuration - which parameter varies along which dimension
    "block_param": "c_init",  # Parameter that varies between blocks ('c_init', 'alpha', or 'beta')
    "row_param": "alpha",  # Parameter that varies along rows ('c_init', 'alpha', or 'beta')
    "col_param": "beta",  # Parameter that varies along columns ('c_init', 'alpha', or 'beta')
}


def setup_parameter_grid(config):
    """Create a 3D grid of parameter combinations"""
    # Linear spacing for each parameter
    c_init_values = np.linspace(
        config["c_init_range"][0], config["c_init_range"][1], config["grid_points"]
    )
    alpha_values = np.linspace(
        config["alpha_range"][0], config["alpha_range"][1], config["grid_points"]
    )
    beta_values = np.linspace(
        config["beta_range"][0], config["beta_range"][1], config["grid_points"]
    )

    # Create parameter grid
    param_grid = []
    for c in c_init_values:
        for a in alpha_values:
            for b in beta_values:
                param_grid.append((c, a, b))

    param_values = {"c_init": c_init_values, "alpha": alpha_values, "beta": beta_values}

    return param_grid, param_values


def run_simulations(param_grid, config):
    """Run Cahn-Hilliard simulations for all parameter combinations"""
    device = get_device(config["device"])
    print(f"Using device: {device}")

    # Generate common noise pattern for consistency
    torch.manual_seed(config["seed"])
    noise_pattern = torch.randn(config["grid_size"], config["grid_size"], device=device)

    results = []
    total = len(param_grid)

    for i, (c_init, alpha, beta) in enumerate(param_grid):
        print(
            f"Running simulation {i+1}/{total}: c_init={c_init:.2f}, alpha={alpha:.2f}, beta={beta:.2f}"
        )

        # Create initial condition
        c0 = torch.full(
            (1, config["grid_size"], config["grid_size"]),
            c_init,
            dtype=torch.float32,
            device=device,
        )
        c0 += config["noise_amplitude"] * noise_pattern

        # Set parameters
        W = torch.tensor([config["W_base"]], device=device)
        dt = torch.tensor([config["dt_base"]], device=device)
        L = torch.tensor([float(config["grid_size"])], device=device)

        # Compute derived parameters
        kappa = config["kappa_base"] * (alpha**2) * (beta**2)
        kappa = torch.tensor([kappa], device=device)

        M = config["M_base"] * (alpha**2) / (beta**2)
        M = torch.tensor([M], device=device)

        # Run simulation
        result = cahn_hilliard_simulation(
            c0=c0,
            W=W,
            kappa=kappa,
            L=L,
            Nsteps=config["nsteps"],
            dt=dt,
            M=M,
            progress_bar=False,
        )

        results.append(result[0].cpu().numpy())

    return results


def create_figure(results, param_grid, param_values, config):
    """Create publication-ready figure showing multiple 3x3 grids

    The layout is configurable with block_param, row_param, and col_param.
    """
    # Set publication-quality parameters
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
        }
    )

    # Extract configuration
    n_grid = config["grid_points"]
    block_param = config["block_param"]
    row_param = config["row_param"]
    col_param = config["col_param"]

    # Get figure size from config or use sensible default
    figsize = config.get(
        "figsize", (n_grid * len(param_values[block_param]) * 1.5, n_grid * 1.5)
    )

    # Ensure all parameters are different
    params = set([block_param, row_param, col_param])
    if len(params) != 3:
        raise ValueError(
            f"Parameters must be different, got: {block_param}, {row_param}, {col_param}"
        )

    # Get parameter values
    block_values = param_values[block_param]
    row_values = param_values[row_param]
    col_values = param_values[col_param]

    n_blocks = len(block_values)

    # Create a figure with appropriate spacing between subplots
    fig, all_axes = plt.subplots(
        n_grid,
        n_grid * n_blocks,
        figsize=figsize,
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},  # Minimal spacing between plots
    )

    # If there's only one row, ensure all_axes is 2D
    if n_grid == 1:
        all_axes = all_axes.reshape(1, -1)

    # Format values with LaTeX
    param_formatters = {
        "c_init": lambda x: f"$\\eta_0 = {x:.2f}$",
        "alpha": lambda x: f"$\\alpha = {x:.2f}$",
        "beta": lambda x: f"$\\beta = {x:.2f}$",
    }

    # Create parameter labels
    block_labels = [param_formatters[block_param](v) for v in block_values]
    row_labels = [param_formatters[row_param](v) for v in row_values]
    col_labels = [param_formatters[col_param](v) for v in col_values]

    # Function to find a specific result from the grid
    def find_result(c_init, alpha, beta):
        param_tuple = (c_init, alpha, beta)
        for i, p in enumerate(param_grid):
            if np.allclose(p, param_tuple):
                return results[i]
        return None

    # # Add a clean, focused title
    # fig.suptitle("Cahn-Hilliard Phase Separation", fontsize=16, y=0.98)

    # # Draw vertical separation lines between blocks using figure coordinates instead
    # for i in range(1, n_blocks):
    #     # Calculate the x-position for separation line in figure coordinates
    #     line_x = (i * n_grid) / (n_grid * n_blocks)
    #     # Create line in figure coordinates
    #     line = plt.Line2D(
    #         [line_x, line_x],  # x-positions: [start, end]
    #         [0.05, 0.9],  # y-positions: [bottom, top] in figure coordinates
    #         transform=fig.transFigure,  # Use figure coordinates
    #         color="black",
    #         linestyle="-",
    #         linewidth=1.5,
    #         zorder=10,  # Make sure it's drawn on top
    #     )
    #     fig.lines.append(line)

    # Iterate through all blocks
    for block_idx, block_value in enumerate(block_values):
        # Add block title with better positioning
        if n_grid % 2 == 1:  # Odd number of grid points
            # Place title over middle column of block
            middle_col = block_idx * n_grid + (n_grid // 2)
            all_axes[0, middle_col].set_title(
                block_labels[block_idx], fontsize=12, fontweight="bold", pad=10
            )
        else:  # Even number of grid points
            # Calculate middle position for this block
            start_col = block_idx * n_grid
            end_col = (block_idx + 1) * n_grid - 1
            # Create an invisible twin axis to hold the title
            middle_ax = all_axes[0, start_col].twinx()
            middle_ax.set_title(
                block_labels[block_idx], fontsize=12, fontweight="bold", pad=10
            )
            middle_ax.axis("off")

        # Iterate through rows and columns
        for row_idx, row_value in enumerate(row_values):
            for col_idx, col_value in enumerate(col_values):
                # Calculate the position in the grid
                grid_col = col_idx + (block_idx * n_grid)
                grid_row = row_idx

                # Get the corresponding axis
                ax = all_axes[grid_row, grid_col]

                # Map parameter values to c_init, alpha, beta
                c_init = (
                    row_value
                    if block_param != "c_init" and row_param == "c_init"
                    else (
                        col_value
                        if block_param != "c_init" and col_param == "c_init"
                        else block_value
                    )
                )

                alpha = (
                    row_value
                    if block_param != "alpha" and row_param == "alpha"
                    else (
                        col_value
                        if block_param != "alpha" and col_param == "alpha"
                        else block_value
                    )
                )

                beta = (
                    row_value
                    if block_param != "beta" and row_param == "beta"
                    else (
                        col_value
                        if block_param != "beta" and col_param == "beta"
                        else block_value
                    )
                )

                # Get the result
                result = find_result(c_init, alpha, beta)

                if result is None:
                    print(
                        f"Warning: No result for c_init={c_init:.2f}, alpha={alpha:.2f}, beta={beta:.2f}"
                    )
                    continue

                # Plot the result
                im = ax.imshow(result, cmap="viridis", interpolation="nearest")

                # Add a thin border around the subplot
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(0.5)

                # Clean up axes
                ax.set_xticks([])
                ax.set_yticks([])

                # Add minimal, focused labels
                if row_idx == 0 and col_idx == 0:
                    # Add small, subtle parameter labels in top-left corner only
                    ax.text(
                        5,
                        5,
                        col_labels[col_idx],
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.7, pad=1),
                    )
                    ax.text(
                        5,
                        result.shape[0] - 10,
                        row_labels[row_idx],
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.7, pad=1),
                    )
                elif row_idx == 0:
                    # Just add column label for the first row
                    ax.text(
                        5,
                        5,
                        col_labels[col_idx],
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.7, pad=1),
                    )
                elif col_idx == 0:
                    # Just add row label for the first column
                    ax.text(
                        5,
                        result.shape[0] - 10,
                        row_labels[row_idx],
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.7, pad=1),
                    )

    # # Adjust layout
    # plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)

    # Create directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    # Save the figure
    fig_path = os.path.join(config["output_dir"], "fig_3D_CH_space")
    plt.savefig(f"{fig_path}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{fig_path}.png", bbox_inches="tight", dpi=300)
    print(f"Figure saved as {fig_path}.pdf and {fig_path}.png")

    return fig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate 3D parameter space visualization for Cahn-Hilliard simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--grid_size",
        type=int,
        default=DEFAULT_CONFIG["grid_size"],
        help="Size of simulation grid",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=DEFAULT_CONFIG["nsteps"],
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--noise_amplitude",
        type=float,
        default=DEFAULT_CONFIG["noise_amplitude"],
        help="Amplitude of noise in initial conditions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help="Directory to save output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG["device"],
        help="Device to use for computation (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--grid_points",
        type=int,
        default=DEFAULT_CONFIG["grid_points"],
        help="Number of points along each parameter dimension",
    )

    # Parameter layout options
    parser.add_argument(
        "--block_param",
        type=str,
        default=DEFAULT_CONFIG["block_param"],
        choices=["c_init", "alpha", "beta"],
        help="Parameter that varies between blocks",
    )
    parser.add_argument(
        "--row_param",
        type=str,
        default=DEFAULT_CONFIG["row_param"],
        choices=["c_init", "alpha", "beta"],
        help="Parameter that varies along rows",
    )
    parser.add_argument(
        "--col_param",
        type=str,
        default=DEFAULT_CONFIG["col_param"],
        choices=["c_init", "alpha", "beta"],
        help="Parameter that varies along columns",
    )

    return parser.parse_args()


def main():
    """Run 3D parameter grid of Cahn-Hilliard simulations and create visualization"""
    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Update with command line arguments
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # Validate parameter choices
    params = [config["block_param"], config["row_param"], config["col_param"]]
    if len(set(params)) != 3:
        raise ValueError(f"Parameters must be all different: {params}")

    print(f"Using configuration: {config}")

    # Setup parameter grid
    param_grid, param_values = setup_parameter_grid(config)

    # Run simulations
    results = run_simulations(param_grid, config)

    # Create figure
    fig = create_figure(results, param_grid, param_values, config)

    return fig


if __name__ == "__main__":
    main()
