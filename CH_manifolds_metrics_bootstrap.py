"""
Compute metrics between original points and simulation results from Cahn-Hilliard simulations.
"""

import math
import torch
import numpy as np
import os
import argparse
import glob
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime
from scipy import special

from utils import (
    compute_metrics_consolidated,
    TopoAELoss,
    RTDLoss,
    MinMaxRTDLoss,
    get_device,
    compute_empirical_lipschitz,
    levina_bickel_dim_multi_k,
    estimate_poisson_intensity,
    estimate_jacobian_delta_batched,
)
from utils_mst import get_mst_edges


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


def get_next_metrics_filename(data_filename: str) -> str:
    """
    Generate the next metrics filename by adding or incrementing a numeric suffix.

    Args:
        data_filename: Path to the data file

    Returns:
        Path to the next metrics filename
    """
    base_path = data_filename.rsplit(".", 1)[0]  # Remove extension
    metrics_pattern = f"{base_path}_metrics_*.npy"
    existing_files = glob.glob(metrics_pattern)

    if not existing_files:
        return f"{base_path}_metrics_1.npy"

    # Extract the numeric suffixes
    suffixes = []
    for file in existing_files:
        try:
            suffix = int(file.rsplit("_", 1)[1].split(".")[0])
            suffixes.append(suffix)
        except ValueError:
            continue

    if not suffixes:
        return f"{base_path}_metrics_1.npy"

    # Return the next number
    return f"{base_path}_metrics_{max(suffixes) + 1}.npy"


def flatten_simulation_results(
    simulation_results: Dict[str, np.ndarray], grid_size: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Flatten the 2D simulation results to 1D for each sample.

    Args:
        simulation_results: Dictionary of simulation results
        grid_size: Size of the simulation grid

    Returns:
        Dictionary of flattened simulation results
    """
    flattened_results = {}
    for transform_name, result_array in simulation_results.items():
        # Shape of result_array is (n_samples, grid_size, grid_size)
        n_samples = result_array.shape[0]
        flattened_results[transform_name] = torch.tensor(
            result_array.reshape(n_samples, grid_size * grid_size), dtype=torch.float32
        )

    return flattened_results


def subsample_data(
    full_transformed_params_norm: Dict[str, np.ndarray],
    full_simulation_results: Dict[str, np.ndarray],
    subsample_percentage: float,
    k_max: int,
    seed: int,
    grid_size: int,
) -> Optional[Dict[str, Union[Dict[str, np.ndarray], Dict[str, any]]]]:
    """
    Subsample the paired point cloud data.

    Args:
        full_transformed_params_norm: Dictionary of original normalized parameters.
        full_simulation_results: Dictionary of simulation results.
        subsample_percentage: Percentage of samples to select (0.0 to 1.0).
        k_max: Maximum k value for k-NN, used to ensure enough samples.
        seed: Random seed for reproducibility.
        grid_size: Size of the simulation grid.

    Returns:
        A dictionary containing subsampled 'transformed_params_norm' and
        'simulation_results', or None if subsample size is too small.
    """
    rng = np.random.default_rng(seed)

    # Assuming all arrays in full_transformed_params_norm['Identity'] and
    # full_simulation_results have the same number of samples (axis 0)
    # Let's pick one to get the total number of samples
    identity_key = "Identity"
    if identity_key not in full_transformed_params_norm:
        print(
            f"Warning: '{identity_key}' not found in full_transformed_params_norm. Skipping subsampling."
        )
        return None

    total_samples = full_transformed_params_norm[identity_key].shape[0]
    num_subsamples = int(total_samples * subsample_percentage)

    min_required_samples = max(k_max + 1, 20)
    if num_subsamples < min_required_samples:
        print(
            f"Warning: Subsample size ({num_subsamples}) is less than "
            f"minimum required ({min_required_samples}) for k_max={k_max}. "
            f"Skipping this bootstrap iteration."
        )
        return None

    indices = rng.choice(total_samples, size=num_subsamples, replace=False)
    indices.sort()  # Keep original order if desired, though not strictly necessary for most metrics

    subsampled_transformed_params_norm = {}
    for key, data_array in full_transformed_params_norm.items():
        if data_array.shape[0] == total_samples:
            subsampled_transformed_params_norm[key] = data_array[indices]
        else:  # Should not happen if data is consistent
            subsampled_transformed_params_norm[key] = data_array

    subsampled_simulation_results = {}
    for key, data_array in full_simulation_results.items():
        if data_array.shape[0] == total_samples:
            # data_array shape is (n_samples, grid_size, grid_size)
            subsampled_simulation_results[key] = data_array[indices, :, :]
        else:  # Should not happen
            subsampled_simulation_results[key] = data_array

    return {
        "transformed_params_norm": subsampled_transformed_params_norm,
        "simulation_results": subsampled_simulation_results,
        "subsample_indices": indices,  # Save the indices for later use
    }


def compute_metrics(
    input_data: Dict,
    output_dir: str,
    device_str: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    k_min: int = 5,
    k_max: int = 10,
    p: float = 2.0,
    alpha: float = 5.0,
    verbose: bool = True,
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute metrics between original and transformed points.

    Args:
        input_data: Dictionary containing simulation data
        output_dir: Output directory for saving metrics
        device_str: Device to use for computations
        metrics: List of metrics to compute
        k_min: Minimum k value for k-NN based metrics
        k_max: Maximum k value for k-NN based metrics
        verbose: Whether to print progress information

    Returns:
        Dictionary of metrics for each transformation, including per-point values
    """
    # Set default metrics if None provided
    # Remove topoae and rtd as they are too expensive
    if metrics is None:
        metrics = ["mle", "beta0", "lip", "ljd", "mst_error"]
    elif "topoae" in metrics or "rtd" in metrics:
        if "topoae" in metrics:
            metrics.remove("topoae")
        if "rtd" in metrics:
            metrics.remove("rtd")
        print(
            "Note: TopoAE and RTD metrics are disabled as they're too computationally expensive"
        )

    # Get device
    device = get_device(device_str)
    if verbose:
        print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the normalized parameters and simulation results
    transformed_params_norm = input_data["transformed_params_norm"]
    simulation_results = input_data["simulation_results"]
    config = input_data["config"]

    # Flatten the simulation results for metric computation
    flattened_results = flatten_simulation_results(
        simulation_results, config["grid_size"]
    )

    # Get original (normalized) points as a tensor
    orig_points = torch.tensor(
        transformed_params_norm["Identity"], dtype=torch.float32, device=device
    )[:, :2]
    print(f"orig_points shape: {orig_points.shape}")

    all_metrics = {}

    print(f"simulation_results keys: {simulation_results.keys()}")

    # Process each transformation
    for transform_name in simulation_results.keys():
        if verbose:
            print(f"\nComputing metrics for {transform_name} transformation:")

        # Get the simulation results for this transformation
        if transform_name in flattened_results:
            result_points = flattened_results[transform_name].to(device)

            dim_input = orig_points.shape[1]
            dim_output = result_points.shape[1]

            # Initialize metric storage with both average and per-point values
            transform_metrics = {}

            # Compute distance matrices once for reuse
            dists_input_p = torch.cdist(orig_points, orig_points, p=p)
            dists_output_p = torch.cdist(result_points, result_points, p=p)

            dists_input_p1 = torch.cdist(orig_points, orig_points, p=1)
            dists_output_p1 = torch.cdist(result_points, result_points, p=1)

            # Get sorted indices for k-NN
            _, input_indices_p = torch.sort(dists_input_p, dim=1)
            _, output_indices_p = torch.sort(dists_output_p, dim=1)

            _, input_indices_p1 = torch.sort(dists_input_p1, dim=1)
            _, output_indices_p1 = torch.sort(dists_output_p1, dim=1)

            mst_pairs_input = None
            mst_pairs_output = None
            mst_edge_weights_input = None
            mst_edge_weights_output = None

            # Compute beta0 (connected components) or MST error if requested
            if "beta0" in metrics or "mst_error" in metrics:
                # Compute MST for input if not already computed
                if mst_pairs_input is None:
                    mst_pairs_input = get_mst_edges(dists_input_p, one_root=True)
                    mst_edge_weights_input = dists_input_p[
                        mst_pairs_input[:, 0], mst_pairs_input[:, 1]
                    ]

                # Compute MST for output if not already computed
                if mst_pairs_output is None:
                    mst_pairs_output = get_mst_edges(dists_output_p, one_root=True)
                    mst_edge_weights_output = dists_output_p[
                        mst_pairs_output[:, 0], mst_pairs_output[:, 1]
                    ]

                # Store MST vertex pairs for visualization
                transform_metrics["MST_input_vertex_pairs"] = (
                    mst_pairs_input.cpu().numpy()
                )
                transform_metrics["MST_output_vertex_pairs"] = (
                    mst_pairs_output.cpu().numpy()
                )

                # Also store the original points and transformed points for easier access in visualization
                transform_metrics["input_points"] = orig_points.cpu().numpy()
                transform_metrics["output_points"] = result_points.cpu().numpy()

            # Compute MLE if requested
            if "mle" in metrics:
                # Calculate MLE for input and output data using already available distance matrices
                n_points = orig_points.shape[0]
                device = dists_input_p1.device

                # Limit k_max if necessary
                if k_max >= n_points:
                    k_max = n_points - 1

                # Remove self-distances (first column) and take only up to k_max
                input_knn_dists = dists_input_p1.gather(
                    1, input_indices_p1[:, 1 : k_max + 1]
                )
                output_knn_dists = dists_output_p1.gather(
                    1, output_indices_p1[:, 1 : k_max + 1]
                )

                # Compute log distances
                input_log_dists = torch.log(input_knn_dists + 1e-6)
                output_log_dists = torch.log(output_knn_dists + 1e-6)

                # Compute MLE estimates for all k values from k_min to k_max
                num_k_vals = k_max - k_min + 1
                k_vals = torch.arange(k_min, k_max + 1, device=device)
                val = 2  # For unbiased estimation

                # MLE calculation for input
                input_mle = (k_vals[None, :] - val) / (
                    (k_vals - 1) * input_log_dists[:, -num_k_vals:]
                    - torch.cumsum(input_log_dists[:, :-1], dim=1)[:, -num_k_vals:]
                )

                # MLE calculation for output
                output_mle = (k_vals[None, :] - val) / (
                    (k_vals - 1) * output_log_dists[:, -num_k_vals:]
                    - torch.cumsum(output_log_dists[:, :-1], dim=1)[:, -num_k_vals:]
                )

                # Store per-point values
                transform_metrics["MLE_original_per_point"] = (
                    input_mle.mean(dim=1).cpu().numpy()
                )
                transform_metrics["MLE_transformed_per_point"] = (
                    output_mle.mean(dim=1).cpu().numpy()
                )

                # Store averages
                transform_metrics["MLE_original"] = input_mle.mean().item()
                transform_metrics["MLE_transformed"] = output_mle.mean().item()

            # Compute Lipschitz constants if requested
            if "lip" in metrics:
                max_fwd_ratios, max_inv_ratios = compute_empirical_lipschitz(
                    orig_points, result_points, p=2.0
                )

                # Store per-point values
                transform_metrics["Forward_Lipschitz_per_point"] = (
                    max_fwd_ratios.cpu().numpy()
                )
                transform_metrics["Inverse_Lipschitz_per_point"] = (
                    max_inv_ratios.cpu().numpy()
                )

                # Store averages
                transform_metrics["Forward_Lipschitz"] = max_fwd_ratios.max().item()
                transform_metrics["Inverse_Lipschitz"] = max_inv_ratios.max().item()
                transform_metrics["Forward_Lipschitz_mean"] = (
                    max_fwd_ratios.mean().item()
                )
                transform_metrics["Inverse_Lipschitz_mean"] = (
                    max_inv_ratios.mean().item()
                )

            # For beta0 and mst_error, use compute_metrics_consolidated
            # as they're not easily computed per-point
            # Compute MST Error if requested
            if "mst_error" in metrics:
                # Get distances for MST edges in each space
                input_mst_dists = mst_edge_weights_input
                output_mst_dists = mst_edge_weights_output

                # Get corresponding distances in the other space
                output_dists_for_input_mst = dists_output_p[
                    mst_pairs_input[:, 0], mst_pairs_input[:, 1]
                ]
                input_dists_for_output_mst = dists_input_p[
                    mst_pairs_output[:, 0], mst_pairs_output[:, 1]
                ]

                rmse_in_mst_in = input_mst_dists / math.sqrt(dim_input)
                rmse_in_mst_ot = input_dists_for_output_mst / math.sqrt(dim_input)
                rmse_ot_mst_in = output_dists_for_input_mst / math.sqrt(dim_output)
                rmse_ot_mst_ot = output_mst_dists / math.sqrt(dim_output)

                sse_rmse_in = (rmse_in_mst_in - rmse_ot_mst_in) ** 2
                sse_rmse_ot = (rmse_ot_mst_ot - rmse_in_mst_ot) ** 2

                sse_rmse_in = sse_rmse_in / (rmse_in_mst_in**2 + 1e-8)
                sse_rmse_ot = sse_rmse_ot / (rmse_ot_mst_ot**2 + 1e-8)

                # Store per-point values
                transform_metrics["MST_error_in_per_point"] = sse_rmse_in.cpu().numpy()
                transform_metrics["MST_error_ot_per_point"] = sse_rmse_ot.cpu().numpy()
                transform_metrics["MST_error_combined_per_point"] = (
                    (sse_rmse_in + sse_rmse_ot).cpu().numpy()
                )
                # Store averages
                transform_metrics["MST_error_in"] = sse_rmse_in.mean().item()
                transform_metrics["MST_error_ot"] = sse_rmse_ot.mean().item()
                transform_metrics["MST_error_combined"] = (
                    sse_rmse_in.mean().item() + sse_rmse_ot.mean().item()
                ) / 2.0

            # Compute beta0 if requested
            if "beta0" in metrics:
                # Make sure to use the estimated dimensionality from the MLE calculation
                # Use MLE estimates from above if available, otherwise use ambient dimension
                if (
                    "MLE_original" not in transform_metrics
                    or "MLE_transformed" not in transform_metrics
                ):
                    raise ValueError("MLE metrics must be computed before beta0")
                d_hat_input = transform_metrics["MLE_original"]
                d_hat_output = transform_metrics["MLE_transformed"]

                # Input space beta0 calculation
                # Calculate lambda using already available kNN distances instead of using estimate_poisson_intensity
                n_points = orig_points.shape[0]

                # Compute volume of unit ball in input dimension in log space
                log_omega_d_input = (d_hat_input / 2) * np.log(np.pi) - special.gammaln(
                    d_hat_input / 2 + 1
                )
                omega_d_input = np.exp(log_omega_d_input)

                # Sort distances to get k-NN distances (excluding self)
                # Remove self-distances (first column)
                k_nearest_dists_input = dists_input_p.gather(
                    1, input_indices_p[:, 1 : k_min + 1]
                )

                # Vectorized approach to calculate lambda for all k values at once
                r_k_sums_input = torch.sum(k_nearest_dists_input**d_hat_input, dim=0)
                lambda_estimates_input = (
                    n_points * torch.arange(1, k_min + 1, device=device)
                ) / (omega_d_input * r_k_sums_input)

                # Use the mean of lambda estimates
                lambda_hat_input = lambda_estimates_input.mean().item()

                # Compute expected 1-NN distance in log space
                log_gamma_term_input = special.gammaln(1 + 1 / d_hat_input)
                gamma_term_input = np.exp(log_gamma_term_input)
                r_expected_input = gamma_term_input / (
                    lambda_hat_input * omega_d_input
                ) ** (1 / d_hat_input)

                # Set connectivity threshold with scaling factor (alpha=3.0)
                connectivity_threshold_input = alpha * r_expected_input

                # B₀ is one plus the number of MST edges longer than 2*threshold
                edge_count_input = torch.sum(
                    mst_edge_weights_input > 2 * connectivity_threshold_input
                ).item()
                beta_zero_input = int(edge_count_input + 1)

                # Compute volume of unit ball in output dimension in log space
                log_omega_d_output = (d_hat_output / 2) * np.log(
                    np.pi
                ) - special.gammaln(d_hat_output / 2 + 1)
                omega_d_output = np.exp(log_omega_d_output)

                # Get k-NN distances for output
                k_nearest_dists_output = dists_output_p.gather(
                    1, output_indices_p[:, 1 : k_min + 1]
                )

                # Calculate lambda for output
                r_k_sums_output = torch.sum(k_nearest_dists_output**d_hat_output, dim=0)
                lambda_estimates_output = (
                    n_points * torch.arange(1, k_min + 1, device=device)
                ) / (omega_d_output * r_k_sums_output)

                # Use the mean of lambda estimates
                lambda_hat_output = lambda_estimates_output.mean().item()

                # Compute expected 1-NN distance in log space
                log_gamma_term_output = special.gammaln(1 + 1 / d_hat_output)
                gamma_term_output = np.exp(log_gamma_term_output)
                r_expected_output = gamma_term_output / (
                    lambda_hat_output * omega_d_output
                ) ** (1 / d_hat_output)

                # Set connectivity threshold with scaling factor (alpha=3.0)
                connectivity_threshold_output = 3.0 * r_expected_output

                # B₀ is one plus the number of MST edges longer than 2*threshold
                edge_count_output = torch.sum(
                    mst_edge_weights_output > 2 * connectivity_threshold_output
                ).item()
                beta_zero_output = int(edge_count_output + 1)

                # Store results
                transform_metrics["B0_original"] = beta_zero_input
                transform_metrics["B0_transformed"] = beta_zero_output

            # Compute LJD if requested - this returns per-point values
            if "ljd" in metrics:

                # Get k-NN indices for Jacobian calculation (excluding self)
                k_jac = min(5, orig_points.shape[0] // 10)
                domain_indices = input_indices_p[:, 1 : k_jac + 1]
                codomain_indices = output_indices_p[:, 1 : k_jac + 1]

                # Calculate Jacobian linearization errors
                domain_errors, codomain_errors = estimate_jacobian_delta_batched(
                    orig_points,
                    result_points,
                    domain_indices,
                    codomain_indices,
                    batch_size=128,
                    rcond=1e-5,
                    p=2.0,
                    pca_project=(orig_points.shape[1] != result_points.shape[1]),
                )

                # Store per-point values
                transform_metrics["LJD_domain_per_point"] = domain_errors.cpu().numpy()
                transform_metrics["LJD_codomain_per_point"] = (
                    codomain_errors.cpu().numpy()
                )
                transform_metrics["LJD_combined_per_point"] = (
                    0.5 * (domain_errors + codomain_errors).cpu().numpy()
                )

                # Store averages
                transform_metrics["LJD_domain"] = domain_errors.mean().item()
                transform_metrics["LJD_codomain"] = codomain_errors.mean().item()
                transform_metrics["LJD_combined"] = 0.5 * (
                    domain_errors.mean().item() + codomain_errors.mean().item()
                )

            all_metrics[transform_name] = transform_metrics
            if verbose:
                print(f"  {transform_name} metrics (averages):")
                for metric_name, value in transform_metrics.items():
                    if (
                        (not metric_name.endswith("_per_point"))
                        and (not metric_name.endswith("vertex_pairs"))
                        and (not metric_name.startswith("input_points"))
                        and (not metric_name.startswith("output_points"))
                    ):
                        print(f"    {metric_name}: {value}")
        else:
            if verbose:
                print(f"  No simulation results found for {transform_name}")

    return all_metrics


def save_metrics(
    bootstrapped_metrics_list: List[Dict[str, Dict[str, Union[float, np.ndarray]]]],
    input_data_path: str,
    input_data: Dict,  # Full original input_data for metadata
    output_dir: str,
    args: argparse.Namespace,
) -> str:
    """
    Save bootstrapped metrics data to a file.

    Args:
        bootstrapped_metrics_list: List of metrics dictionaries, one for each bootstrap.
        input_data_path: Path to the input data file.
        input_data: Dictionary containing the original full simulation data.
        output_dir: Output directory for saving metrics.
        args: Command-line arguments.

    Returns:
        Path to the saved metrics file.
    """
    # Generate metrics filename
    metrics_filename = get_next_metrics_filename(input_data_path)

    # Prepare save dictionary
    save_dict = {
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "bootstrapped_metrics": bootstrapped_metrics_list,
        "source_data_path": input_data_path,
        "source_timestamp": input_data.get("timestamp", "unknown"),
        "command_args": vars(args),
        "original_config": input_data.get("config", {}),
        "num_bootstraps": args.num_bootstraps,
        "subsample_percentage": args.subsample_percentage,
    }

    # Save metrics data
    np.save(metrics_filename, save_dict)

    return metrics_filename


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Compute metrics for Cahn-Hilliard simulation results",
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
        default="ch_manifold_results",
        help="Directory for saving metrics results",
    )

    parser.add_argument(
        "--k_min",
        type=int,
        default=5,
        help="Minimum k value for k-NN based metrics",
    )

    parser.add_argument(
        "--k_max",
        type=int,
        default=10,
        help="Maximum k value for k-NN based metrics",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=2.0,
        help="p-norm to use for distance calculations",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="Scaling factor for connectivity threshold in beta0 calculation",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metrics to compute. If not provided, computes all metrics",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "xpu", "mps"],
        default=None,
        help="Device to use for computation (default: auto-detect)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information",
    )

    parser.add_argument(
        "--num_bootstraps",
        type=int,
        default=20,
        help="Number of bootstrap samples to generate.",
    )

    parser.add_argument(
        "--subsample_percentage",
        type=float,
        default=0.5,
        help="Percentage of data to subsample for each bootstrap (0.0 to 1.0).",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to compute metrics from simulation results.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Determine the input file
    input_file = args.input_file
    if input_file is None:
        try:
            input_file = get_latest_simulation_file(args.output_dir)
            if args.verbose:
                print(f"Using most recent simulation file: {input_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify an input file with --input_file")
            return

    # Load the simulation data
    if args.verbose:
        print(f"Loading data from: {input_file}")

    input_data = np.load(input_file, allow_pickle=True).item()

    # Extract full data for subsampling
    full_transformed_params_norm = input_data["transformed_params_norm"]
    full_simulation_results = input_data["simulation_results"]
    original_config = input_data["config"]
    grid_size = original_config.get("grid_size", 128)  # Get grid_size from config

    all_bootstrap_metrics = []
    base_seed = np.random.randint(0, 2**32 - 1)  # Generate a base seed for this run

    if args.verbose:
        print(
            f"Starting {args.num_bootstraps} bootstrap iterations with {args.subsample_percentage*100:.2f}% subsampling."
        )

    for i in range(args.num_bootstraps):
        current_seed = base_seed + i
        if args.verbose:
            print(
                f"\nBootstrap iteration {i+1}/{args.num_bootstraps} (seed: {current_seed})"
            )

        subsampled_data_dict = subsample_data(
            full_transformed_params_norm=full_transformed_params_norm,
            full_simulation_results=full_simulation_results,
            subsample_percentage=args.subsample_percentage,
            k_max=args.k_max,
            seed=current_seed,
            grid_size=grid_size,
        )

        if subsampled_data_dict:
            # Construct a temporary input_data-like dictionary for compute_metrics
            current_input_for_metrics = {
                "transformed_params_norm": subsampled_data_dict[
                    "transformed_params_norm"
                ],
                "simulation_results": subsampled_data_dict["simulation_results"],
                "config": original_config,  # Use original config, grid_size is important here
                # Add other necessary keys from original input_data if compute_metrics needs them
                # For now, assuming 'config' is the main one besides the data itself.
            }

            # Save the subsample indices for later use in visualizations
            bootstrap_indices = subsampled_data_dict.get("subsample_indices", None)

            metrics = compute_metrics(
                input_data=current_input_for_metrics,
                output_dir=args.output_dir,  # output_dir is for compute_metrics internal use, not file saving here
                device_str=args.device,
                metrics=args.metrics,
                k_min=args.k_min,
                k_max=args.k_max,
                p=args.p,
                alpha=args.alpha,
                verbose=args.verbose,  # Control verbosity of compute_metrics
            )

            # Store bootstrap indices with metrics
            if bootstrap_indices is not None:
                for transform_name, transform_metrics in metrics.items():
                    transform_metrics["subsample_indices"] = bootstrap_indices

            all_bootstrap_metrics.append(metrics)
        else:
            if args.verbose:
                print(
                    f"Skipped metrics computation for bootstrap iteration {i+1} due to insufficient sample size."
                )

    if not all_bootstrap_metrics:
        print("No metrics were computed from any bootstrap iteration. Exiting.")
        return

    # Save all collected bootstrap metrics
    metrics_file = save_metrics(
        bootstrapped_metrics_list=all_bootstrap_metrics,
        input_data_path=input_file,
        input_data=input_data,  # Pass the original full input_data for metadata
        output_dir=args.output_dir,
        args=args,
    )

    if args.verbose:
        print(f"\nMetrics saved to: {metrics_file}")
        print("Done!")


if __name__ == "__main__":
    main()
