import numpy as np
import pandas as pd
import os
import glob
import argparse
from typing import Dict
from datetime import datetime

# Import metric labels from CH_manifolds_fig.py for consistency
from CH_manifolds_fig import METRIC_LABELS, TRANSFORM_LABELS


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


def transform_metrics_to_dataframe(
    metrics_data: Dict, bootstrap_sample_pct: float = 90, bootstrap_iterations: int = 10
) -> pd.DataFrame:
    """
    Transform metrics dictionary into a pandas DataFrame with statistics from per-point data.
    Uses bootstrap resampling to estimate confidence intervals.

    Args:
        metrics_data: Dictionary containing metrics data
        bootstrap_sample_pct: Percentage of data to use in each bootstrap sample (default: 90%)
        bootstrap_iterations: Number of bootstrap iterations to perform (default: 10)

    Returns:
        DataFrame with all metrics, means, and bootstrap-based confidence intervals
    """
    # Get the metrics from the data
    metrics = metrics_data["metrics"]

    # Create a list to hold rows for the DataFrame
    rows = []

    # For each transformation, extract metrics
    for transform_name, transform_metrics in metrics.items():
        # Skip metadata entries
        if not isinstance(transform_metrics, dict):
            continue

        # Create a row with transformation name
        row = {"transformation": transform_name}

        # First add non-per-point metrics (scalar values) directly
        for metric_name, metric_value in transform_metrics.items():
            # Skip per-point data and other large arrays
            if (
                not metric_name.endswith("_per_point")
                and not metric_name.endswith("_vertex_pairs")
                and not metric_name.endswith("_points")
            ):
                # Add scalar metrics directly to the row
                row[metric_name] = metric_value

        # Then process metrics that have per-point values to calculate bootstrap statistics
        for metric_name, metric_value in transform_metrics.items():
            if metric_name.endswith("_per_point"):
                base_metric_name = metric_name.replace("_per_point", "")

                # Skip if no values
                if metric_value is None or len(metric_value) == 0:
                    continue

                # Clean the data - remove NaN values
                per_point_values = np.array(metric_value)
                per_point_values = per_point_values[~np.isnan(per_point_values)]

                if len(per_point_values) == 0:
                    continue

                # Calculate true mean over all points
                mean = np.mean(per_point_values)

                # Perform bootstrap resampling
                bootstrap_means = []
                n_samples = len(per_point_values)
                sample_size = int(
                    n_samples * bootstrap_sample_pct / 100
                )  # Convert percentage to count

                for _ in range(bootstrap_iterations):
                    # Random sampling with replacement
                    indices = np.random.choice(n_samples, sample_size, replace=True)
                    bootstrap_sample = per_point_values[indices]
                    bootstrap_means.append(np.mean(bootstrap_sample))

                # Calculate bootstrap statistics
                bootstrap_means = np.array(bootstrap_means)
                bootstrap_std = np.std(bootstrap_means)

                # Store statistics
                row[f"{base_metric_name}_mean"] = mean
                row[f"{base_metric_name}_bootstrap_std"] = bootstrap_std

                # Store range for confidence interval (mean ± 2*std for ~95% CI)
                row[f"{base_metric_name}_ci_lower"] = mean - 2 * bootstrap_std
                row[f"{base_metric_name}_ci_upper"] = mean + 2 * bootstrap_std

        # Add the row if it has calculated statistics
        if len(row) > 1:  # More than just the transformation name
            rows.append(row)

    # Convert to DataFrame
    return pd.DataFrame(rows)


def format_num(value: float, sig_figs: int = 3) -> str:
    """
    Format a number with specified significant figures.

    Args:
        value: Number to format
        sig_figs: Number of significant figures

    Returns:
        Formatted number string
    """
    if np.isnan(value):
        return "N/A"

    # Use scientific notation for very small or large numbers
    if abs(value) < 0.01 or abs(value) >= 10000:
        return f"{{:.{sig_figs}g}}".format(value)
    else:
        # Calculate number of decimal places needed
        if abs(value) >= 1:
            decimal_places = max(0, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)
        else:
            # For values < 1, keep sig_figs decimal places
            decimal_places = sig_figs

        return f"{{:.{decimal_places}f}}".format(value)


def format_mean_ci(
    mean: float, ci_lower: float, ci_upper: float, sig_figs: int = 3
) -> str:
    """
    Format mean value with confidence interval in LaTeX format.

    Args:
        mean: Mean value
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        sig_figs: Number of significant figures

    Returns:
        Formatted string with mean and CI
    """
    if np.isnan(mean) or np.isnan(ci_lower) or np.isnan(ci_upper):
        return "N/A"

    # Calculate CI half-width
    ci_half = max(mean - ci_lower, ci_upper - mean)

    # For scientific notation (very large or very small numbers)
    if abs(mean) < 0.01 or abs(mean) >= 10000:
        # Format mean in scientific notation
        if abs(mean) >= 10000:
            # For large numbers, use scientific notation with proper LaTeX
            power = int(np.floor(np.log10(abs(mean))))
            mantissa = mean / (10**power)
            mantissa_str = format_num(mantissa, sig_figs)

            # Format CI values relative to mantissa
            ci_lower_mantissa = ci_lower / (10**power)
            ci_upper_mantissa = ci_upper / (10**power)

            ci_lower_rel = mantissa - ci_lower_mantissa
            ci_upper_rel = ci_upper_mantissa - mantissa

            # Use proper LaTeX with error term scaled to same power
            ci_scaled = ci_half / (10**power)
            ci_str = format_num(ci_scaled, 2)

            # Return properly formatted scientific notation with CI
            return f"${mantissa_str}\\times 10^{{{power}}}$ $\\pm$ ${ci_str}\\times 10^{{{power}}}$"
        else:
            # For small numbers (< 0.01)
            mean_str = format_num(mean, sig_figs)
            ci_str = format_num(ci_half, 2)
            return f"${mean_str}$ $\\pm$ ${ci_str}$"

    # For regular numbers, use standard format
    mean_str = format_num(mean, sig_figs)

    # Determine decimal places for CI
    if abs(mean) >= 1:
        decimal_places = max(0, sig_figs - int(np.floor(np.log10(abs(mean)))) - 1)
    else:
        decimal_places = sig_figs

    # Format CI with consistent decimal places
    ci_str = f"{{:.{decimal_places}f}}".format(ci_half)

    return f"{mean_str} $\\pm$ {ci_str}"


def latex_safe_name(name: str) -> str:
    """
    Make a string safe for LaTeX by escaping special characters.

    Args:
        name: Original string

    Returns:
        LaTeX-safe string
    """
    # Use the transform labels if available
    if name in TRANSFORM_LABELS:
        return TRANSFORM_LABELS[name]

    # Otherwise escape any special characters
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name


def format_latex_table(metrics_df: pd.DataFrame) -> str:
    """
    Format metrics DataFrame into two LaTeX tables:
    1. Non-confidence interval metrics (MLE and B0)
    2. Metrics with 95% confidence intervals

    Args:
        metrics_df: DataFrame with metrics including mean and CI values

    Returns:
        LaTeX table code as string
    """
    # Create a natural ordering of transformations
    natural_order = {
        "Identity": 0,
        "Swiss_Roll": 1,
        "Waveform": 2,
        "Ribbon": 3,
        "Cylinder": 4,
        "Split": 5,
        "Hole": 6,
        "Pinch": 7,
        "Collapse": 8,
    }

    # Sort the DataFrame by the natural ordering
    metrics_df["sort_order"] = metrics_df["transformation"].map(
        lambda x: natural_order.get(x, 999)
    )
    metrics_df = metrics_df.sort_values("sort_order").drop("sort_order", axis=1)

    # Start the table environment
    latex_output = "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\n"

    # Separate metrics into two categories
    non_ci_metrics = [
        "MLE_original",
        "MLE_transformed",
        "B0_original",
        "B0_transformed",
    ]
    ci_metrics = []

    # List all metrics that have per-point values (for confidence intervals)
    metric_base_names = set()
    for col in metrics_df.columns:
        if col.endswith("_mean"):
            base_name = col.replace("_mean", "")
            if base_name not in non_ci_metrics:
                metric_base_names.add(base_name)
                ci_metrics.append(base_name)

    # Get original values (should be the same across all transformations)
    # Just use the first row since original values should be the same for all transforms
    first_row = metrics_df.iloc[0]
    orig_mle = first_row.get("MLE_original", np.nan)
    orig_b0 = first_row.get("B0_original", np.nan)

    # ----------------------------------------------------------------------
    # First table: Non-confidence interval metrics (MLE and B0)
    # ----------------------------------------------------------------------
    latex_output += "\\begin{tabular}{lccc}\n\\toprule\n"
    latex_output += "F & $\\hat{d}$ & $\\beta_0$ \\\\\n\\midrule\n"

    # Show original values in their own row
    latex_output += "Original & "
    # Format with appropriate precision
    if not np.isnan(orig_mle):
        formatted_mle = format_num(orig_mle, 3)
        latex_output += f"{formatted_mle} & "
    else:
        latex_output += "N/A & "

    if not np.isnan(orig_b0):
        formatted_b0 = format_num(orig_b0, 0)
        latex_output += f"{formatted_b0}"
    else:
        latex_output += "N/A"

    latex_output += " \\\\\n\\midrule\n"

    # Data rows for first table (transformed values only)
    for _, row in metrics_df.iterrows():
        transform = row["transformation"]
        transform_label = TRANSFORM_LABELS.get(transform, transform)

        # Just show F_n instead of the full transformation name
        if transform_label.startswith("$F_"):
            transform_label = transform_label

        latex_output += f"{transform_label}"

        # Add transformed metrics only
        mle_transformed = row.get("MLE_transformed", np.nan)
        b0_transformed = row.get("B0_transformed", np.nan)

        # Format with appropriate precision
        if not np.isnan(mle_transformed):
            formatted_mle = format_num(mle_transformed, 3)
            latex_output += f" & {formatted_mle}"
        else:
            latex_output += " & N/A"

        if not np.isnan(b0_transformed):
            formatted_b0 = format_num(b0_transformed, 0)
            latex_output += f" & {formatted_b0}"
        else:
            latex_output += " & N/A"

        latex_output += " \\\\\n"

    # Close first table
    latex_output += "\\bottomrule\n\\end{tabular}\n\n"

    latex_output += "\\vspace{3mm}\n\n"  # Add space between tables

    # ----------------------------------------------------------------------
    # Second table: Confidence interval metrics
    # ----------------------------------------------------------------------

    # Determine how many columns we need
    num_ci_metrics = len(ci_metrics)
    col_spec = "l" + "c" * num_ci_metrics

    latex_output += f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"

    # Get the proper LaTeX labels for each metric
    latex_metric_labels = {}
    for metric in ci_metrics:
        latex_metric_labels[metric] = METRIC_LABELS.get(
            metric, metric.replace("_", " ")
        )

    # Header row with metric names
    latex_output += "F"
    for metric in sorted(ci_metrics):
        latex_output += f" & {latex_metric_labels[metric]}"
    latex_output += " \\\\\n\\midrule\n"

    # Data rows for second table
    for _, row in metrics_df.iterrows():
        transform = row["transformation"]
        transform_label = TRANSFORM_LABELS.get(transform, transform)

        # Just show F_n instead of the full transformation name
        if transform_label.startswith("$F_"):
            transform_label = transform_label

        latex_output += f"{transform_label}"

        # Add each CI metric
        for metric in sorted(ci_metrics):
            mean = row.get(f"{metric}_mean", np.nan)
            ci_lower = row.get(f"{metric}_ci_lower", np.nan)
            ci_upper = row.get(f"{metric}_ci_upper", np.nan)

            # For values around 0.001, ensure at least 2 significant figures
            sig_figs = 3
            if mean is not None and 0.0001 < abs(mean) < 0.01:
                sig_figs = 4  # Use more significant figures for small values

            # Format with appropriate precision and confidence intervals
            formatted_value = format_mean_ci(mean, ci_lower, ci_upper, sig_figs)
            latex_output += f" & {formatted_value}"

        latex_output += " \\\\\n"

    # Close the second table
    latex_output += "\\bottomrule\n\\end{tabular}\n\n"

    # Add a caption and label
    latex_output += "\\caption{\\textbf{Cahn-Hilliard Manifold Analysis with Bootstrap Resampling.} "
    latex_output += "The top table shows intrinsic dimensionality estimates ($\\hat{d}$) and the number of connected components ($\\beta_0$) "
    latex_output += (
        "for the original point cloud (first row) and each transformed point cloud. "
    )
    latex_output += "The bottom table presents continuity and distortion metrics with 95\\% confidence intervals derived from bootstrap resampling. "
    latex_output += "Metrics include forward and inverse Lipschitz constants, "
    latex_output += "minimum spanning tree distortion (MST), and Local Jacobian Distortion (LJD).}\n"
    latex_output += "\\label{tab:ch_manifolds_bootstrap}\n"
    latex_output += "\\end{table}\n"

    return latex_output


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from CH manifold metrics using bootstrap resampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--metrics_file",
        type=str,
        default=None,
        help="Path to metrics file. If not provided, uses most recent file.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="tables",
        help="Folder to save output files",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ch_manifolds_bootstrap.tex",
        help="Output filename for LaTeX table",
    )

    parser.add_argument(
        "--csv_output",
        type=str,
        default="ch_manifolds_bootstrap.csv",
        help="Output filename for CSV data",
    )

    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to output filenames",
    )

    parser.add_argument(
        "--bootstrap_pct",
        type=float,
        default=90.0,
        help="Percentage of data points to sample in each bootstrap iteration (0-100)",
    )

    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=10,
        help="Number of bootstrap iterations to perform",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to generate LaTeX tables from metrics.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Add timestamp to filenames if requested
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(args.output)
        args.output = f"{base_name}_{timestamp}{ext}"

        base_name, ext = os.path.splitext(args.csv_output)
        args.csv_output = f"{base_name}_{timestamp}{ext}"

    # Determine metrics file
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

    # Transform metrics to DataFrame
    metrics_df = transform_metrics_to_dataframe(
        metrics_data, args.bootstrap_pct, args.bootstrap_iterations
    )

    # Format LaTeX table
    print("Generating LaTeX table with 95% confidence intervals...")
    latex_code = format_latex_table(metrics_df)

    # Save results
    os.makedirs(args.output_folder, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(args.output_folder, args.csv_output)
    metrics_df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # Save LaTeX
    tex_path = os.path.join(args.output_folder, args.output)
    with open(tex_path, "w") as f:
        f.write(latex_code)
    print(f"LaTeX table saved to: {tex_path}")


if __name__ == "__main__":
    main()
