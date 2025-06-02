"""
Generate tables from Cahn-Hilliard simulation metrics.
"""

import numpy as np
import pandas as pd
import os
import glob
import argparse
from typing import Dict
from datetime import datetime

# Define metric name mapping for better labels
METRIC_LABELS = {
    "MLE_original": r"$\hat{d}_{\text{original}}$",
    "MLE_transformed": r"$\hat{d}$",  # Simplified label
    "B0_original": r"$\beta_0^{\text{original}}$",
    "B0_transformed": r"$\beta_0$",  # Simplified label
    "Forward_Lipschitz": r"$\text{Lip}(f)$",
    "Forward_Lipschitz_mean": r"$\text{Mean Lip}(f)$",  # Keep mean distinct if needed, or remove if only max is shown
    "Inverse_Lipschitz": r"$\text{Lip}(f^{-1})$",
    "Inverse_Lipschitz_mean": r"$\text{Mean Lip}(f^{-1})$",  # Keep mean distinct if needed
    "LJD_domain": r"$\text{LJD}_{\text{domain}}$",
    "LJD_codomain": r"$\text{LJD}_{\text{codomain}}$",
    "LJD_combined": r"$\text{LJD}$",  # Simplified label
    "MST_error_in": r"$\text{MST}_{\text{in}}$",  # Adjusted based on likely output from metrics script
    "MST_error_ot": r"$\text{MST}_{\text{ot}}$",  # Adjusted based on likely output from metrics script
    "MST_error_combined": r"$\text{MST}$",  # Simplified label
    # "TopoAE": r"$\text{TopoAE}$", # Likely not computed
    # "RTD": r"$\text{RTD}$", # Likely not computed
}

# Define transformation name mapping to match fig_points_manifolds.py
TRANSFORM_LABELS = {
    "Identity": r"$F_0$",
    "Swiss_Roll": r"$F_1$",
    "Dog_Ear": r"$F_2$",  # Restored Dog_Ear instead of Waveform
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


def transform_metrics_to_dataframe(metrics_data: Dict) -> pd.DataFrame:
    """
    Transform metrics dictionary into a pandas DataFrame with statistics from per-point data.

    Args:
        metrics_data: Dictionary containing metrics data

    Returns:
        DataFrame with all metrics, means, and confidence intervals
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

        # Then process metrics that have per-point values to calculate confidence intervals
        for metric_name, metric_value in transform_metrics.items():
            if metric_name.endswith("_per_point"):
                base_metric_name = metric_name.replace("_per_point", "")

                # Skip if no values
                if metric_value is None or len(metric_value) == 0:
                    continue

                # Compute statistics directly from all per-point values
                per_point_values = np.array(metric_value)
                per_point_values = per_point_values[~np.isnan(per_point_values)]

                if len(per_point_values) == 0:
                    continue

                # Calculate mean, std, and confidence intervals
                mean = np.mean(per_point_values)
                std = np.std(per_point_values)

                # 95% confidence interval based on standard error
                n = len(per_point_values)
                se = std / np.sqrt(n)
                ci_lower = mean - 1.96 * se
                ci_upper = mean + 1.96 * se

                # Store statistics
                row[f"{base_metric_name}_mean"] = mean
                row[f"{base_metric_name}_std"] = std
                row[f"{base_metric_name}_ci_lower"] = ci_lower
                row[f"{base_metric_name}_ci_upper"] = ci_upper

        # Add the row if it has calculated statistics
        if len(row) > 1:  # More than just the transformation name
            rows.append(row)

    # Convert to DataFrame
    return pd.DataFrame(rows)


def format_num(value: float, sig_figs: int = 3) -> str:
    """
    Format a number with specified significant figures using 'g' format.
    For B0 values, always display as full integers.

    Args:
        value: Number to format
        sig_figs: Number of significant figures

    Returns:
        Formatted number string
    """
    if np.isnan(value):
        return "N/A"

    # Special case for B0 values - always display as full integers
    if isinstance(value, (int, np.integer)) or (abs(value - round(value)) < 1e-10):
        return f"{int(round(value))}"

    # Use 'g' format for significant figures, handles sci notation automatically
    return f"{{:.{sig_figs}g}}".format(value)


def format_mean_ci(
    mean: float, ci_lower: float, ci_upper: float, error_sig_figs: int = 2
) -> str:
    """
    Format mean value with confidence interval in LaTeX format.
    Formats the mean based on the precision of the CI half-width.

    Args:
        mean: Mean value
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        error_sig_figs: Number of significant figures for the error term (CI half-width)

    Returns:
        Formatted string with mean and CI (e.g., "1.23 $\\pm$ 0.04")
    """
    if np.isnan(mean) or np.isnan(ci_lower) or np.isnan(ci_upper):
        return "N/A"

    ci_half = max(mean - ci_lower, ci_upper - mean)

    # Handle zero or very small error case separately
    if ci_half <= 1e-15:  # Effectively zero error
        # Format mean based on its own magnitude if error is zero
        return format_num(
            mean, error_sig_figs + 1
        )  # Use slightly more sig figs for mean itself

    # Determine decimal places based on the error's magnitude and desired sig figs
    # Example: ci_half=0.0123, error_sig_figs=2 -> floor(log10)= -2. error_decimals = max(0, -(-2) + (2-1)) = 3. Formats to 0.012
    # Example: ci_half=123.4, error_sig_figs=2 -> floor(log10)= 2. error_decimals = max(0, -(2) + (2-1)) = -1 -> 0. Formats to 120 (incorrect, need rounding logic)

    # Alternative: Round error to specified sig figs, then find decimal places
    with np.errstate(
        invalid="ignore"
    ):  # Ignore warnings for log10(0) if ci_half is tiny
        exponent = np.floor(np.log10(ci_half))
    factor = 10 ** (exponent - (error_sig_figs - 1))
    rounded_ci_half = np.round(ci_half / factor) * factor

    # Determine decimal places from the rounded error
    if rounded_ci_half == 0:  # Check again after rounding
        return format_num(mean, error_sig_figs + 1)

    with np.errstate(invalid="ignore"):
        error_decimals = max(
            0, -int(np.floor(np.log10(rounded_ci_half))) + (error_sig_figs - 1)
        )
        # Handle cases where error rounds to integer, e.g. 0.98 -> 1.0
        if rounded_ci_half >= 1 and np.floor(np.log10(rounded_ci_half)) >= (
            error_sig_figs - 1
        ):
            error_decimals = 0

    # Format mean and error to these decimal places
    mean_str = f"{{:.{error_decimals}f}}".format(mean)
    ci_str = f"{{:.{error_decimals}f}}".format(
        rounded_ci_half
    )  # Use rounded error for display

    # Check if scientific notation is more appropriate for the mean
    # Use scientific notation if mean is very small or large compared to its precision
    if abs(mean) > 0 and (
        abs(mean) < 10 ** (-error_decimals - 1) or abs(mean) >= 10000
    ):
        # Reformat in scientific notation using 'g' format for simplicity
        mean_sci_str = format_num(mean, error_sig_figs + 1)
        ci_sci_str = format_num(rounded_ci_half, error_sig_figs)
        return f"${mean_sci_str} \\pm {ci_sci_str}$"  # Use general sci format

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
    Format metrics DataFrame into two LaTeX tables with baseline comparison.
    Values showing the largest relative deviation from the Identity baseline are bolded,
    but only for specific metrics and non-homeomorphic transformations.

    Args:
        metrics_df: DataFrame with metrics including mean and CI values

    Returns:
        LaTeX table code as string
    """
    # Create a natural ordering of transformations
    natural_order = {
        "Identity": 0,
        "Swiss_Roll": 1,
        "Dog_Ear": 2,  # Updated from Waveform to Dog_Ear
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

    # Get the Identity row for baseline comparison
    try:
        identity_row = metrics_df[metrics_df["transformation"] == "Identity"].iloc[0]
    except IndexError:
        print(
            "Warning: 'Identity' transformation not found in metrics data. Cannot perform baseline comparison."
        )
        identity_row = None
        max_deviations = {}

    # Define specific metrics to consider for bolding
    metrics_to_bold = [
        "Forward_Lipschitz",  # Max metric
        "Inverse_Lipschitz",  # Max metric
        "LJD_domain",  # Mean metric
        "LJD_codomain",  # Mean metric
        "MST_error_in",  # Mean metric
        "MST_error_ot",  # Mean metric
    ]

    if identity_row is not None:
        # --- Calculate Deviations, tracking the max deviation per transformation ---
        transform_max_deviations = {}  # transform_name -> {metric_name: deviation}
        non_homeo_transforms = [
            t for t in natural_order.keys() if t not in ["Identity", "Swiss_Roll"]
        ]

        # Initialize tracking dictionaries
        for transform in non_homeo_transforms:
            transform_max_deviations[transform] = {"metric": "", "deviation": -1}

        # Find the metric with max deviation for EACH transformation
        for idx, row in metrics_df.iterrows():
            transform = row["transformation"]

            # Skip Identity and Swiss Roll
            if transform in ["Identity", "Swiss_Roll"]:
                continue

            # Process each of the 6 specific metrics
            for metric in metrics_to_bold:
                # Determine the actual column name (could be base or with _mean)
                current_val_col = metric
                identity_val_col = metric

                # Check if we need to use mean values
                if (
                    f"{metric}_mean" in row.index
                    and "Forward_Lipschitz" not in metric
                    and "Inverse_Lipschitz" not in metric
                ):
                    current_val_col = f"{metric}_mean"

                if (
                    f"{metric}_mean" in identity_row.index
                    and "Forward_Lipschitz" not in metric
                    and "Inverse_Lipschitz" not in metric
                ):
                    identity_val_col = f"{metric}_mean"

                # Skip if columns don't exist
                if (
                    current_val_col not in row.index
                    or identity_val_col not in identity_row.index
                ):
                    continue

                current_val = row.get(current_val_col, np.nan)
                identity_val = identity_row.get(identity_val_col, np.nan)

                if (
                    pd.isna(current_val)
                    or pd.isna(identity_val)
                    or identity_val == 0
                    or abs(identity_val) < 1e-9
                ):
                    deviation = 0  # Cannot compare if baseline is zero or NaN
                else:
                    normalized_val = current_val / identity_val
                    deviation = abs(normalized_val - 1.0)

                # Update max deviation for THIS TRANSFORMATION if this one is larger
                if deviation > transform_max_deviations[transform]["deviation"]:
                    transform_max_deviations[transform]["deviation"] = deviation
                    transform_max_deviations[transform]["metric"] = metric
    else:  # No identity row found
        transform_max_deviations = {}  # Ensure it exists but is empty

    # --- Generate LaTeX Table ---
    latex_output = "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\n"

    # Identify non-CI and CI metrics again for table structure
    non_ci_metrics_display = [  # Metrics for the first table
        "MLE_transformed",
        "B0_transformed",
    ]

    # Only include the 6 specified metrics in the second table
    ci_metrics_base_names = [
        m for m in metrics_to_bold if m not in non_ci_metrics_display
    ]

    # --- First table: Non-CI metrics ---
    latex_output += "\\begin{tabular}{lccc}\n\\toprule\n"
    # Use simplified labels from METRIC_LABELS
    mle_label = METRIC_LABELS.get("MLE_transformed", "$\\hat{d}$")
    b0_label = METRIC_LABELS.get("B0_transformed", "$\\beta_0$")
    latex_output += f"F & {mle_label} & {b0_label} \\\\\n\\midrule\n"

    # Original row (no bolding) - Display original values from identity_row if available
    latex_output += "Original & "
    if identity_row is not None:
        orig_mle = identity_row.get("MLE_original", np.nan)
        orig_b0 = identity_row.get("B0_original", np.nan)
        latex_output += f"{format_num(orig_mle, 3)} & "
        latex_output += f"{format_num(orig_b0, 0)}"
    else:
        latex_output += "N/A & N/A"
    latex_output += " \\\\\n\\midrule\n"

    # Data rows for all transformations (including Identity and Swiss_Roll)
    for idx, row in metrics_df.iterrows():
        transform = row["transformation"]
        transform_label = TRANSFORM_LABELS.get(transform, transform)
        latex_output += f"{transform_label}"

        # Add transformed metrics only
        for metric_base_name in non_ci_metrics_display:
            value = row.get(metric_base_name, np.nan)
            sig_figs = (
                3 if "MLE" in metric_base_name else 0
            )  # B0 is now always displayed as integer
            formatted_value = format_num(value, sig_figs)

            latex_output += f" & {formatted_value}"

        latex_output += " \\\\\n"

    latex_output += "\\bottomrule\n\\end{tabular}\n\n"
    latex_output += "\\vspace{3mm}\n\n"

    # --- Second table: CI metrics (only the 6 specified ones) ---
    if ci_metrics_base_names:
        num_ci_metrics = len(ci_metrics_base_names)
        col_spec = "l" + "c" * num_ci_metrics
        latex_output += f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"

        # Header row using METRIC_LABELS
        latex_output += "F"
        for metric_base_name in ci_metrics_base_names:
            label = METRIC_LABELS.get(
                metric_base_name, metric_base_name.replace("_", " ")
            )
            latex_output += f" & {label}"
        latex_output += " \\\\\n\\midrule\n"

        # Data rows for all transformations
        for idx, row in metrics_df.iterrows():
            transform = row["transformation"]
            transform_label = TRANSFORM_LABELS.get(transform, transform)
            latex_output += f"{transform_label}"

            # Add each CI metric
            for metric_base_name in ci_metrics_base_names:
                # Determine if we should use mean value or just the base value
                is_mean_metric = metric_base_name not in [
                    "Forward_Lipschitz",
                    "Inverse_Lipschitz",
                ]

                if is_mean_metric:
                    mean = row.get(f"{metric_base_name}_mean", np.nan)
                    ci_lower = row.get(f"{metric_base_name}_ci_lower", np.nan)
                    ci_upper = row.get(f"{metric_base_name}_ci_upper", np.nan)
                    formatted_value = format_mean_ci(
                        mean, ci_lower, ci_upper, error_sig_figs=2
                    )
                else:
                    # For Lipschitz constants, just use the main value without CI
                    value = row.get(metric_base_name, np.nan)
                    formatted_value = format_num(value, 3)

                # Only bold if:
                # 1. This is the metric with max deviation for this transformation
                # 2. This is not the Identity or Swiss_Roll transform
                should_bold = False
                if (
                    transform not in ["Identity", "Swiss_Roll"]
                    and transform in transform_max_deviations
                ):
                    # Bold if this is the metric with max deviation for this transform
                    if (
                        transform_max_deviations[transform]["metric"]
                        == metric_base_name
                    ):
                        should_bold = True

                latex_output += " & "
                if should_bold:
                    latex_output += f"\\textbf{{{formatted_value}}}"
                else:
                    latex_output += formatted_value

            latex_output += " \\\\\n"

        latex_output += "\\bottomrule\n\\end{tabular}\n\n"

    # --- Caption and Label ---
    latex_output += (
        "\\caption{\\textbf{Cahn-Hilliard Manifold Analysis (Baseline Comparison).} "
    )
    latex_output += "Metrics compared against the Identity transformation ($F_0$). "
    latex_output += "The top table shows intrinsic dimensionality ($\\hat{d}$) and connected components ($\\beta_0$) for each transformation. "
    latex_output += "The bottom table shows key continuity and distortion metrics with 95\\% confidence intervals where applicable. "
    latex_output += "For non-homeomorphic transformations ($F_2$ through $F_8$, excluding $F_1$ which is a homeomorphism), "
    latex_output += "the metric with the largest relative deviation from the Identity baseline is bolded in each row.}\n"
    latex_output += "\\label{tab:ch_manifolds_baseline}\n"  # Updated label
    latex_output += "\\end{table}\n"

    return latex_output


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from CH manifold metrics",
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
        default="ch_manifolds_baseline.tex",
        help="Output filename for LaTeX table",
    )

    parser.add_argument(
        "--csv_output",
        type=str,
        default="ch_manifolds_baseline.csv",
        help="Output filename for CSV data",
    )

    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to output filenames",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to generate LaTeX tables from metrics.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Modify output filenames to reflect baseline comparison
    if not args.output or args.output == "ch_manifolds.tex":
        args.output = "ch_manifolds_baseline.tex"
    if not args.csv_output or args.csv_output == "ch_manifolds.csv":
        args.csv_output = "ch_manifolds_baseline.csv"

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
    metrics_df = transform_metrics_to_dataframe(metrics_data)

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
