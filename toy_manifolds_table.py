import torch
import pandas as pd
import numpy as np
from scipy import stats
import argparse
from typing import Dict, List, Callable, Optional
from datetime import datetime
import os

# Import necessary functions from other modules
from utils import (
    compute_metrics_consolidated,
    TopoAELoss,
    MinMaxRTDLoss,
    get_device,
)

# Import transformations from toy_point_probs module
from src_toy_manifolds import (
    split,
    hole,
    pinch,
    collapse,
    swiss_roll,
    dog_ear,
    ribbon,
    cylinder,
)


def get_transformations() -> Dict[str, Callable]:
    """
    Return dictionary of all transformation functions.

    Returns:
        Dict[str, Callable]: Mapping from transformation names to their functions
    """
    return {
        "F_1": swiss_roll,
        "F_2": dog_ear,
        "F_3": ribbon,
        "F_4": cylinder,
        "F_5": split,
        "F_6": hole,
        "F_7": collapse,
        "F_8": pinch,
    }


def run_simulations(
    n_points: int = 256,
    n_sims: int = 10,
    seed: int = 42,
    transforms: Optional[List[str]] = None,
    device: str = None,
) -> pd.DataFrame:
    """
    Run all simulations and collect results in a pandas DataFrame.

    Args:
        n_points: Number of points in each point cloud
        n_sims: Number of simulation repeats for statistical analysis
        seed: Random seed for reproducibility
        transforms: List of transformation names to run (if None, run all)
        device: Device to use for computations (None for auto-detection)
        shuffle_order: Whether to shuffle the order of transformations

    Returns:
        pd.DataFrame: Results organized in a DataFrame
    """
    device = get_device(device)

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize loss functions
    tloss = TopoAELoss()
    rloss = MinMaxRTDLoss()

    # Get all transformations
    all_transformations = get_transformations()

    # Filter transformations if specified
    if transforms is not None:
        transformations = {
            k: v for k, v in all_transformations.items() if k in transforms
        }
        if not transformations:
            raise ValueError(
                f"No valid transformations found in {transforms}. Available: {list(all_transformations.keys())}"
            )
    else:
        transformations = all_transformations

    # Prepare data collection
    all_results = []

    for sim in range(n_sims):
        print(f"Running simulation {sim + 1}/{n_sims}")

        # Generate uniform point cloud in [-1, 1]²
        points = torch.rand(n_points, 2, device=device) * 2 - 1

        # Add z=0 dimension for 3D transformations
        points = torch.cat((points, torch.zeros(n_points, 1, device=device)), dim=1)

        # Get items as list to allow for order shuffling
        transform_items = list(transformations.items())

        for transform_name, transform_fn in transform_items:
            # Apply transformation
            transformed_points = transform_fn(points.clone())

            # Compute metrics including the new LJD metric
            metrics = compute_metrics_consolidated(
                points,
                transformed_points,
                metrics=["mle", "beta0", "lip", "topoae", "rtd", "ljd"],
                tloss=tloss,
                rloss=rloss,
                k_min=5,
                k_max=10,
                k_jac=5,  # Number of neighbors for Jacobian calculations
                p=2.0,
            )

            # Store results with metadata
            result = {"simulation": sim, "transformation": transform_name, **metrics}
            all_results.append(result)

    # Convert to DataFrame for easier analysis
    return pd.DataFrame(all_results)


def compute_statistical_analysis(results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform comprehensive statistical analysis on the simulation results.

    Args:
        results_df: DataFrame containing all simulation results

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames with statistical analyses
    """
    # Define metrics for analysis
    paired_metrics = [
        ("B0_original", "B0_transformed"),
        ("MLE_original", "MLE_transformed"),
    ]

    # Standalone metrics
    standalone_metrics = [
        "Forward_Lipschitz",
        "Inverse_Lipschitz",
        "TopoAE",
        "RTD",
        "LJD_combined",  # Add the new LJD metric
    ]

    # Group by transformation to compute statistics
    grouped = results_df.groupby("transformation")

    # Dictionary to store all statistical results
    stats_dict = {}

    # Process paired metrics with appropriate tests
    paired_stats = []
    for orig_metric, trans_metric in paired_metrics:
        for transform_name, group in grouped:
            # Get paired values
            orig_values = group[orig_metric].values
            trans_values = group[trans_metric].values

            # Compute basic statistics
            orig_mean, orig_std = np.mean(orig_values), np.std(orig_values)
            trans_mean, trans_std = np.mean(trans_values), np.std(trans_values)

            # Test for normality of differences
            diff_values = orig_values - trans_values
            _, shapiro_p = stats.shapiro(diff_values)
            is_normal = shapiro_p > 0.05

            # For B₀, check consistency (how often we get the same value)
            if orig_metric == "B0_original":
                # Count most common B₀ value and its frequency
                unique, counts = np.unique(trans_values, return_counts=True)
                most_common_idx = np.argmax(counts)
                most_common_b0 = unique[most_common_idx]
                consistency = counts[most_common_idx] / len(trans_values)

                # For discrete B₀ values, try Wilcoxon signed-rank test first
                try:
                    w_stat, w_p_value = stats.wilcoxon(
                        orig_values,
                        trans_values,
                        zero_method="zsplit",
                        alternative="two-sided",
                    )
                    if w_p_value == 1.0:
                        # Try parametric test
                        t_stat, p_value = stats.ttest_rel(orig_values, trans_values)
                        test_name = "paired t-test"
                        test_stat = t_stat
                    else:
                        test_name = "Wilcoxon signed-rank"
                        test_stat = w_stat
                        p_value = w_p_value
                except ValueError:
                    # If test fails (e.g., due to identical values), use t-test
                    t_stat, p_value = stats.ttest_rel(orig_values, trans_values)
                    test_name = "paired t-test"
                    test_stat = t_stat
            else:
                # For continuous metrics (MLE), choose test based on normality
                if is_normal:
                    # Use paired t-test for normally distributed differences
                    t_stat, p_value = stats.ttest_rel(orig_values, trans_values)
                    test_name = "paired t-test"
                    test_stat = t_stat
                else:
                    # Use Wilcoxon signed-rank test for non-normal distributions
                    try:
                        w_stat, p_value = stats.wilcoxon(
                            orig_values,
                            trans_values,
                            zero_method="zsplit",
                            alternative="two-sided",
                        )
                        test_name = "Wilcoxon signed-rank"
                        test_stat = w_stat
                    except ValueError:
                        # Fallback to t-test if Wilcoxon fails
                        t_stat, p_value = stats.ttest_rel(orig_values, trans_values)
                        test_name = "paired t-test"
                        test_stat = t_stat

                # Compute Cohen's d effect size for paired samples
                diff = orig_values - trans_values
                consistency = np.nan  # Not applicable for continuous metrics
                most_common_b0 = np.nan

            # Get confidence intervals using bootstrap
            n_bootstrap = 1000
            bootstrap_diffs = []
            n_samples = len(orig_values)

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_diff = np.mean(orig_values[indices]) - np.mean(
                    trans_values[indices]
                )
                bootstrap_diffs.append(bootstrap_diff)

            # Calculate 95% confidence intervals
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)

            # Store results
            paired_stats.append(
                {
                    "transformation": transform_name,
                    "metric": orig_metric.split("_")[
                        0
                    ],  # Just the metric name without _original
                    "original_mean": orig_mean,
                    "original_std": orig_std,
                    "transformed_mean": trans_mean,
                    "transformed_std": trans_std,
                    "mean_difference": orig_mean - trans_mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "test_statistic": test_stat,
                    "p_value": p_value,
                    "test_name": test_name,
                    "is_normal": is_normal,
                    "shapiro_p": shapiro_p,
                    "consistency": consistency if not np.isnan(consistency) else None,
                    "most_common_value": (
                        most_common_b0 if not np.isnan(most_common_b0) else None
                    ),
                }
            )

    # Convert to DataFrame
    stats_dict["paired_metrics"] = pd.DataFrame(paired_stats)

    # Process standalone metrics with bootstrap confidence intervals
    standalone_stats = []
    for metric in standalone_metrics:
        for transform_name, group in grouped:
            values = group[metric].values
            mean, std = np.mean(values), np.std(values)

            # Bootstrap for confidence intervals
            n_bootstrap = 1000
            bootstrap_means = []
            n_samples = len(values)

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_mean = np.mean(values[indices])
                bootstrap_means.append(bootstrap_mean)

            # Calculate 95% confidence intervals
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)

            standalone_stats.append(
                {
                    "transformation": transform_name,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "cv": (
                        std / mean if mean != 0 else np.nan
                    ),  # Coefficient of variation
                }
            )

    # Convert to DataFrame
    stats_dict["standalone_metrics"] = pd.DataFrame(standalone_stats)

    return stats_dict


def latex_safe_name(name: str) -> str:
    """
    Convert transformation names to LaTeX-safe format.

    Args:
        name: Original transformation name

    Returns:
        str: LaTeX-safe name with proper formatting
    """
    # Dictionary of replacements for better LaTeX formatting
    replacements = {
        "F_1": "$F_1$",
        "F_2": "$F_2$",
        "F_3": "$F_3$",
        "F_4": "$F_4$",
        "F_5": "$F_5$",
        "F_6": "$F_6$",
        "F_7": "$F_7$",
        "F_8": "$F_8$",
    }

    return replacements.get(name, name.replace("_", " ").title())


def format_num(value, precision=2):
    """
    Format numbers with appropriate significant figures.

    Args:
        value: Number to format
        precision: Desired precision

    Returns:
        Formatted number string
    """
    if value is None or np.isnan(value):
        return "--"
    elif abs(value) < 0.01:
        return f"{value:.2e}"
    elif abs(value) >= 10000:
        return f"{value:.2e}"
    elif abs(value) >= 100:
        return f"{value:.0f}"
    elif abs(value) >= 10:
        return f"{value:.1f}"
    else:
        return f"{value:.{precision}f}"


def format_mean_std(mean, std, precision=2):
    """
    Format mean ± std with appropriate precision.

    Args:
        mean: Mean value
        std: Standard deviation
        precision: Desired precision

    Returns:
        Formatted mean ± std string
    """
    if np.isnan(mean) or np.isnan(std):
        return "--"

    # If std is very small (would show as 0.00), increase precision or use scientific notation
    if 0 < std < 0.01:
        return f"{format_num(mean, precision)} $\\pm$ {std:.2e}"
    else:
        mean_str = format_num(mean, precision)
        std_str = format_num(std, precision)
        return f"{mean_str} $\\pm$ {std_str}"


def format_mean_ci(mean, ci_lower, ci_upper, precision=2):
    """
    Format mean with confidence intervals.

    Args:
        mean: Mean value
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        precision: Desired precision

    Returns:
        Formatted string with mean and CI
    """
    if np.isnan(mean) or np.isnan(ci_lower) or np.isnan(ci_upper):
        return "--"

    mean_str = format_num(mean, precision)
    ci_width = format_num(ci_upper - ci_lower, precision)
    return f"{mean_str} $\\pm$ {ci_width}"


def get_significance_stars(p_value: float) -> str:
    """
    Return significance stars based on p-value.

    Args:
        p_value: Statistical p-value

    Returns:
        str: Star symbols indicating significance level
    """
    if p_value < 0.001:
        return "$^{***}$"
    elif p_value < 0.01:
        return "$^{**}$"
    elif p_value < 0.05:
        return "$^{*}$"
    else:
        return ""


def format_latex_tables(
    stats_dict: Dict[str, pd.DataFrame],
    n_sims: int,
    output_format: str = "standard",
) -> str:
    """
    Format results into LaTeX tables with appropriate statistical information.

    Args:
        stats_dict: Dictionary of DataFrames with statistical analyses
        output_format: Output format type ("standard", "simple", or "full")

    Returns:
        str: Formatted LaTeX table code
    """
    paired_df = stats_dict["paired_metrics"]
    standalone_df = stats_dict["standalone_metrics"]

    # Get all transformation names
    transformations = paired_df["transformation"].unique()

    # Extract original values (same for all transformations of the same type)
    b0_entries = paired_df[paired_df["metric"] == "B0"]
    mle_entries = paired_df[paired_df["metric"] == "MLE"]

    # Get the first entry for each metric to extract original values
    # (These should be the same across all transformations)
    first_b0 = b0_entries.iloc[0]
    orig_b0_mean, orig_b0_std = first_b0["original_mean"], first_b0["original_std"]

    first_mle = mle_entries.iloc[0]
    orig_mle_mean, orig_mle_std = first_mle["original_mean"], first_mle["original_std"]

    # Start the full table environment
    latex_output = "\\begin{table}[htbp]\n\\centering\n\\small\n\n"

    if output_format == "simple":
        # Simplified table format
        latex_output += "\\begin{tabular}{lcccc}\n\\toprule\n"
        latex_output += (
            "Transformation & $\\beta_0$ & Dim. & Lip. F & Lip. B \\\\\n\\midrule\n"
        )

        # Original point cloud row
        latex_output += f"Original & {format_num(orig_b0_mean, 1)} & {format_num(orig_mle_mean, 2)} & -- & -- \\\\\n"
        latex_output += "\\midrule\n"

        for transform in transformations:
            # Get B0 row for this transformation
            b0_row = b0_entries[b0_entries["transformation"] == transform].iloc[0]
            b0_mean = b0_row["transformed_mean"]

            # Get MLE row for this transformation
            mle_row = mle_entries[mle_entries["transformation"] == transform].iloc[0]
            mle_mean = mle_row["transformed_mean"]

            # Get Lipschitz constants
            fwd_lip = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "Forward_Lipschitz")
            ].iloc[0]["mean"]

            inv_lip = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "Inverse_Lipschitz")
            ].iloc[0]["mean"]

            # Format row
            latex_output += f"{latex_safe_name(transform)} & "
            latex_output += f"{format_num(b0_mean, 1)} & "
            latex_output += f"{format_num(mle_mean, 2)} & "
            latex_output += f"{format_num(fwd_lip, 1)} & "
            latex_output += f"{format_num(inv_lip, 1)} \\\\\n"

    elif output_format == "full":
        # Flatter top table for dimensionality and B0 values
        latex_output += "\\begin{tabular}{lcccccccccc}\n\\toprule\n"
        latex_output += " & Original "

        # Add column for each transformation
        for transform in transformations:
            latex_output += f"& {latex_safe_name(transform)} "
        latex_output += "\\\\\n\\midrule\n"

        # Dimensionality row
        latex_output += "$\\hat{d}$ & "
        latex_output += f"{format_mean_std(orig_mle_mean, orig_mle_std, 2)} "

        for transform in transformations:
            # Get MLE row for this transformation
            mle_row = mle_entries[mle_entries["transformation"] == transform].iloc[0]
            mle_mean = mle_row["transformed_mean"]
            mle_std = mle_row["transformed_std"]

            latex_output += f"& {format_mean_std(mle_mean, mle_std, 2)} "

        latex_output += "\\\\\n"

        # B0 row
        latex_output += "$\\beta_0$ & "
        latex_output += f"{format_mean_std(orig_b0_mean, orig_b0_std, 1)} "

        for transform in transformations:
            # Get B0 row for this transformation
            b0_row = b0_entries[b0_entries["transformation"] == transform].iloc[0]
            b0_mean = b0_row["transformed_mean"]
            b0_std = b0_row["transformed_std"]

            # Special formatting for split transformation to highlight B₀=2
            if (transform == "F_6" or transform == "split") and b0_mean > 1.5:
                b0_val_str = f"\\textbf{{{format_mean_std(b0_mean, b0_std, 1)}}}"
            else:
                b0_val_str = format_mean_std(b0_mean, b0_std, 1)

            latex_output += f"& {b0_val_str} "

        latex_output += "\\\\\n\\bottomrule\n\\end{tabular}\n\n"

        # Add spacing
        latex_output += "\\vspace{3mm}\n\n"

        # Second table for all other metrics - now with 2-sigma error bars
        latex_output += "\\begin{tabular}{lccccc}\n\\toprule\n"
        latex_output += "Func. & Forward Lipschitz & Inverse Lipschitz & TopoAE & RTD & LJD \\\\\n\\midrule\n"

        for transform in transformations:
            fwd_lip = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "Forward_Lipschitz")
            ].iloc[0]

            inv_lip = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "Inverse_Lipschitz")
            ].iloc[0]

            topoae = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "TopoAE")
            ].iloc[0]

            rtd = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "RTD")
            ].iloc[0]

            ljd = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "LJD_combined")
            ].iloc[0]

            # Format row with 1-sigma error bars (mean ± std)
            latex_output += f"{latex_safe_name(transform)} & "
            latex_output += f"{format_mean_std(fwd_lip['mean'], fwd_lip['std'], 2)} & "
            latex_output += f"{format_mean_std(inv_lip['mean'], inv_lip['std'], 2)} & "
            latex_output += f"{format_mean_std(topoae['mean'], topoae['std'], 2)} & "
            latex_output += f"{format_mean_std(rtd['mean'], rtd['std'], 2)} & "
            latex_output += f"{format_mean_std(ljd['mean'], ljd['std'], 2)} \\\\\n"

    else:  # Standard format (default)
        # Flatter table format for dimensionality and B0
        latex_output += "\\begin{tabular}{lcccccccccc}\n\\toprule\n"
        latex_output += " & Original "

        # Add column for each transformation
        for transform in transformations:
            latex_output += f"& {latex_safe_name(transform)} "
        latex_output += "\\\\\n\\midrule\n"

        # Dimensionality row
        latex_output += "$\\hat{d}$ & "
        latex_output += f"{format_num(orig_mle_mean, 2)} "

        for transform in transformations:
            # Get MLE row for this transformation
            mle_row = mle_entries[mle_entries["transformation"] == transform].iloc[0]
            mle_mean = mle_row["transformed_mean"]

            latex_output += f"& {format_num(mle_mean, 2)} "

        latex_output += "\\\\\n"

        # B0 row
        latex_output += "$\\beta_0$ & "
        latex_output += f"{format_num(orig_b0_mean, 1)} "

        for transform in transformations:
            # Get B0 row for this transformation
            b0_row = b0_entries[b0_entries["transformation"] == transform].iloc[0]
            b0_mean = b0_row["transformed_mean"]

            # Special formatting for split transformation to highlight B₀=2
            if (transform == "F_6" or transform == "split") and b0_mean > 1.5:
                b0_val_str = f"\\textbf{{{format_num(b0_mean, 1)}}}"
            else:
                b0_val_str = format_num(b0_mean, 1)

            latex_output += f"& {b0_val_str} "

        latex_output += "\\\\\n\\bottomrule\n\\end{tabular}\n\n"

        # Add spacing
        latex_output += "\\vspace{3mm}\n\n"

        # Subtable for transformation metrics
        latex_output += "\\begin{tabular}{lccccc}\n\\toprule\n"
        latex_output += "Transformation & Forward Lipschitz & Inverse Lipschitz & TopoAE & RTD & LJD \\\\\n\\midrule\n"

        for transform in transformations:
            # Get metrics for this transformation
            fwd_lip = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "Forward_Lipschitz")
            ].iloc[0]

            inv_lip = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "Inverse_Lipschitz")
            ].iloc[0]

            topoae = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "TopoAE")
            ].iloc[0]

            rtd = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "RTD")
            ].iloc[0]

            ljd = standalone_df[
                (standalone_df["transformation"] == transform)
                & (standalone_df["metric"] == "LJD_combined")
            ].iloc[0]

            # Format the metrics with appropriate precision
            fwd_lip_str = format_mean_std(fwd_lip["mean"], fwd_lip["std"], 2)

            # Special formatting for very large inverse Lipschitz values
            if inv_lip["mean"] > 1000:
                inv_mean = inv_lip["mean"]
                inv_std = inv_lip["std"]
                # Format as scientific notation with appropriate prefix
                inv_lip_str = f"{inv_mean:.1e} $\\pm$ {inv_std:.1e}"
                # Replace e+0x with ×10$^x$ for LaTeX
                inv_lip_str = inv_lip_str.replace("e+", "×10$^")
                inv_lip_str = inv_lip_str.replace("e-", "×10$^{-")
                if "×10$^" in inv_lip_str:
                    inv_lip_str = inv_lip_str.replace("$^", "$^{") + "}"
            else:
                inv_lip_str = format_mean_std(inv_lip["mean"], inv_lip["std"], 2)

            # Format TopoAE and RTD with appropriate precision
            if topoae["mean"] < 10:
                topoae_str = format_mean_std(topoae["mean"], topoae["std"], 2)
            else:
                topoae_str = format_mean_std(
                    round(topoae["mean"]), round(topoae["std"]), 0
                )

            rtd_str = format_mean_std(rtd["mean"], rtd["std"], 2)

            # Format LJD with appropriate precision
            ljd_str = format_mean_std(ljd["mean"], ljd["std"], 2)

            # Format row
            latex_output += f"{latex_safe_name(transform)} & "
            latex_output += f"{fwd_lip_str} & "
            latex_output += f"{inv_lip_str} & "
            latex_output += f"{topoae_str} & "
            latex_output += f"{rtd_str} & "
            latex_output += f"{ljd_str} \\\\\n"

    latex_output += "\\bottomrule\n\\end{tabular}\n\n"

    # Add table caption explaining the metrics with updated error bar description
    latex_output += (
        "\\caption{\\textbf{Topological analysis of geometric transformations.} \n"
    )
    latex_output += "The top portion shows intrinsic dimensionality ($\\hat{d}$) and $\\beta_0$ values of original and transformed point clouds. "
    latex_output += "The bottom portion presents continuity and topological similarity metrics: Lipschitz constants measure the continuity of mappings, "
    latex_output += "while TopoAE, RTD, and LJD measure topological preservation. "
    if output_format == "full":
        latex_output += (
            "All error bars represent $\\pm \\sigma$ (standard deviation), computed over $n="
            + str(n_sims)
            + "$ independent runs "
        )
        latex_output += "with different random seeds for point cloud generation, where $\\sigma$ represents the standard deviation over values found per sample. "
    else:
        latex_output += (
            "Error bars represent $\\pm \\sigma$ (standard deviation), computed over $n="
            + str(n_sims)
            + "$ independent runs, where $\\sigma$ represents the standard deviation over values found per sample. "
        )
    latex_output += "}\n"
    latex_output += "\\label{tab:topo_analysis}\n"
    latex_output += "\\end{table}"

    return latex_output


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate topological analysis tables for different transformations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n_points", type=int, default=256, help="Number of points in each point cloud"
    )

    parser.add_argument(
        "--n_sims", type=int, default=10, help="Number of simulation repeats"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--transformations",
        type=str,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="tables",
        help="Output filename for LaTeX table",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="toy_manifolds.tex",
        help="Output filename for LaTeX table",
    )

    parser.add_argument(
        "--csv_output",
        type=str,
        default="toy_manifolds.csv",
        help="Output filename for raw results CSV",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["standard", "simple", "full"],
        default="full",
        help="Table format to generate",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: auto-detect)",
    )

    parser.add_argument(
        "--timestamp", action="store_true", help="Add timestamp to output filenames"
    )

    return parser.parse_args()


def main():
    """
    Main function to run simulations and generate LaTeX tables.
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

    print(f"Running {args.n_sims} simulations with {args.n_points} points each...")

    # Run simulations
    results_df = run_simulations(
        n_points=args.n_points,
        n_sims=args.n_sims,
        seed=args.seed,
        transforms=args.transformations,
        device=args.device,
    )

    # Perform statistical analysis
    print("Performing statistical analysis...")
    stats_dict = compute_statistical_analysis(results_df)

    # Format results into LaTeX tables
    print(f"Generating LaTeX table in {args.format} format...")
    latex_code = format_latex_tables(
        stats_dict,
        n_sims=args.n_sims,
        output_format=args.format,
    )

    # Save results to CSV for further analysis
    path = os.path.join(args.output_folder, args.csv_output)
    os.makedirs(args.output_folder, exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f"CSV saved to {path}")
    # Save LaTeX code to file
    path = os.path.join(args.output_folder, args.output)
    os.makedirs(args.output_folder, exist_ok=True)
    with open(path, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table code saved to {path}")


if __name__ == "__main__":
    main()
