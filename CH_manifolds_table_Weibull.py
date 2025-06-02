"""
Generate tables from Cahn-Hilliard simulation metrics.

"""

import numpy as np
import pandas as pd
import os
import glob
import argparse
from typing import Dict, List, Any
from datetime import datetime
from scipy import stats

# Define metric name mapping for better labels
METRIC_LABELS = {
    "MLE_original": r"$\hat{d}_{\text{original}}$",
    "MLE_transformed": r"$\hat{d}$",  # Simplified label
    "B0_original": r"$\beta_0^{\text{original}}$",
    "B0_transformed": r"$\beta_0$",  # Simplified label
    "Forward_Lipschitz_Weibull_loc": r"$\text{Lip}(f)_{\text{loc}}$",  # New
    "Forward_Lipschitz_SNR": r"$\text{SNR}_{\text{Lip}(f)}$",  # New
    "Inverse_Lipschitz_Weibull_loc": r"$\text{Lip}(f^{-1})_{\text{loc}}$",  # New
    "Inverse_Lipschitz_SNR": r"$\text{SNR}_{\text{Lip}(f^{-1})}$",  # New
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
    Transform metrics dictionary, now expecting bootstrapped data, into a pandas DataFrame.
    - Averages scalar metrics (MLE, B0) over bootstrap iterations and calculates their std dev.
    - Fits Weibull to max Lipschitz values from bootstraps and calculates mean SNR.
    - Aggregates per-point LJD, MST data from all bootstraps for overall stats (mean, std dev).

    Args:
        metrics_data: Dictionary containing bootstrapped metrics data.

    Returns:
        DataFrame with processed metrics.
    """
    bootstrapped_metrics_list = metrics_data.get("bootstrapped_metrics")
    if not bootstrapped_metrics_list:
        print(
            "Warning: 'bootstrapped_metrics' not found or is empty. Returning empty DataFrame."
        )
        return pd.DataFrame()

    all_transform_names = set()
    if bootstrapped_metrics_list:  # Ensure it's not empty
        for bs_metric_set in bootstrapped_metrics_list:
            all_transform_names.update(bs_metric_set.keys())

    valid_transform_names = sorted(
        [
            name
            for name in all_transform_names
            if bootstrapped_metrics_list
            and isinstance(bootstrapped_metrics_list[0].get(name), dict)
        ]
    )

    processed_rows = []

    for transform_name in valid_transform_names:
        row_data: Dict[str, Any] = {"transformation": transform_name}

        # For averaging scalar metrics over bootstraps
        bs_mle_original_vals, bs_mle_transformed_vals = [], []
        bs_b0_original_vals, bs_b0_transformed_vals = [], []

        # For Lipschitz: max per bootstrap for Weibull, SNR per bootstrap
        bs_fwd_lip_maxes, bs_inv_lip_maxes = [], []
        bs_fwd_lip_snrs, bs_inv_lip_snrs = [], []

        # For LJD and MST, keep bootstrap means (not per-point values)
        bs_ljd_domain_means, bs_ljd_codomain_means, bs_ljd_combined_means = [], [], []
        bs_mst_in_means, bs_mst_ot_means, bs_mst_combined_means = [], [], []

        # Per-point aggregations for depicting variability (not for computing means of means)
        agg_ljd_domain_pp, agg_ljd_codomain_pp, agg_ljd_combined_pp = [], [], []
        agg_mst_error_in_pp, agg_mst_error_ot_pp, agg_mst_error_combined_pp = [], [], []

        for bootstrap_sample in bootstrapped_metrics_list:
            current_metrics = bootstrap_sample.get(transform_name)
            if not isinstance(current_metrics, dict):
                continue

            bs_mle_original_vals.append(current_metrics.get("MLE_original", np.nan))
            bs_mle_transformed_vals.append(
                current_metrics.get("MLE_transformed", np.nan)
            )
            bs_b0_original_vals.append(current_metrics.get("B0_original", np.nan))
            bs_b0_transformed_vals.append(current_metrics.get("B0_transformed", np.nan))

            # Lipschitz processing
            for lip_key_pp, max_list, snr_list in [
                ("Forward_Lipschitz_per_point", bs_fwd_lip_maxes, bs_fwd_lip_snrs),
                ("Inverse_Lipschitz_per_point", bs_inv_lip_maxes, bs_inv_lip_snrs),
            ]:
                pp_values = current_metrics.get(lip_key_pp)
                if pp_values is not None and len(pp_values) > 0:
                    pp_values = np.array(pp_values)[~np.isnan(pp_values)]
                    if len(pp_values) > 0:
                        max_list.append(np.max(pp_values))
                        mean_val = np.mean(pp_values)
                        var_val = np.var(pp_values)
                        # Add epsilon to variance to prevent division by zero/very small numbers
                        snr_list.append(
                            mean_val / (var_val + 1e-9) if var_val > 1e-12 else np.nan
                        )

            # Aggregating LJD, MST per-point values
            for pp_key, agg_list in [
                ("LJD_domain_per_point", agg_ljd_domain_pp),
                ("LJD_codomain_per_point", agg_ljd_codomain_pp),
                ("LJD_combined_per_point", agg_ljd_combined_pp),
                ("MST_error_in_per_point", agg_mst_error_in_pp),
                ("MST_error_ot_per_point", agg_mst_error_ot_pp),
                ("MST_error_combined_per_point", agg_mst_error_combined_pp),
            ]:
                values = current_metrics.get(pp_key)
                if values is not None:
                    values = np.array(values)[~np.isnan(values)]
                    if len(values) > 0:
                        agg_list.extend(values)

            # Gather per-bootstrap means for LJD and MST metrics
            for metric_name, bs_means_list in [
                ("LJD_domain", bs_ljd_domain_means),
                ("LJD_codomain", bs_ljd_codomain_means),
                ("LJD_combined", bs_ljd_combined_means),
                ("MST_error_in", bs_mst_in_means),
                ("MST_error_ot", bs_mst_ot_means),
                ("MST_error_combined", bs_mst_combined_means),
            ]:
                # Try direct value first (some metrics might be scalar)
                value = current_metrics.get(metric_name)
                if value is not None and not np.isnan(value):
                    bs_means_list.append(value)
                    continue

                # If not available directly, try computing from per-point values
                pp_key = f"{metric_name}_per_point"
                values = current_metrics.get(pp_key)
                if values is not None and len(values) > 0:
                    values = np.array(values)[~np.isnan(values)]
                    if len(values) > 0:
                        bs_means_list.append(np.mean(values))

        # Calculate final metrics for the row
        # For MLE and B0, calculate mean and std dev across bootstrap samples
        for metric_key_base, bs_values_list in [
            ("MLE_original", bs_mle_original_vals),
            ("MLE_transformed", bs_mle_transformed_vals),
            ("B0_original", bs_b0_original_vals),
            ("B0_transformed", bs_b0_transformed_vals),
        ]:
            clean_bs_values = [v for v in bs_values_list if not np.isnan(v)]
            if clean_bs_values:
                row_data[metric_key_base] = np.mean(clean_bs_values)
                row_data[f"{metric_key_base}_std"] = np.std(clean_bs_values)
                if "B0" in metric_key_base:  # Round B0 values
                    row_data[metric_key_base] = np.round(row_data[metric_key_base])
                    # Std for B0 might be small, format appropriately or decide if needed
                    # For now, std is kept as float, formatting will handle it.
            else:
                row_data[metric_key_base] = np.nan
                row_data[f"{metric_key_base}_std"] = np.nan

        # Lipschitz Weibull fit and SNR
        for lip_name_prefix, max_list, snr_list, weibull_col, snr_col in [
            (
                "Forward",
                bs_fwd_lip_maxes,
                bs_fwd_lip_snrs,
                "Forward_Lipschitz_Weibull_loc",
                "Forward_Lipschitz_SNR",
            ),
            (
                "Inverse",
                bs_inv_lip_maxes,
                bs_inv_lip_snrs,
                "Inverse_Lipschitz_Weibull_loc",
                "Inverse_Lipschitz_SNR",
            ),
        ]:
            clean_max_list = [x for x in max_list if not np.isnan(x)]
            if (
                clean_max_list and len(clean_max_list) >= 3
            ):  # Min points for 3-param fit
                try:
                    # c = shape, loc = location, scale = scale for weibull_max
                    c, loc, scale = stats.weibull_max.fit(
                        clean_max_list, optimizer="powell"
                    )
                    row_data[weibull_col] = loc
                except RuntimeError:  # Catch fit errors
                    print(
                        f"Warning: Weibull fit failed for {lip_name_prefix} Lipschitz ({transform_name}). Using mean of maxes."
                    )
                    row_data[weibull_col] = np.nanmean(clean_max_list)
                except Exception as e:
                    print(
                        f"Warning: Weibull fit failed for {lip_name_prefix} Lipschitz ({transform_name}) with error: {e}. Using mean of maxes."
                    )
                    row_data[weibull_col] = np.nanmean(clean_max_list)

            elif clean_max_list:  # Not enough for fit, use mean
                row_data[weibull_col] = np.nanmean(clean_max_list)
            else:  # No data
                row_data[weibull_col] = np.nan

            # Calculate SNR as mean(maxima)/std(maxima) instead of mean of per-sample SNRs
            max_array = np.array(clean_max_list)
            if len(max_array) > 1:
                # SNR = mean of maxima / std dev of maxima
                max_mean = np.mean(max_array)
                max_std = np.std(max_array)
                # Add small epsilon to avoid division by zero
                row_data[snr_col] = (
                    max_mean / (max_std + 1e-10) if max_std > 1e-10 else np.nan
                )
            else:
                row_data[snr_col] = np.nan

        # Stats for bootstrap-level means for LJD, MST
        for base_name, bs_means_list in [
            ("LJD_domain", bs_ljd_domain_means),
            ("LJD_codomain", bs_ljd_codomain_means),
            ("LJD_combined", bs_ljd_combined_means),
            ("MST_error_in", bs_mst_in_means),
            ("MST_error_ot", bs_mst_ot_means),
            ("MST_error_combined", bs_mst_combined_means),
        ]:
            if bs_means_list:
                # Calculate mean and std dev across bootstrap samples
                values_arr = np.array(bs_means_list)
                mean, std_dev = np.mean(values_arr), np.std(values_arr)
                row_data[f"{base_name}_mean"] = mean
                row_data[f"{base_name}_std"] = std_dev
            else:
                for suffix in ["_mean", "_std"]:
                    row_data[f"{base_name}{suffix}"] = np.nan

        processed_rows.append(row_data)

    return pd.DataFrame(processed_rows)


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


def format_mean_std(mean: float, std_dev: float, error_sig_figs: int = 2) -> str:
    """
    Format mean value with standard deviation in LaTeX format.
    Formats the mean based on the precision of the standard deviation.

    Args:
        mean: Mean value
        std_dev: Standard deviation
        error_sig_figs: Number of significant figures for the std dev

    Returns:
        Formatted string with mean and std dev (e.g., "1.23 $\\pm$ 0.04")
    """
    if np.isnan(mean) or np.isnan(std_dev):
        # If mean is valid but std is not, just show mean
        if not np.isnan(mean):
            # Determine sig figs for mean based on its own magnitude if std is nan
            if abs(mean) >= 100 or abs(mean) < 0.1 and mean != 0:
                return format_num(mean, error_sig_figs + 1)
            else:
                # Attempt to guess reasonable decimal places
                return f"{mean:.{error_sig_figs}f}"
        return "N/A"

    # Handle zero or very small std_dev case separately
    if std_dev <= 1e-15:  # Effectively zero std_dev
        return format_num(mean, error_sig_figs + 1)

    # Round std_dev to specified sig figs
    with np.errstate(invalid="ignore"):  # Ignore warnings for log10(0)
        exponent = np.floor(np.log10(std_dev))
    factor = 10 ** (exponent - (error_sig_figs - 1))
    rounded_std_dev = np.round(std_dev / factor) * factor

    # Determine decimal places from the rounded std_dev
    if rounded_std_dev == 0:  # Check again after rounding
        return format_num(mean, error_sig_figs + 1)

    with np.errstate(invalid="ignore"):
        std_dev_decimals = max(
            0, -int(np.floor(np.log10(rounded_std_dev))) + (error_sig_figs - 1)
        )
        if rounded_std_dev >= 1 and np.floor(np.log10(rounded_std_dev)) >= (
            error_sig_figs - 1
        ):
            std_dev_decimals = 0

    mean_str = f"{{:.{std_dev_decimals}f}}".format(mean)
    std_dev_str = f"{{:.{std_dev_decimals}f}}".format(rounded_std_dev)

    # Use scientific notation if mean is very small or large compared to its precision
    if abs(mean) > 0 and (
        abs(mean) < 10 ** (-std_dev_decimals - 1) or abs(mean) >= 10000
    ):
        mean_sci_str = format_num(mean, error_sig_figs + 1)
        std_sci_str = format_num(rounded_std_dev, error_sig_figs)
        return f"${mean_sci_str} \\pm {std_sci_str}$"

    return f"{mean_str} $\\pm$ {std_dev_str}"


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

    # Define specific metrics to consider for bolding (using new Weibull loc and LJD/MST means)
    metrics_to_bold = [
        "Forward_Lipschitz_Weibull_loc",
        "Inverse_Lipschitz_Weibull_loc",
        "LJD_domain",  # Will use LJD_domain_mean for comparison
        "LJD_codomain",  # Will use LJD_codomain_mean for comparison
        "MST_error_in",  # Will use MST_error_in_mean for comparison
        "MST_error_ot",  # Will use MST_error_ot_mean for comparison
    ]

    transform_max_deviations: Dict[str, Dict[str, Any]] = (
        {}
    )  # Ensure it's always defined

    if identity_row is not None:
        non_homeo_transforms = [
            t for t in natural_order.keys() if t not in ["Identity", "Swiss_Roll"]
        ]
        for transform in non_homeo_transforms:
            transform_max_deviations[transform] = {"metric": "", "deviation": -1.0}

        for idx, row in metrics_df.iterrows():
            transform = row["transformation"]
            if transform not in non_homeo_transforms:
                continue

            for metric_base_name in metrics_to_bold:
                # Determine column names for current row and identity row
                current_val_col = metric_base_name
                identity_val_col = metric_base_name

                if metric_base_name in [
                    "LJD_domain",
                    "LJD_codomain",
                    "MST_error_in",
                    "MST_error_ot",
                ]:
                    current_val_col = f"{metric_base_name}_mean"
                    identity_val_col = f"{metric_base_name}_mean"

                # Skip if columns don't exist (should not happen with new processing)
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
                    or abs(identity_val) < 1e-9
                ):
                    deviation = 0.0
                else:
                    # For Lipschitz, higher is "worse" or more deviation.
                    # For LJD/MST errors, higher is worse.
                    # We are looking for largest *relative deviation from Identity's value*.
                    # If Identity is X and current is Y, deviation is |Y/X - 1|.
                    deviation = (
                        abs(current_val / identity_val - 1.0)
                        if identity_val != 0
                        else (abs(current_val) if current_val != 0 else 0)
                    )

                if deviation > transform_max_deviations[transform]["deviation"]:
                    transform_max_deviations[transform]["deviation"] = deviation
                    transform_max_deviations[transform][
                        "metric"
                    ] = metric_base_name  # Store base name

    # --- Generate LaTeX Table ---
    latex_output = "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\n"

    # Table 1: Dimensionality (MLE) and Connectivity (B0)
    latex_output += "\\begin{tabular}{lccc}\n\\toprule\n"
    mle_label = METRIC_LABELS.get("MLE_transformed", "$\\hat{d}$")
    b0_label = METRIC_LABELS.get("B0_transformed", "$\\beta_0$")
    latex_output += f"F & {mle_label} $\\pm$ std & {b0_label} $\\pm$ std \\\\\n\\midrule\n"  # Added std to header

    latex_output += "Original & "
    if identity_row is not None:
        orig_mle = identity_row.get("MLE_original", np.nan)
        orig_mle_std = identity_row.get("MLE_original_std", np.nan)
        orig_b0 = identity_row.get("B0_original", np.nan)
        orig_b0_std = identity_row.get("B0_original_std", np.nan)
        latex_output += f"{format_mean_std(orig_mle, orig_mle_std, 2)} & "  # MLE with 2 sig figs for std
        latex_output += f"{format_mean_std(orig_b0, orig_b0_std, 0)}"  # B0 with 0 sig figs for std (integer mean)
    else:
        latex_output += "N/A & N/A"
    latex_output += " \\\\\n\\midrule\n"

    for idx, row in metrics_df.iterrows():
        transform = row["transformation"]
        transform_label = TRANSFORM_LABELS.get(transform, transform)
        latex_output += f"{transform_label}"
        for metric_key_base in ["MLE_transformed", "B0_transformed"]:
            mean_val = row.get(metric_key_base, np.nan)
            std_val = row.get(f"{metric_key_base}_std", np.nan)
            sig_figs_std = (
                2 if "MLE" in metric_key_base else 0
            )  # B0 std with 0 sig figs if mean is integer
            formatted_val = format_mean_std(mean_val, std_val, sig_figs_std)
            latex_output += f" & {formatted_val}"
        latex_output += " \\\\\n"
    latex_output += "\\bottomrule\n\\end{tabular}\n\n"
    latex_output += "\\vspace{3mm}\n\n"

    # Table 2: Continuity/Distortion Metrics
    # Columns: F, Lip(f)_loc, SNR_Lip(f), Lip(f^-1)_loc, SNR_Lip(f^-1), LJD_domain, LJD_codomain, MST_in, MST_ot
    # (Using LJD_domain/codomain and MST_in/ot as per previous structure for bolding)

    # Metrics for the second table
    continuity_metrics_display = [
        (
            "Forward_Lipschitz_Weibull_loc",
            False,
        ),  # False indicates not a mean/CI metric
        ("Forward_Lipschitz_SNR", False),
        ("Inverse_Lipschitz_Weibull_loc", False),
        ("Inverse_Lipschitz_SNR", False),
        (
            "LJD_domain",
            True,
        ),  # True indicates it's a mean/CI metric (uses _mean, _ci_lower, _ci_upper)
        ("LJD_codomain", True),
        ("MST_error_in", True),
        ("MST_error_ot", True),
    ]

    num_continuity_metrics = len(continuity_metrics_display)
    col_spec = "l" + "c" * num_continuity_metrics
    latex_output += f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
    latex_output += "F"
    for metric_key, _ in continuity_metrics_display:
        label = METRIC_LABELS.get(metric_key, metric_key.replace("_", " "))
        latex_output += f" & {label}"
    latex_output += " \\\\\n\\midrule\n"

    for idx, row in metrics_df.iterrows():
        transform = row["transformation"]
        transform_label = TRANSFORM_LABELS.get(transform, transform)
        latex_output += f"{transform_label}"

        for (
            metric_key,
            is_mean_std_metric,
        ) in continuity_metrics_display:  # Renamed from is_mean_ci_metric
            formatted_value = "N/A"
            if is_mean_std_metric:  # Changed from is_mean_ci_metric
                mean = row.get(f"{metric_key}_mean", np.nan)
                std_dev = row.get(f"{metric_key}_std", np.nan)  # Get std dev
                # ci_lower = row.get(f"{metric_key}_ci_lower", np.nan) # Not needed
                # ci_upper = row.get(f"{metric_key}_ci_upper", np.nan) # Not needed
                formatted_value = format_mean_std(
                    mean, std_dev, error_sig_figs=2
                )  # Use format_mean_std
            else:  # Direct value (Lipschitz Weibull loc or SNR)
                value = row.get(metric_key, np.nan)
                formatted_value = format_num(
                    value, 3 if "Lipschitz" in metric_key else 2
                )  # SNR with 2 sig figs

            should_bold = False
            if transform in transform_max_deviations:
                # metric_key could be "Forward_Lipschitz_Weibull_loc" or "LJD_domain"
                # transform_max_deviations[transform]["metric"] stores the base name used in metrics_to_bold
                # e.g. "Forward_Lipschitz_Weibull_loc" or "LJD_domain"
                if transform_max_deviations[transform]["metric"] == metric_key:
                    should_bold = True

            latex_output += " & "
            if should_bold:
                latex_output += f"\\textbf{{{formatted_value}}}"
            else:
                latex_output += formatted_value
        latex_output += " \\\\\n"

    latex_output += "\\bottomrule\n\\end{tabular}\n\n"

    # --- Caption and Label ---
    latex_output += "\\caption{\\textbf{Cahn-Hilliard Manifold Analysis with Bootstrapped Metrics.} "
    latex_output += "Metrics are derived from bootstrapped subsamples. "
    latex_output += "The top table shows mean intrinsic dimensionality ($\\hat{d}$) and mean connected components ($\\beta_0$) across bootstraps, reported as mean $\\pm$ standard deviation. "
    latex_output += "The bottom table presents continuity and distortion metrics: "
    latex_output += "Lipschitz constants estimated via Weibull location parameter ($\\text{Lip}_{\\text{loc}}$) from bootstrap maximums, "
    latex_output += "Signal-to-Noise Ratio (SNR) for Lipschitz estimates (mean over variance from per-point data, averaged across bootstraps), "
    latex_output += "and mean Local Jacobian Distortion (LJD) and Minimum Spanning Tree (MST) errors, reported as mean $\\pm$ standard deviation, aggregated over all per-point data from all bootstraps. "
    latex_output += "For non-homeomorphic transformations ($F_2$ through $F_8$), "
    latex_output += "the metric in the bottom table with the largest relative deviation from the Identity ($F_0$) baseline is bolded.}\n"
    latex_output += "\\label{tab:ch_manifolds_weibull_bootstrap}\n"  # Updated label
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
    if (
        not args.output
        or args.output == "ch_manifolds.tex"
        or args.output == "ch_manifolds_baseline.tex"
    ):
        args.output = "ch_manifolds_weibull_bootstrap.tex"
    if (
        not args.csv_output
        or args.csv_output == "ch_manifolds.csv"
        or args.csv_output == "ch_manifolds_baseline.csv"
    ):
        args.csv_output = "ch_manifolds_weibull_bootstrap.csv"

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
    print(
        "Generating LaTeX table with Weibull and SNR for Lipschitz, and aggregated mean +/- std dev..."
    )
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
