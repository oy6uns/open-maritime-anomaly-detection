"""Stratification module for creating indices CSV files."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def compute_route_statistics(routes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each route.

    Args:
        routes_df: DataFrame with routes data containing ROUTE_ID, SPEED, COURSE_SIN, COURSE_COS

    Returns:
        DataFrame with route statistics: ROUTE_ID, avg_sog, avg_dcog, avg_dsog, vessel_type
    """
    # Group by ROUTE_ID
    grouped = routes_df.groupby("ROUTE_ID")

    stats_list = []
    for route_id, group in grouped:
        # Sort by ROUTE_POINT_IDX or TIMESTAMP
        if "ROUTE_POINT_IDX" in group.columns:
            group = group.sort_values("ROUTE_POINT_IDX")
        elif "TIMESTAMP" in group.columns:
            group = group.sort_values("TIMESTAMP")

        # Average speed (SOG)
        avg_sog = group["SPEED"].mean()

        # Delta COG (course change)
        # Reconstruct COG from sin/cos
        cog = np.arctan2(group["COURSE_SIN"].values, group["COURSE_COS"].values)
        cog = np.degrees(cog) % 360
        dcog = np.abs(np.diff(cog))
        dcog = np.minimum(dcog, 360 - dcog)  # Circular difference
        avg_dcog = dcog.mean() if len(dcog) > 0 else 0

        # Delta SOG (speed change)
        dsog = np.abs(np.diff(group["SPEED"].values))
        avg_dsog = dsog.mean() if len(dsog) > 0 else 0

        # Vessel type (try to infer from TYPE column or CRAFT_ID)
        if "TYPE" in group.columns:
            vessel_type = group["TYPE"].iloc[0].capitalize()
        else:
            # Default to Cargo
            vessel_type = "Cargo"

        stats_list.append({
            "ROUTE_ID": route_id,
            "avg_sog": avg_sog,
            "avg_dcog": avg_dcog,
            "avg_dsog": avg_dsog,
            "vessel_type": vessel_type,
        })

    return pd.DataFrame(stats_list)


def create_stratum_bins(stats_df: pd.DataFrame, n_bins: int = 5) -> Dict[str, np.ndarray]:
    """
    Create quantile-based bins for each feature.

    Args:
        stats_df: DataFrame with route statistics
        n_bins: Number of bins per feature

    Returns:
        Dictionary mapping feature name to bin edges
    """
    bins = {}

    for feature in ["avg_sog", "avg_dcog", "avg_dsog"]:
        # Use quantile-based binning
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = stats_df[feature].quantile(quantiles).values
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        bins[feature] = bin_edges

    return bins


def assign_stratum_labels(
    stats_df: pd.DataFrame,
    bins: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Assign stratum labels to routes.

    Args:
        stats_df: DataFrame with route statistics
        bins: Dictionary of bin edges per feature

    Returns:
        DataFrame with added 'stratum' column
    """
    df = stats_df.copy()

    # Bin each feature
    sog_bins = pd.cut(df["avg_sog"], bins["avg_sog"], include_lowest=True, duplicates='drop')
    dcog_bins = pd.cut(df["avg_dcog"], bins["avg_dcog"], include_lowest=True, duplicates='drop')
    dsog_bins = pd.cut(df["avg_dsog"], bins["avg_dsog"], include_lowest=True, duplicates='drop')

    # Create stratum label: "{VesselType}|sog={range}|dcog={range}|dsog={range}"
    def format_interval(interval):
        if pd.isna(interval):
            return "(-inf, inf)"
        return f"({interval.left:.3f}, {interval.right:.3f}]"

    sog_str = sog_bins.apply(format_interval).astype(str)
    dcog_str = dcog_bins.apply(format_interval).astype(str)
    dsog_str = dsog_bins.apply(format_interval).astype(str)

    df["stratum"] = (
        df["vessel_type"] + "|" +
        "sog=" + sog_str + "|" +
        "dcog=" + dcog_str + "|" +
        "dsog=" + dsog_str
    )

    return df


def stratified_sample_percentages(
    df: pd.DataFrame,
    percentages: List[int] = [10, 5, 3, 1],
    seed: int = 0
) -> pd.DataFrame:
    """
    Perform stratified random sampling for each percentage tier.

    Args:
        df: DataFrame with 'stratum' column
        percentages: List of percentages (e.g., [10, 5, 3, 1])
        seed: Random seed

    Returns:
        DataFrame with added use_{pct}pct columns (boolean)
    """
    result = df.copy()
    rng = np.random.default_rng(seed)

    # Initialize percentage columns
    for pct in percentages:
        result[f"use_{pct}pct"] = False

    # For each stratum, sample routes
    for stratum_name, group in result.groupby("stratum"):
        indices = group.index.tolist()
        rng.shuffle(indices)

        # Nested sampling: 1pct ⊆ 3pct ⊆ 5pct ⊆ 10pct
        for pct in sorted(percentages, reverse=True):
            n_sample = max(1, int(len(indices) * pct / 100))
            sampled = indices[:n_sample]
            result.loc[sampled, f"use_{pct}pct"] = True

    return result


def assign_k_values(df: pd.DataFrame, mode: str, T: int, seed: int = 0) -> pd.DataFrame:
    """
    Assign K values for A1/A2 anomaly types.

    K is drawn from {T*0.25, T*0.5, T*0.75} with equal (1/3) proportion
    within each stratum.

    Args:
        df: DataFrame with route data (must have 'stratum' column)
        mode: Anomaly mode ('a1', 'a2', 'a3')
        T: Time slice length (e.g., 12, 24)
        seed: Random seed for shuffling

    Returns:
        DataFrame with 'K' column added (for A1/A2 only)
    """
    result = df.copy()

    if mode.lower() in ["a1", "a2"]:
        k_options = [int(T * 0.25), int(T * 0.5), int(T * 0.75)]
        rng = np.random.default_rng(seed + 7)  # offset to avoid correlation with sampling seed

        result["K"] = 0
        for stratum_name, group in result.groupby("stratum"):
            indices = group.index.tolist()
            rng.shuffle(indices)
            # Assign K values in round-robin: equal 1/3 proportion
            for j, idx in enumerate(indices):
                result.loc[idx, "K"] = k_options[j % len(k_options)]
    # A3 doesn't have K column

    return result


def generate_indices_csv(
    routes_df: pd.DataFrame,
    T: int,
    mode: str,
    output_path: str | Path,
    seed: int = 0,
    n_bins: int = 5,
    percentages: List[int] = [10, 5, 3, 1]
) -> pd.DataFrame:
    """
    Generate indices CSV file for a given T and mode.

    Args:
        routes_df: DataFrame with route data
        T: Time slice length
        mode: Anomaly mode ('a1', 'a2', 'a3')
        output_path: Output CSV path
        seed: Random seed for sampling
        n_bins: Number of bins for stratification
        percentages: List of percentages for sampling

    Returns:
        Generated indices DataFrame
    """
    # Step 1: Compute route statistics
    stats_df = compute_route_statistics(routes_df)

    # Step 2: Create stratum bins
    bins = create_stratum_bins(stats_df, n_bins=n_bins)

    # Step 3: Assign stratum labels
    stats_df = assign_stratum_labels(stats_df, bins)

    # Step 4: Stratified sampling for percentages
    stats_df = stratified_sample_percentages(stats_df, percentages=percentages, seed=seed)

    # Step 5: Assign K values (for A1/A2 only)
    if mode.lower() in ["a1", "a2"]:
        stats_df = assign_k_values(stats_df, mode, T=T, seed=seed)

    # Step 6: Add T column
    stats_df["T"] = T

    # Step 7: Select and order columns
    base_cols = ["ROUTE_ID", "T", "stratum"]
    pct_cols = [f"use_{pct}pct" for pct in percentages]

    if mode.lower() in ["a1", "a2"]:
        final_cols = base_cols + pct_cols + ["K"]
    else:  # A3
        final_cols = base_cols + pct_cols

    indices_df = stats_df[final_cols].copy()

    # Step 8: Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices_df.to_csv(output_path, index=False)

    return indices_df
