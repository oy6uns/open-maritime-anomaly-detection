"""Prompt generation module for creating user_query text files."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def calculate_deltas(route_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ΔSOG and ΔCOG for a single route.

    Args:
        route_df: DataFrame for a single route with SPEED, COURSE_SIN, COURSE_COS columns

    Returns:
        DataFrame with added delta_sog and delta_cog columns
    """
    df = route_df.copy()

    # Sort by ROUTE_POINT_IDX or TIMESTAMP
    if "ROUTE_POINT_IDX" in df.columns:
        df = df.sort_values("ROUTE_POINT_IDX")
    elif "T_INDEX" in df.columns:
        df = df.sort_values("T_INDEX")
    elif "TIMESTAMP" in df.columns:
        df = df.sort_values("TIMESTAMP")

    df = df.reset_index(drop=True)

    # Calculate ΔSOG
    sog = df["SPEED"].values
    delta_sog = np.concatenate([[0], np.diff(sog)])

    # Calculate ΔCOG (circular difference)
    cog_rad = np.arctan2(df["COURSE_SIN"].values, df["COURSE_COS"].values)
    cog_deg = np.degrees(cog_rad) % 360

    delta_cog_raw = np.diff(cog_deg)
    # Handle circular wrap-around
    delta_cog = np.where(delta_cog_raw > 180, delta_cog_raw - 360, delta_cog_raw)
    delta_cog = np.where(delta_cog < -180, delta_cog + 360, delta_cog)
    delta_cog = np.concatenate([[0], delta_cog])

    df["delta_sog"] = delta_sog
    df["delta_cog"] = delta_cog
    df["cog"] = cog_deg

    return df


def format_trajectory_prompt(
    route_df: pd.DataFrame,
    route_id: int,
    metadata: Dict,
    T: int
) -> str:
    """
    Format trajectory as LLM prompt text.

    Args:
        route_df: DataFrame for a single route
        route_id: Route ID
        metadata: Dictionary with vessel type, duration, statistics
        T: Time slice length

    Returns:
        Formatted prompt string
    """
    # Calculate deltas
    df = calculate_deltas(route_df)

    # Format metadata
    vessel_type = metadata.get("vessel_type", "Cargo")
    duration = f"{T} hours"

    # Calculate statistics
    avg_sog = df["SPEED"].mean()
    std_sog = df["SPEED"].std()
    avg_cog = df["cog"].mean()
    std_cog = df["cog"].std()

    # Build prompt
    lines = [
        "[Input Trajectory Segment]",
        "",
        f'- **Metadata:** {{ "VesselType": "{vessel_type}", "Duration": "{duration}"}}',
        f'- **Statistics:** {{ "Avg_SOG": {avg_sog:.3f} knots, "Std_SOG": {std_sog:.3f}, "Avg_COG": {avg_cog:.3f}, "Std_COG": {std_cog:.3f} }}',
        f"- **Time Series (T={T}h)**",
        "",
    ]

    # Add time series points
    for i, row in df.iterrows():
        t = i + 1
        lat = row["LAT"]
        lon = row["LON"]
        sog = row["SPEED"]
        cog = row["cog"]
        delta_sog = row["delta_sog"]
        delta_cog = row["delta_cog"]

        lines.append(f"t={t}: [")
        lines.append(f"  Lat: {lat:.6f}, Lon: {lon:.6f},")
        lines.append(f"  SOG: {sog:.3f}, COG: {cog:.1f},")
        lines.append(f"  ΔSOG: {delta_sog:+.3f}, ΔCOG: {delta_cog:+.1f}")
        lines.append("]")
        if i < len(df) - 1:
            lines.append("")

    return "\n".join(lines)


def determine_anomaly_types(
    route_id: int,
    indices_dict: Dict[str, pd.DataFrame]
) -> List[str]:
    """
    Determine which anomaly types (A1/A2/A3) to generate prompts for.

    Args:
        route_id: Route ID
        indices_dict: Dictionary mapping mode ('a1', 'a2', 'a3') to indices DataFrame

    Returns:
        List of anomaly types to generate (e.g., ['A1', 'A3'])
    """
    anomaly_types = []

    for mode in ["a1", "a2", "a3"]:
        if mode in indices_dict:
            indices_df = indices_dict[mode]
            # Check if route is in any percentage column
            route_mask = indices_df["ROUTE_ID"] == route_id
            if route_mask.any():
                row = indices_df[route_mask].iloc[0]
                # Check if any use_{pct}pct column is True
                pct_cols = [col for col in row.index if col.startswith("use_") and col.endswith("pct")]
                if any(row[col] for col in pct_cols):
                    anomaly_types.append(mode.upper())

    return anomaly_types


def generate_prompts_for_routes(
    routes_df: pd.DataFrame,
    indices_dir: str | Path,
    output_dir: str | Path,
    T: int,
    vessel_type: str = "Cargo"
) -> Dict[str, int]:
    """
    Generate prompt files for all routes.

    Args:
        routes_df: DataFrame with route data
        indices_dir: Directory containing indices CSV files
        output_dir: Output directory for user_query/*.txt files
        T: Time slice length
        vessel_type: Default vessel type

    Returns:
        Dictionary with generation statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    indices_dir = Path(indices_dir)

    # Load indices files
    indices_dict = {}
    for mode in ["a1", "a2", "a3"]:
        indices_file = indices_dir / f"indices_{T}_{mode}.csv"
        if indices_file.exists():
            indices_dict[mode] = pd.read_csv(indices_file)

    if not indices_dict:
        raise FileNotFoundError(f"No indices files found in {indices_dir}")

    stats = {"total_prompts": 0, "routes_processed": 0}

    # Group routes by ROUTE_ID
    for route_id, group in routes_df.groupby("ROUTE_ID"):
        # Sort by ROUTE_POINT_IDX or TIMESTAMP
        if "ROUTE_POINT_IDX" in group.columns:
            group = group.sort_values("ROUTE_POINT_IDX")
        elif "TIMESTAMP" in group.columns:
            group = group.sort_values("TIMESTAMP")

        group = group.reset_index(drop=True)

        # Skip if route doesn't have exactly T points
        if len(group) != T:
            continue

        # Determine which anomaly types to generate
        anomaly_types = determine_anomaly_types(route_id, indices_dict)

        if not anomaly_types:
            continue

        # Extract metadata
        if "TYPE" in group.columns:
            route_vessel_type = group["TYPE"].iloc[0].capitalize()
        else:
            route_vessel_type = vessel_type

        metadata = {
            "vessel_type": route_vessel_type,
            "duration": f"{T} hours",
        }

        # Generate prompt for each anomaly type
        for anom_type in anomaly_types:
            prompt_text = format_trajectory_prompt(group, route_id, metadata, T)

            # Save to file: route_{ROUTE_ID}_{A1|A2|A3}.txt
            filename = f"route_{route_id}_{anom_type}.txt"
            filepath = output_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(prompt_text)

            stats["total_prompts"] += 1

        stats["routes_processed"] += 1

    return stats
