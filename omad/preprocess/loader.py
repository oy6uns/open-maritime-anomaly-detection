# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
from pathlib import Path


BASE_PATH = "./data"

def _resolve_base_path():
    """Auto-detect data directory (handles both ./data/{year}/ and ./data/West Grid/{year}/)."""
    # Check if West Grid subdirectory exists
    west_grid = os.path.join(BASE_PATH, "West Grid")
    if os.path.isdir(west_grid):
        return west_grid
    return BASE_PATH

def load_data(year_list):
    """
    Load all monthly CSV data (cargo, passenger, tanker) for specified years
    """
    resolved_base = _resolve_base_path()
    columns = ["CRAFT_ID", "LON", "LAT", "COURSE", "SPEED", "TIMESTAMP", "Track_ID"]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_mapping = {
        "Jan":("Jan","Jan"), "Feb":("Feb","Feb"), "Mar":("Mar","Mar"),
        "Apr":("Apr","Apr"), "May":("May","May"), "Jun":("Jun","Jun"),
        "Jul":("Jul","Jul"), "Aug":("Aug","Aug"), "Sep":("Sept","Sep"),
        "Oct":("Oct","Oct"), "Nov":("Nov","Nov"), "Dec":("Dec","Dec")
    }
    dfs = []
    for year in year_list:
        base_path=f"{resolved_base}/{year}"
        types = ["cargo", "tanker"]
        
        for t in types:
            for m in months:
                f_m, f_n = month_mapping[m]
                file_path = f"{base_path}/{t}/{f_m}/MPF_{f_n}_{year}_Grid_{t.capitalize()}.csv"
                if os.path.exists(file_path):
                    tmp = pd.read_csv(file_path, names=columns, skiprows=1)
                    tmp["TYPE"] = t
                    tmp["MONTH"] = m
                    dfs.append(tmp)
                else:
                    print(f"[WARN] File not found: {file_path}")
    if not dfs:
        raise ValueError(
            f"No AIS data files found in '{resolved_base}/'. "
            f"Expected structure: {resolved_base}/{{year}}/{{cargo|tanker}}/{{month}}/MPF_*.csv\n"
            f"Tip: Copy raw AIS data to ./data/ (e.g., cp -r /path/to/West\\ Grid ./data/)"
        )
    df = pd.concat(dfs, ignore_index=True)
    df.dropna(inplace=True)
    
    # TIMESTAMP conversion
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df.dropna(subset=["TIMESTAMP"], inplace=True)
    df["TIMESTAMP"] = df["TIMESTAMP"].dt.round("1H")
    
    # COURSE -> sin, cos
    df["COURSE_SIN"] = np.sin(np.deg2rad(df["COURSE"]))
    df["COURSE_COS"] = np.cos(np.deg2rad(df["COURSE"]))
    
    df["Track_ID"] = df["Track_ID"].astype(str)
    return df

def interpolate_group(group):
    """Hourly interpolation per Track_ID"""
    track_id = group["Track_ID"].iloc[0]
    craft_id = group["CRAFT_ID"].iloc[0]
    group = group.set_index("TIMESTAMP")
    group = group[~group.index.duplicated(keep="first")]
    group = group.resample("1H").interpolate()
    group["Track_ID"] = track_id
    group["CRAFT_ID"] = group["CRAFT_ID"].fillna(craft_id).astype(float).astype(int)
    if "ANOMALY" not in group.columns:
        group["ANOMALY"] = False
    else:
        group["ANOMALY"] = group["ANOMALY"].fillna(False)
    return group

def interpolate_data(df):
    return df.groupby("Track_ID", group_keys=False).apply(interpolate_group).reset_index()

def prepare_data_pipeline():
    year_list = [2018, 2019, 2020]
    df = load_data(year_list)
    df_interp = interpolate_data(df)
    track_counts = df_interp["Track_ID"].value_counts()
    exclude_ids = track_counts[track_counts < 10].index
    df_filtered = df_interp[~df_interp["Track_ID"].isin(exclude_ids)].copy()
    return df_filtered

if __name__ == "__main__":
    df = prepare_data_pipeline()
    print(df.head())
    out_path = Path("outputs") / "omtad.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"CSV saved: {out_path.resolve()}")