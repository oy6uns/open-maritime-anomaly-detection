"""
OMAD - A1 Anomaly Injector
================================
A1: Position anomaly based on SED (cross-track deviation).

"How many times the usual deviation qualifies as anomaly"

Algorithm:
1. LLM selects top-K positions: I_K = TopK({s_t}, K)
2. Calculate local SED baseline for the route
3. For each i in I_K:
   - Calculate cross-track direction u_perp (perpendicular to i-1 → i+1)
   - Randomly choose side (-1 or +1)
   - If chosen side increases SED: alpha = target_sed - original_sed
     else: alpha = target_sed + original_sed
     (baseline: i-1 → i+1, measured point: i)
4. Inject: p_i^fake = p_i + alpha_i * u_perp
"""
from __future__ import annotations

import random
from typing import List, Optional

from inject_utils import (
    InjectionPlan,
    calculate_sed,
    calculate_sed_local,
    destination_point,
    get_cross_track_direction,
    minmax_normalize_scores,
    EPSILON_SED,
    THETA_G,
)


def inject_a1(
    rows: List[dict],
    plan: InjectionPlan,
    rng: Optional[random.Random] = None,
    theta_g: float = THETA_G,
    epsilon_sed: float = EPSILON_SED,
    min_alpha: float = 100.0,  # Minimum displacement in meters
    max_alpha: float = 7039.479574,  # Maximum displacement in meters
    track_index: Optional[dict] = None,
    track_pos: Optional[dict] = None,
) -> List[dict]:
    """
    Inject A1 anomaly (position perturbation based on SED).

    Args:
        rows: List of route point dictionaries
        plan: InjectionPlan with scores and K
        rng: Random number generator (unused, kept for API compatibility)
        theta_g: SED multiplier (default 3.0)
        epsilon_sed: Minimum SED_local clipping value (default Q10=110.49m)
        min_alpha: Minimum displacement in meters

    Returns:
        List of modified route point dictionaries
    """
    if rng is None:
        rng = random.Random()

    original_rows = [dict(r) for r in rows]
    rows = [dict(r) for r in rows]  # Deep copy (to mutate)
    W = len(rows)

    if W < 3:
        return rows

    def _get_track_prev(row: dict, offset: int) -> Optional[dict]:
        if not track_index or not track_pos:
            return None
        track_id = row.get("TRACK_ID")
        ts = row.get("TIMESTAMP")
        if track_id is None or ts is None:
            return None
        idx = track_pos.get((track_id, ts))
        if idx is None or (idx - offset) < 0:
            return None
        track_rows = track_index.get(track_id, [])
        if idx - offset >= len(track_rows):
            return None
        return track_rows[idx - offset]

    def _get_track_next(row: dict, offset: int) -> Optional[dict]:
        if not track_index or not track_pos:
            return None
        track_id = row.get("TRACK_ID")
        ts = row.get("TIMESTAMP")
        if track_id is None or ts is None:
            return None
        idx = track_pos.get((track_id, ts))
        if idx is None:
            return None
        track_rows = track_index.get(track_id, [])
        if idx + offset >= len(track_rows):
            return None
        return track_rows[idx + offset]

    # 1. Get top-K indices from LLM scores
    top_k_indices = set(plan.get_top_k_indices())
    norm_scores = minmax_normalize_scores(plan.scores)

    # 2. Calculate local SED baseline (clipped)
    sed_local = calculate_sed_local(original_rows)
    sed_local_clipped = max(sed_local, epsilon_sed)

    # 3. For each top-K index, inject anomaly
    for i in top_k_indices:
        # Target index for movement (default: i)
        target_idx = i

        # Boundary handling: baseline is (i-1 → i+1), measured point is i
        if i == 0:
            # Front boundary: use track prev (1 previous) for rows[0]
            if W < 2:
                continue
            row_prev = _get_track_prev(original_rows[0], 1)
            if row_prev is None:
                continue
            row_curr = original_rows[0]
            row_next = original_rows[1]
        elif i == W - 1:
            # Back boundary: use track next (1 next) for rows[W-1]
            if W < 2:
                continue
            row_prev = original_rows[W - 2]
            row_curr = original_rows[W - 1]
            row_next = _get_track_next(original_rows[W - 1], 1)
            if row_next is None:
                # fallback: use last point as baseline end even without track next
                row_next = row_curr
        else:
            row_prev = original_rows[i - 1]
            row_curr = original_rows[i]
            row_next = original_rows[i + 1]

        # Get positions
        lon_prev, lat_prev = float(row_prev['LON']), float(row_prev['LAT'])
        lon_curr, lat_curr = float(row_curr['LON']), float(row_curr['LAT'])
        lon_next, lat_next = float(row_next['LON']), float(row_next['LAT'])

        # Get LLM score for this position
        s_i = norm_scores[i] if i < len(norm_scores) else 0.0

        # Calculate target SED threshold
        target_sed = theta_g * (1 + s_i) * sed_local_clipped

        # 1. 원래 SED 계산 (baseline: i-1 → i+1, measured point: i)
        original_sed = calculate_sed(lon_prev, lat_prev, lon_next, lat_next, lon_curr, lat_curr)

        # 2. Find direction (left/right) that increases SED
        perp_bearing_left = get_cross_track_direction(lon_prev, lat_prev, lon_next, lat_next, side=1)
        perp_bearing_right = get_cross_track_direction(lon_prev, lat_prev, lon_next, lat_next, side=-1)

        test_dist = 10.0
        lon_left, lat_left = destination_point(lon_curr, lat_curr, perp_bearing_left, test_dist)
        lon_right, lat_right = destination_point(lon_curr, lat_curr, perp_bearing_right, test_dist)

        sed_left = calculate_sed(lon_prev, lat_prev, lon_next, lat_next, lon_left, lat_left)
        sed_right = calculate_sed(lon_prev, lat_prev, lon_next, lat_next, lon_right, lat_right)

        # Determine which side increases SED
        increase_side = 1 if sed_left > sed_right else -1

        # 3. Random side selection (-1 or +1)
        chosen_side = rng.choice([1, -1])
        perp_bearing = perp_bearing_left if chosen_side == 1 else perp_bearing_right

        # 4. Direct computation (direction-dependent)
        if chosen_side == increase_side:
            alpha_raw = target_sed - original_sed
        else:
            alpha_raw = target_sed + original_sed
        alpha = max(min_alpha, alpha_raw)
        alpha = min(alpha, max_alpha) # clipping max value

        # 4. Apply
        # Move from original position along normal direction
        new_lon, new_lat = destination_point(lon_curr, lat_curr, perp_bearing, alpha)
        rows[target_idx]['LON'] = str(new_lon)
        rows[target_idx]['LAT'] = str(new_lat)
        rows[target_idx]['ANOMALY'] = 'True'

    return rows
