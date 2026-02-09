"""
KRISO2026 - A2 Anomaly Injector
================================
A2: Speed and heading anomaly based on acceleration/angular velocity.

"평소 가속도의 theta_v * (1 + s_i) 배만큼 갑자기 빨라지거나 느려짐"

Algorithm:
1. Calculate local acceleration baseline: a_local = mean(|a_i|)
2. Calculate local angular velocity baseline: omega_local = mean(|omega_i|)
3. For each i in top-K (first 2/3 injection, last 1/3 recovery):
   - v_i^fake = v_{i-1} + theta_v * (1 + s_i) * a_local * dt
   - h_i^fake = h_{i-1} + theta_v * (1 + s_i) * omega_local
4. Final position: p_i^fake = p_{i-1} + v_i^fake * dt * (cos(h_i^fake), sin(h_i^fake))
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from inject_utils import (
    InjectionPlan,
    bearing,
    calculate_a_local,
    calculate_omega_local,
    calculate_headings,
    calculate_velocities,
    destination_point,
    haversine_distance,
    heading_from_sin_cos,
    minmax_normalize_scores,
    wrap_angle,
    EPSILON_A,
    EPSILON_OMEGA,
    THETA_V,
)


def inject_a2(
    rows: List[dict],
    plan: InjectionPlan,
    rng: Optional[random.Random] = None,
    theta_v: float = THETA_V,
    epsilon_a: float = EPSILON_A,
    epsilon_omega: float = EPSILON_OMEGA,
    dt_seconds: float = 3600.0,
    track_index: Optional[dict] = None,
    track_pos: Optional[dict] = None,
) -> List[dict]:
    """
    Inject A2 anomaly (speed and heading perturbation).

    Args:
        rows: List of route point dictionaries
        plan: InjectionPlan with scores and K
        rng: Random number generator for sign selection
        theta_v: Speed/heading multiplier (default 3.0)
        epsilon_a: Minimum a_local clipping value (default Q10=0.0001 m/s²)
        epsilon_omega: Minimum omega_local clipping value (default 3°)
        dt_seconds: Time interval in seconds (default 3600 = 1 hour)

    Returns:
        List of modified route point dictionaries
    """
    if rng is None:
        rng = random.Random()

    normal_rows = [dict(r) for r in rows]  # Keep original (normal) trajectory
    rows = [dict(r) for r in rows]  # Deep copy for injection
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

    # 1. Get K consecutive indices centered around top-1 score
    # (A2 anomalies must be in consecutive timestamps)
    scores = np.array(plan.scores)
    norm_scores = minmax_normalize_scores(plan.scores)
    T = len(scores)
    K = min(plan.K, T)

    top1_idx = int(np.argmax(scores))

    # Calculate window: K consecutive indices centered around top1
    half_left = (K - 1) // 2
    half_right = K - 1 - half_left

    start = top1_idx - half_left
    end = top1_idx + half_right

    # Adjust if out of bounds
    if start < 0:
        start = 0
        end = min(K - 1, T - 1)
    if end >= T:
        end = T - 1
        start = max(0, T - K)

    # Must be sorted for sequential injection (each point depends on previous)
    top_k_indices = list(range(start, end + 1))

    # 2. Calculate local baselines (clipped)
    a_local = calculate_a_local(rows, dt_seconds)
    a_local_clipped = max(a_local, epsilon_a)

    omega_local = calculate_omega_local(rows)
    omega_local_clipped = max(omega_local, epsilon_omega)

    # 3. Get current velocities and headings
    velocities = calculate_velocities(rows, dt_seconds)  # m/s, length W-1
    headings = calculate_headings(rows)  # degrees, length W

    # 4. Acceleration-only: always speed up during anomaly window
    sign_a = 1
    sign_omega = rng.choice([1, -1])

    # 5. For each index in consecutive window, inject anomaly (in order)
    #    First 2/3 = injection, last 1/3 = recovery (smooth return to normal).
    k_window = len(top_k_indices)
    inject_len = max(1, int((2 * k_window) // 3)) if k_window > 0 else 0
    recover_len = max(0, k_window - inject_len)
    t_e = top_k_indices[inject_len - 1] if inject_len > 0 else -1
    target_idx = min(t_e + recover_len + 1, W - 1) if t_e >= 0 else 0

    # Pre-compute score-weighted cumulative f for recovery phase.
    # Each recovery step's "share" of total travel is proportional to (1 + score).
    # The final jump to target also gets weight 1.0 (no score).
    # Cumulative sums / total → f values that sum to 1.0 at target.
    _recover_cum_f: List[float] = []
    if recover_len > 0:
        _recover_indices = top_k_indices[inject_len:]          # recovery point indices
        _recover_weights = [
            1.0 + (norm_scores[idx] if idx < len(norm_scores) else 0.0)
            for idx in _recover_indices
        ] + [1.0]                                              # +1 for the final jump to target
        _total_w = sum(_recover_weights)
        _cum = 0.0
        for _w in _recover_weights[:-1]:                       # exclude the last (target) weight
            _cum += _w
            _recover_cum_f.append(_cum / _total_w)

    for pos, i in enumerate(top_k_indices):

        # Get LLM score for this position
        s_i = norm_scores[i] if i < len(norm_scores) else 0.0

        # Calculate multiplier
        multiplier = theta_v * (1 + s_i)

        # Get previous velocity/heading (use track context for i==0)
        if i >= 1:
            if i - 1 < len(velocities):
                v_prev = velocities[i - 1]
            else:
                v_prev = velocities[-1] if velocities else 5.0
            h_prev = headings[i - 1]
            lon_prev = float(rows[i - 1]['LON'])
            lat_prev = float(rows[i - 1]['LAT'])
        else:
            prev_row = _get_track_prev(rows[i], 1)
            if prev_row is None:
                continue
            lon_prev = float(prev_row['LON'])
            lat_prev = float(prev_row['LAT'])
            v_prev = haversine_distance(
                lon_prev, lat_prev,
                float(rows[i]['LON']), float(rows[i]['LAT'])
            ) / dt_seconds
            h_prev = heading_from_sin_cos(
                float(prev_row['COURSE_SIN']),
                float(prev_row['COURSE_COS'])
            )

        if pos < inject_len:
            # Injection phase
            delta_v = sign_a * multiplier * a_local_clipped * dt_seconds
            v_fake = max(0.1, v_prev + delta_v)

            delta_h = sign_omega * multiplier * omega_local_clipped
            h_fake = wrap_angle(h_prev + delta_h)

            distance_m = v_fake * dt_seconds
            bearing_deg = (h_fake + 360) % 360
            new_lon, new_lat = destination_point(lon_prev, lat_prev, bearing_deg, distance_m)
        else:
            # Recovery phase: smoothly return toward normal trajectory target
            if t_e >= 0:
                lon_start = float(rows[t_e]['LON'])
                lat_start = float(rows[t_e]['LAT'])
            else:
                lon_start = float(rows[i - 1]['LON'])
                lat_start = float(rows[i - 1]['LAT'])

            lon_target = float(normal_rows[target_idx]['LON'])
            lat_target = float(normal_rows[target_idx]['LAT'])

            # Score-weighted recovery: high logit → covers more distance this step
            recover_step = pos - inject_len          # 0-based index into _recover_cum_f
            f = _recover_cum_f[recover_step] if recover_step < len(_recover_cum_f) else 1.0
            new_lon = lon_start + (lon_target - lon_start) * f
            new_lat = lat_start + (lat_target - lat_start) * f

            # Gradually rotate COG toward target direction
            if i >= 1:
                lon_prev = float(rows[i - 1]['LON'])
                lat_prev = float(rows[i - 1]['LAT'])
            else:
                prev_row = _get_track_prev(rows[i], 1)
                if prev_row is None:
                    continue
                lon_prev = float(prev_row['LON'])
                lat_prev = float(prev_row['LAT'])

            bearing_to_target = bearing(lon_prev, lat_prev, lon_target, lat_target)
            h_fake = wrap_angle(h_prev + (wrap_angle(bearing_to_target - h_prev)) * f)

            # Mild speed adjustment toward reaching target without sharp changes
            dist_remain = haversine_distance(lon_prev, lat_prev, lon_target, lat_target)
            steps_left = max(1, recover_len - recover_step)
            desired_speed = dist_remain / (steps_left * dt_seconds)
            v_fake = max(0.1, 0.7 * v_prev + 0.3 * desired_speed)
            # Keep acceleration-only behavior in recovery as well
            v_fake = max(v_prev, v_fake)

        # Update row
        rows[i]['LON'] = str(new_lon)
        rows[i]['LAT'] = str(new_lat)
        rows[i]['ANOMALY'] = 'True'

        # Update SPEED column (convert m/s to knots)
        speed_knots = v_fake / 0.514444
        rows[i]['SPEED'] = str(speed_knots)

        # Update COURSE_SIN and COURSE_COS
        h_rad = np.radians(h_fake)
        rows[i]['COURSE_SIN'] = str(np.sin(h_rad))
        rows[i]['COURSE_COS'] = str(np.cos(h_rad))

        # Update velocities and headings for next iteration
        if i == 0 and len(velocities) > 0:
            # Update segment speed between row0 and row1 using injected row0
            next_lon = float(rows[i + 1]['LON'])
            next_lat = float(rows[i + 1]['LAT'])
            velocities[0] = haversine_distance(new_lon, new_lat, next_lon, next_lat) / dt_seconds
        elif i - 1 < len(velocities):
            velocities[i - 1] = v_fake
        headings[i] = h_fake

    return rows
