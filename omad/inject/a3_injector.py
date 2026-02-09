"""
OMAD - A3 Anomaly Injector
================================
A3: Close-approach anomaly (virtual vessel generation).

Creates a virtual vessel trajectory that closely approaches the original vessel.
The virtual vessel's closest approach point (CPA) is determined by LLM scores.

Based on: a3_close_approach_realizer.py
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from inject_utils import A3Plan, load_a3_plans, minmax_normalize_scores

# =========================
# Constants
# =========================
R_EARTH = 6_371_000.0         # meters
KNOT_TO_MPS = 0.514444        # 1 knot = 0.514444 m/s
NM_TO_M = 1852.0              # 1 nautical mile = 1852 m
DT_SECONDS = 3600.0           # 1 hour sampling


@dataclass(frozen=True)
class InjectMeta:
    """Injection metadata for debugging."""
    T: int
    anchor_idx: int
    risk_range: Tuple[int, int]
    D_star_nm: float
    side: str


# =========================
# Coordinate Utilities
# =========================
def _latlon_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> np.ndarray:
    """Convert lat/lon to local ENU (meters)."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    e = (lon_r - lon0_r) * np.cos(lat0_r) * R_EARTH
    n = (lat_r - lat0_r) * R_EARTH
    return np.array([e, n], dtype=float)


def _enu_to_latlon(e: float, n: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Convert local ENU (meters) to lat/lon."""
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    lat = lat0_r + n / R_EARTH
    lon = lon0_r + e / (R_EARTH * np.cos(lat0_r))
    return float(np.degrees(lat)), float(np.degrees(lon))


def _cog_sog_to_velocity(cog_deg: float, sog_knots: float) -> np.ndarray:
    """Convert COG/SOG to ENU velocity vector (m/s)."""
    cog = np.radians(float(cog_deg))
    speed = max(float(sog_knots), 0.2) * KNOT_TO_MPS
    ve = speed * np.sin(cog)
    vn = speed * np.cos(cog)
    return np.array([ve, vn], dtype=float)


def _norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def _left_perp_unit(v, eps: float = 1e-12) -> np.ndarray:
    """Perpendicular unit vector (left side)."""
    v = np.asarray(v, dtype=float)
    p = np.array([-v[1], v[0]], dtype=float)
    n = _norm(p)
    if n < eps:
        return np.array([1.0, 0.0])
    return p / n


def _rotate_unit(u2: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate 2D vector by theta degrees CCW."""
    u2 = np.asarray(u2, dtype=float)
    th = np.radians(float(theta_deg))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=float)
    v = R @ u2
    n = _norm(v)
    if n > 1e-12:
        v = v / n
    return v


# =========================
# Score Processing
# =========================
def _pick_anchor_from_scores(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Pick anchor index from argmax of scores."""
    scores = np.asarray(scores, dtype=float)
    max_val = float(np.max(scores))
    cand = np.flatnonzero(np.isclose(scores, max_val, rtol=0.0, atol=1e-12))
    return int(rng.choice(cand))


def _fixed_risk_window(
    T: int,
    t_anchor: int,
    m_fixed: int = 3,
    scores: Optional[np.ndarray] = None,
) -> Tuple[set, int, int]:
    """Determine risk window containing anchor."""
    m = min(m_fixed, T)

    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        s0_min = max(0, t_anchor - (m - 1))
        s0_max = min(t_anchor, T - m)

        best_s0 = s0_min
        best_sum = -1e18
        for s0_cand in range(s0_min, s0_max + 1):
            s = float(np.sum(scores[s0_cand:s0_cand + m]))
            if s > best_sum:
                best_sum = s
                best_s0 = s0_cand

        s0, s1 = best_s0, best_s0 + m - 1
    else:
        left = (m - 1) // 2
        s0 = max(0, t_anchor - left)
        s1 = min(T - 1, s0 + m - 1)
        if s1 >= T:
            s1 = T - 1
            s0 = max(0, s1 - m + 1)

    return set(range(s0, s1 + 1)), s0, s1


def _map_severity_to_Dstar_nm(sev01: float, D_range: Tuple[float, float] = (0.1, 0.3)) -> float:
    """Map severity [0,1] to D* in nautical miles. Higher severity = closer approach."""
    dmin, dmax = D_range
    sev01 = float(np.clip(sev01, 0.0, 1.0))
    return float(dmax - (dmax - dmin) * sev01)


def _build_cpa_distance_profile(
    T: int,
    t_a: int,
    D_star_m: float,
    u_norm_mps: float,
    dt_s: float,
) -> np.ndarray:
    """Build CPA-like distance profile: d(t) = sqrt(D*^2 + (|u|*dt*(t-t_a))^2)."""
    d = np.zeros(T, dtype=float)
    for t in range(T):
        tau = float((t - t_a) * dt_s)
        d[t] = np.sqrt(D_star_m ** 2 + (u_norm_mps * tau) ** 2)
    return d


def _enu_course_speed_from_positions(
    p_enu: np.ndarray,
    dt_s: float = DT_SECONDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate course and speed from ENU positions."""
    T = p_enu.shape[0]
    d = np.diff(p_enu, axis=0)
    speed_mps = np.linalg.norm(d, axis=1) / dt_s
    speed_knots = speed_mps / KNOT_TO_MPS
    course = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360.0) % 360.0
    # Extend to length T
    speed_knots = np.concatenate([speed_knots, speed_knots[-1:]])
    course = np.concatenate([course, course[-1:]])
    return course, speed_knots


# =========================
# Virtual Trajectory Generation
# =========================
def _generate_virtual_trajectory(
    rows: List[dict],
    scores: np.ndarray,
    rng: np.random.Generator,
    m_fixed: int = 3,
    D_star_nm_range: Tuple[float, float] = (0.1, 0.3),
) -> Tuple[List[dict], InjectMeta]:
    """Generate virtual vessel trajectory for close-approach scenario."""
    T = len(rows)

    # 1. Pick anchor from scores
    t_a = _pick_anchor_from_scores(scores, rng)

    # 2. Determine risk window
    S, s0, s1 = _fixed_risk_window(T, t_a, m_fixed, scores)

    # 3. Reference point (anchor lat/lon)
    lat0 = float(rows[t_a]['LAT'])
    lon0 = float(rows[t_a]['LON'])

    # 4. Convert own positions to ENU
    p_o = np.stack([
        _latlon_to_enu(float(rows[i]['LAT']), float(rows[i]['LON']), lat0, lon0)
        for i in range(T)
    ], axis=0)

    # 5. Own anchor velocity
    sog_o = float(rows[t_a]['SPEED'])
    cog_o = np.degrees(np.arctan2(
        float(rows[t_a]['COURSE_SIN']),
        float(rows[t_a]['COURSE_COS'])
    ))
    v_o = _cog_sog_to_velocity(cog_o, sog_o)

    # 6. Virtual anchor velocity (slightly different speed)
    scale = rng.uniform(0.9, 1.1)
    v_v = scale * v_o
    u = v_v - v_o
    u_norm = max(_norm(u), 0.05)

    # 7. Select perpendicular direction (left or right)
    side = "left" if rng.random() < 0.5 else "right"
    n_perp = _left_perp_unit(u if _norm(u) > 1e-6 else v_o)
    if side == "right":
        n_perp = -n_perp

    # 8. Calculate severity and D*
    s_clip = np.clip(scores, 0.0, 1.0)
    win_idx = np.array(sorted(list(S)), dtype=int)
    sev = float(0.6 * np.mean(s_clip[win_idx]) + 0.4 * s_clip[t_a])
    sev = float(np.clip(sev, 0.0, 1.0))
    D_star_nm = _map_severity_to_Dstar_nm(sev, D_star_nm_range)
    D_star_m = D_star_nm * NM_TO_M

    # 9. Build distance profile
    d_profile = _build_cpa_distance_profile(T, t_a, D_star_m, u_norm, DT_SECONDS)

    # 10. Adjust distances outside risk window
    D_edge_m = min(2.0 * NM_TO_M, max(D_star_m * 2.0, D_star_m + 0.2 * NM_TO_M))
    for t in range(T):
        if t not in S:
            k = (s0 - t) if t < s0 else (t - s1)
            d_profile[t] = max(d_profile[t], D_edge_m + k * 0.8 * NM_TO_M)
            d_profile[t] *= 3.0  # outer_scale

    # Ensure anchor is closest
    for t in range(s0, s1 + 1):
        if t == t_a:
            d_profile[t] = D_star_m
        else:
            d_profile[t] = max(d_profile[t], D_star_m + 5.0)

    # 11. Build heading variation profile
    theta_max = 15.0 + (70.0 - 15.0) * sev
    theta = np.zeros(T, dtype=float)
    for t in range(T):
        w = max(0.0, 1.0 - abs(t - t_a) / max(2, T // 6))
        theta[t] = theta_max * w * (1.0 if side == "left" else -1.0)

    # 12. Calculate normal vectors with rotation
    n_t = np.stack([_rotate_unit(n_perp, theta[t]) for t in range(T)], axis=0)

    # 13. Virtual positions
    r = d_profile[:, None] * n_t
    p_v = p_o + r

    # 14. Convert back to lat/lon and calculate course/speed
    course_v, speed_v = _enu_course_speed_from_positions(p_v)

    virtual_rows = []
    for i in range(T):
        lat, lon = _enu_to_latlon(p_v[i, 0], p_v[i, 1], lat0, lon0)
        row = dict(rows[i])
        row['LAT'] = str(lat)
        row['LON'] = str(lon)
        row['SPEED'] = str(speed_v[i])
        row['COURSE_SIN'] = str(np.sin(np.radians(course_v[i])))
        row['COURSE_COS'] = str(np.cos(np.radians(course_v[i])))
        row['ANOMALY'] = 'True' if i in S else 'False'
        virtual_rows.append(row)

    meta = InjectMeta(
        T=T,
        anchor_idx=t_a,
        risk_range=(s0, s1),
        D_star_nm=D_star_nm,
        side=side,
    )

    return virtual_rows, meta


# =========================
# Public API
# =========================
def inject_a3_virtual_rows(
    rows: List[dict],
    route_id: int,
    scores: List[float],
    seed: int = 42,
) -> List[dict]:
    """
    Generate A3 virtual vessel rows for close-approach scenario.

    Args:
        rows: Original route point dictionaries
        route_id: Original route ID (virtual will use negative)
        scores: LLM scores for each timestep
        seed: Random seed

    Returns:
        List of virtual vessel row dictionaries with negative ROUTE_ID
    """
    if len(rows) < 3:
        return []

    if len(scores) != len(rows):
        return []

    # Create deterministic seed from route_id
    seed_str = f"{route_id}_{seed}"
    h = hashlib.md5(seed_str.encode()).hexdigest()
    rng_seed = int(h, 16) % (2 ** 32)
    rng = np.random.default_rng(rng_seed)

    scores_arr = np.asarray(minmax_normalize_scores(scores), dtype=float)

    try:
        virtual_rows, meta = _generate_virtual_trajectory(rows, scores_arr, rng)
    except Exception:
        return []

    # Assign negative route ID and update metadata
    virtual_route_id = -abs(route_id)
    for i, row in enumerate(virtual_rows):
        row['ROUTE_ID'] = str(virtual_route_id)
        row['ROUTE_POINT_IDX'] = str(i)
        row['ANOMALY_TYPE'] = 'A3' if row.get('ANOMALY') == 'True' else ''

    return virtual_rows


# Re-export for compatibility
__all__ = ['inject_a3_virtual_rows', 'load_a3_plans', 'A3Plan']
