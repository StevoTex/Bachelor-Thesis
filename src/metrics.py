from __future__ import annotations
"""
Metrics and utilities for multi-objective benchmarking.

Normalized hypervolume:
- If `bounds` (min/max) are provided, HV is computed in normalized
  minimization space: z = (f_min - lo_min) / (hi_min - lo_min).
  The reference point is normalized with the same bounds. If the bounds'
  upper limits coincide with the reference point, r_norm = 1 in each dimension.
- Points outside [lo, hi] in the original space are dropped.
- Other metrics also use the normalized space when `bounds` are given.

Notes:
- `objectives = {"f": "min"|"max", ...}` defines the orientation.
- For "max" objectives, values are mirrored to minimization space before
  normalization (f_min = -f).
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Internal helpers: bounds, cleaning, space transforms
# ============================================================================

def _apply_objective_bounds_mask(
    arr: np.ndarray,
    cols: List[str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Return a boolean mask for points inside `[lo, hi]` in the original space."""
    if not bounds:
        return np.ones(arr.shape[0], dtype=bool)

    mask = np.ones(arr.shape[0], dtype=bool)
    for j, c in enumerate(cols):
        if c in bounds and bounds[c] is not None:
            lo, hi = bounds[c]
            if lo is not None:
                mask &= arr[:, j] >= float(lo)
            if hi is not None:
                mask &= arr[:, j] <= float(hi)
    return mask


def _clean_objective_rows(
    df: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Drop invalid rows in the original space: NaN/inf, negative, and out-of-bounds."""
    if df is None or df.empty:
        return df

    cols = list(objectives.keys())
    sub = df[cols].to_numpy(dtype=float)

    finite_mask = np.isfinite(sub).all(axis=1)
    nonneg_mask = (sub >= 0.0).all(axis=1)
    bounds_mask = _apply_objective_bounds_mask(sub, cols, bounds=bounds)

    keep = finite_mask & nonneg_mask & bounds_mask
    if keep.sum() == len(df):
        return df.loc[:, cols].copy()

    return df.loc[keep, cols].reset_index(drop=True)


def _to_min_space(df: pd.DataFrame, objectives: Dict[str, str]) -> pd.DataFrame:
    """Flip sign for 'max' objectives so that all objectives are minimized."""
    out = df.copy()
    for c, s in objectives.items():
        if s.lower() == "max":
            out[c] = -out[c]
    return out


def _hv_ref_point_to_min_space(
    ref_point: Dict[str, float],
    objectives: Dict[str, str]
) -> np.ndarray:
    """Map the reference point into minimization space in `objectives` key order."""
    r = []
    for c, s in objectives.items():
        v = float(ref_point[c])
        r.append(v if s.lower() == "min" else -v)
    return np.array(r, dtype=float)


def _bounds_to_min_space(
    bounds: Dict[str, Tuple[float, float]],
    objectives: Dict[str, str],
    cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform (lo, hi) bounds into minimization space (handle 'max' by mirroring)."""
    if cols is None:
        cols = list(objectives.keys())
    lo_min, hi_min = [], []
    for c in cols:
        if c not in bounds or bounds[c] is None:
            lo_min.append(-np.inf)
            hi_min.append(np.inf)
            continue
        lo, hi = bounds[c]
        s = objectives[c].lower()
        if s == "min":
            lo_min.append(float(-np.inf) if lo is None else float(lo))
            hi_min.append(float(np.inf)  if hi is None else float(hi))
        else:
            lo_m = None if hi is None else -float(hi)
            hi_m = None if lo is None else -float(lo)
            lo_min.append(float(-np.inf) if lo_m is None else lo_m)
            hi_min.append(float(np.inf)  if hi_m is None else hi_m)
    return np.array(lo_min, dtype=float), np.array(hi_min, dtype=float)


def _filter_points_dominated_by_ref(
    F_min: np.ndarray,
    r_min: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Keep only points `<= r` in minimization space."""
    mask = np.all(F_min <= r_min[None, :], axis=1)
    return F_min[mask], bool(np.any(~mask))


# ============================================================================
# Pareto front extraction
# ============================================================================

def extract_pareto_front(
    df: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Return the non-dominated set w.r.t. `objectives`; drops out-of-bounds."""
    if df is None or df.empty:
        return df

    df = _clean_objective_rows(df, objectives, bounds=bounds)
    if df.empty:
        return df

    cols = list(objectives.keys())
    S = df[cols].to_numpy(dtype=float)
    sense = np.array([1.0 if objectives[c].lower() == "min" else -1.0 for c in cols], dtype=float)
    S_min = S * sense

    n = S_min.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        pi = S_min[i]
        le = np.all(S_min <= pi, axis=1)
        lt = np.any(S_min <  pi, axis=1)
        dom_i = le & lt
        dom_i[i] = False
        if np.any(dom_i):
            is_dominated[i] = True

    return df.loc[~is_dominated, cols].reset_index(drop=True)


# ============================================================================
# Hypervolume (exact via pygmo, otherwise MC approximation) – normalized
# ============================================================================

def _normalize_min_space_points(
    F_min: np.ndarray,
    objectives: Dict[str, str],
    bounds: Dict[str, Tuple[float, float]],
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize minimization-space points using `bounds`. Returns (Z, lo_min, hi_min)."""
    lo_min, hi_min = _bounds_to_min_space(bounds, objectives, cols)
    denom = np.maximum(hi_min - lo_min, 1e-12)
    Z = (F_min - lo_min[None, :]) / denom[None, :]
    return Z, lo_min, hi_min


def calculate_hypervolume_exact(
    pareto_df: pd.DataFrame,
    ref_point: Dict[str, float],
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Exact HV via pygmo; if `bounds` provided, compute in normalized min-space."""
    if pareto_df is None or pareto_df.empty:
        return 0.0

    df_clean = _clean_objective_rows(pareto_df, objectives, bounds=bounds)
    if df_clean.empty:
        return 0.0

    cols = list(objectives.keys())

    try:
        import pygmo as pg
        F_min = _to_min_space(df_clean[cols], objectives).to_numpy(dtype=float)

        if bounds:
            Z, lo_min, hi_min = _normalize_min_space_points(F_min, objectives, bounds, cols)
            r_min = _hv_ref_point_to_min_space(ref_point, objectives)
            denom = np.maximum(hi_min - lo_min, 1e-12)
            r = (r_min - lo_min) / denom
            F_use, _ = _filter_points_dominated_by_ref(Z, r)
            if F_use.size == 0:
                return 0.0
            hv = pg.hypervolume(F_use)
            return float(hv.compute(r))

        r = _hv_ref_point_to_min_space(ref_point, objectives)
        F_use, _ = _filter_points_dominated_by_ref(F_min, r)
        if F_use.size == 0:
            return 0.0
        hv = pg.hypervolume(F_use)
        return float(hv.compute(r))
    except Exception:
        return calculate_hypervolume_approx(pareto_df, ref_point, objectives, bounds=bounds)


def calculate_hypervolume_approx(
    pareto_df: pd.DataFrame,
    ref_point: Dict[str, float],
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_samples: int = 200_000,
    seed: Optional[int] = None,
) -> float:
    """Monte-Carlo HV in (normalized) minimization space."""
    if pareto_df is None or pareto_df.empty:
        return 0.0

    df_clean = _clean_objective_rows(pareto_df, objectives, bounds=bounds)
    if df_clean.empty:
        return 0.0

    rng = np.random.default_rng(seed)
    cols = list(objectives.keys())
    F_min = _to_min_space(df_clean[cols], objectives).to_numpy(dtype=float)

    if bounds:
        Z, lo_min, hi_min = _normalize_min_space_points(F_min, objectives, bounds, cols)
        r_min = _hv_ref_point_to_min_space(ref_point, objectives)
        denom = np.maximum(hi_min - lo_min, 1e-12)
        r = (r_min - lo_min) / denom
        F_use, _ = _filter_points_dominated_by_ref(Z, r)
        if F_use.size == 0:
            return 0.0

        lo = np.minimum(F_use.min(axis=0), r)
        hi = r
    else:
        r = _hv_ref_point_to_min_space(ref_point, objectives)
        F_use, _ = _filter_points_dominated_by_ref(F_min, r)
        if F_use.size == 0:
            return 0.0
        lo = np.minimum(F_use.min(axis=0), r)
        hi = r

    side = np.maximum(hi - lo, 0.0)
    box_vol = float(np.prod(side))
    if box_vol <= 0.0:
        return 0.0

    U = rng.random((n_samples, F_use.shape[1]))
    S = lo + U * side

    batch = 4096
    dominated = 0
    n = S.shape[0]
    for i in range(0, n, batch):
        Sb = S[i:i + batch]
        any_point = np.zeros(Sb.shape[0], dtype=bool)
        stepN = 4096
        for j in range(0, F_use.shape[0], stepN):
            Fj = F_use[j:j + stepN]
            mask = Fj[None, :, :] <= Sb[:, None, :]
            any_point |= np.all(mask, axis=2).any(axis=1)
            if np.all(any_point):
                break
        dominated += int(np.sum(any_point))

    hv_est = (dominated / n) * box_vol
    return float(hv_est)


# ============================================================================
# Normalized metrics utilities
# ============================================================================

def _normalized_points_for_metrics(
    df: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Return points in normalized minimization space; drops out-of-bounds first."""
    df = _clean_objective_rows(df, objectives, bounds=bounds)
    if df.empty:
        return np.empty((0, len(objectives)), dtype=float)

    cols = list(objectives.keys())
    X_min = _to_min_space(df[cols], objectives).to_numpy(dtype=float)

    if bounds:
        lo_min, hi_min = _bounds_to_min_space(bounds, objectives, cols)
        denom = np.maximum(hi_min - lo_min, 1e-12)
        return (X_min - lo_min[None, :]) / denom[None, :]

    mn = X_min.min(axis=0)
    mx = X_min.max(axis=0)
    denom = np.maximum(mx - mn, 1e-12)
    return (X_min - mn[None, :]) / denom[None, :]


# ============================================================================
# Bench metrics (unary)
# ============================================================================

def metric_hypervolume(
    pareto_df: pd.DataFrame,
    objectives: Dict[str, str],
    ref_point: Dict[str, float],
    hv_mode: str = "approx",
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Wrapper for HV: normalized min-space if `bounds` given, else raw min-space."""
    mode = (hv_mode or "approx").lower()
    if mode == "exact":
        return calculate_hypervolume_exact(pareto_df, ref_point, objectives, bounds=bounds)
    return calculate_hypervolume_approx(pareto_df, ref_point, objectives, bounds=bounds)


def metric_spacing(
    pareto_df: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Schott’s spacing in normalized minimization space."""
    if pareto_df is None or len(pareto_df) < 2:
        return 0.0
    P = _normalized_points_for_metrics(pareto_df, objectives, bounds=bounds)
    if P.size == 0:
        return 0.0
    n = P.shape[0]
    d = np.full(n, np.inf)
    for i in range(n):
        diff = P - P[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist[i] = np.inf
        d[i] = float(np.min(dist))
    mu = float(np.mean(d))
    if not np.isfinite(mu):
        return 0.0
    return float(math.sqrt(np.mean((d - mu) ** 2)))


def metric_delta_spread(
    pareto_df: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Coefficient of variation of nearest-neighbor distances (smaller is better)."""
    if pareto_df is None or len(pareto_df) < 2:
        return 0.0
    P = _normalized_points_for_metrics(pareto_df, objectives, bounds=bounds)
    if P.size == 0:
        return 0.0
    n = P.shape[0]
    d = np.full(n, np.inf)
    for i in range(n):
        diff = P - P[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist[i] = np.inf
        d[i] = float(np.min(dist))
    mu = float(np.mean(d))
    if mu <= 1e-12 or not np.isfinite(mu):
        return 0.0
    return float(np.std(d) / mu)


def metric_diversity_measure(
    pareto_df: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Mean per-dimension variance in normalized minimization space."""
    if pareto_df is None or pareto_df.empty:
        return 0.0
    P = _normalized_points_for_metrics(pareto_df, objectives, bounds=bounds)
    if P.size == 0:
        return 0.0
    return float(np.var(P, axis=0).mean())


def metric_maximum_spread(
    pareto_df: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Maximum pairwise L2 distance in normalized minimization space."""
    if pareto_df is None or len(pareto_df) < 2:
        return 0.0
    P = _normalized_points_for_metrics(pareto_df, objectives, bounds=bounds)
    if P.size == 0:
        return 0.0
    n = P.shape[0]
    best = 0.0
    for i in range(n):
        diff = P[i+1:] - P[i]
        if diff.size == 0:
            break
        d = np.sqrt(np.sum(diff * diff, axis=1))
        m = float(np.max(d)) if d.size else 0.0
        if m > best:
            best = m
    return best


def metric_radial_coverage(
    pareto_df: pd.DataFrame,
    objectives: Dict[str, str],
    ref_point: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Angular coverage (projection on first two objectives) after normalization if available."""
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)

    if pareto_df is None or pareto_df.empty:
        return 0.0, {"bin_edges": edges, "counts": np.zeros(n_bins, dtype=int)}

    cols = list(objectives.keys())
    df_clean = _clean_objective_rows(pareto_df, objectives, bounds=bounds)
    if df_clean.empty:
        return 0.0, {"bin_edges": edges, "counts": np.zeros(n_bins, dtype=int)}

    F_min = _to_min_space(df_clean[cols], objectives).to_numpy(dtype=float)

    if bounds:
        Z, lo_min, hi_min = _normalize_min_space_points(F_min, objectives, bounds, cols)
        r_min = _hv_ref_point_to_min_space(ref_point, objectives)
        denom = np.maximum(hi_min - lo_min, 1e-12)
        r = (r_min - lo_min) / denom
        F_use, _ = _filter_points_dominated_by_ref(Z, r)
        if F_use.size == 0 or F_use.shape[1] < 2:
            return 0.0, {"bin_edges": edges, "counts": np.zeros(n_bins, dtype=int)}
        V = r[None, :] - F_use
    else:
        r = _hv_ref_point_to_min_space(ref_point, objectives)
        F_use, _ = _filter_points_dominated_by_ref(F_min, r)
        if F_use.size == 0 or F_use.shape[1] < 2:
            return 0.0, {"bin_edges": edges, "counts": np.zeros(n_bins, dtype=int)}
        V = r[None, :] - F_use

    v2 = V[:, :2]
    mask = np.linalg.norm(v2, axis=1) > 1e-12
    v2 = v2[mask]
    if v2.size == 0:
        return 0.0, {"bin_edges": edges, "counts": np.zeros(n_bins, dtype=int)}

    ang = np.arctan2(v2[:, 1], v2[:, 0])
    counts, _ = np.histogram(ang, bins=edges)
    coverage = float(np.count_nonzero(counts) / n_bins)
    return coverage, {"bin_edges": edges, "counts": counts.astype(int)}


def metric_success_counting(
    df: pd.DataFrame,
    target_ranges: Dict[str, Tuple[float, float]],
    require_all: bool = True,
) -> Tuple[int, float]:
    """Count points inside a target box in the original space."""
    if df is None or df.empty or not target_ranges:
        return 0, 0.0

    df = _clean_objective_rows(df, {k: "min" for k in target_ranges.keys()}, bounds=None)
    if df.empty:
        return 0, 0.0

    cols = list(target_ranges.keys())
    sub = df[cols].to_numpy(dtype=float)
    ok_mat = []
    for i, c in enumerate(cols):
        lo, hi = target_ranges[c]
        ok_mat.append((sub[:, i] >= float(lo)) & (sub[:, i] <= float(hi)))
    ok_mat = np.vstack(ok_mat)
    ok = np.all(ok_mat, axis=0) if require_all else np.any(ok_mat, axis=0)
    count = int(np.count_nonzero(ok))
    frac = float(count / len(df))
    return count, frac


# ============================================================================
# Learning curves: HV over evaluations (with strict bounds)
# ============================================================================

def _infer_eval_order(df: pd.DataFrame) -> np.ndarray:
    """Heuristic for evaluation ordering (unused by the public API)."""
    candidates = ["evaluation", "eval", "step", "iteration", "iter", "t", "budget", "timestamp"]
    for c in candidates:
        if c in df.columns:
            return np.argsort(df[c].to_numpy())
    return np.arange(len(df))


def hv_curve_over_evals(
    df: pd.DataFrame,
    objectives: Dict[str, str],
    ref_point: Dict[str, float],
    hv_mode: str = "approx",
    stride: int = 5,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    exclude_phases: Tuple[str, ...] = ("update",),
) -> pd.Series:
    """HV(t) over valid evaluations, robust to duplicates per `evaluation` id.

    Index is 1..N_valid (not raw evaluation ids). Phases listed in
    `exclude_phases` (e.g., 'update') are filtered out if present.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float, name="HV")

    cols = list(objectives.keys())

    if "phase" in df.columns and exclude_phases:
        df = df[~df["phase"].astype(str).str.lower().isin([p.lower() for p in exclude_phases])].copy()

    if not set(cols).issubset(df.columns):
        return pd.Series(dtype=float, name="HV")

    X = df[cols].to_numpy(dtype=float, copy=True)
    finite_mask  = np.isfinite(X).all(axis=1)
    nonneg_mask  = (X >= 0.0).all(axis=1)
    bounds_mask  = _apply_objective_bounds_mask(X, cols, bounds=bounds)
    keep = finite_mask & nonneg_mask & bounds_mask
    if not np.any(keep):
        return pd.Series(dtype=float, name="HV")

    if "evaluation" in df.columns:
        work = df.loc[keep, ["evaluation"] + cols].copy()
        work["evaluation"] = pd.to_numeric(work["evaluation"], errors="coerce")
        work = work.dropna(subset=["evaluation"])
        work = work.sort_values("evaluation").drop_duplicates(subset="evaluation", keep="last")
        valid_df = work[cols].reset_index(drop=True)
    else:
        valid_df = df.loc[keep, cols].reset_index(drop=True)

    n_valid = len(valid_df)
    if n_valid == 0:
        return pd.Series(dtype=float, name="HV")

    hv_vals: List[float] = []
    steps:   List[int]   = []
    for k in range(1, n_valid + 1):
        if (k % max(1, stride) != 0) and (k != n_valid):
            continue
        front = extract_pareto_front(valid_df.iloc[:k], objectives, bounds=bounds)
        hv = metric_hypervolume(front, objectives, ref_point, hv_mode=hv_mode, bounds=bounds)
        hv_vals.append(hv)
        steps.append(k)

    return pd.Series(hv_vals, index=steps, name="HV")


def evals_to_reach_fraction(
    df: pd.DataFrame,
    objectives: Dict[str, str],
    ref_point: Dict[str, float],
    hv_mode: str = "approx",
    frac: float = 0.95,
    stride: int = 5,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    exclude_phases: Tuple[str, ...] = ("update",),
) -> int:
    """Smallest number of valid evaluations where HV ≥ `frac` × final HV."""
    curve = hv_curve_over_evals(
        df, objectives, ref_point, hv_mode=hv_mode, stride=stride, bounds=bounds,
        exclude_phases=exclude_phases
    )
    if curve.empty:
        return 0
    final_hv = float(curve.iloc[-1])
    target = frac * final_hv
    hit = curve[curve >= target]
    return int(hit.index[0]) if not hit.empty else int(len(curve))


# ============================================================================
# Binary epsilon indicator (multiplicative) – optional, bounds-aware
# ============================================================================

def epsilon_indicator(
    A: pd.DataFrame,
    B: pd.DataFrame,
    objectives: Dict[str, str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Multiplicative epsilon indicator in the original space with bounds filtering."""
    cols = list(objectives.keys())
    A_ = _clean_objective_rows(A[cols], objectives, bounds=bounds)
    B_ = _clean_objective_rows(B[cols], objectives, bounds=bounds)
    if A_.empty or B_.empty:
        return float("inf")

    Amin = _to_min_space(A_, objectives).to_numpy(dtype=float)
    Bmin = _to_min_space(B_, objectives).to_numpy(dtype=float)

    eps = 1e-12
    Bsafe = np.maximum(Bmin, eps)

    worst = -np.inf
    for b in Bsafe:
        ratios = Amin / b[None, :]
        max_per_a = np.max(ratios, axis=1)
        best_for_b = float(np.min(max_per_a))
        if best_for_b > worst:
            worst = best_for_b
    return float(max(worst, 1.0))


# ============================================================================
# Small helpers for reporting
# ============================================================================

def valid_rate(
    points: Iterable[Tuple[float, ...]],
    target_box: Dict[str, Tuple[float, float]],
    keys: Optional[List[str]] = None
) -> float:
    """Fraction of points inside a (lo, hi) box in the original space."""
    pts = list(points)
    if len(pts) == 0:
        return 0.0
    if keys is None:
        keys = list(target_box.keys())
    ok = 0
    for p in pts:
        inside = True
        for (k, v) in zip(keys, p):
            lo, hi = target_box[k]
            if not (lo <= v <= hi):
                inside = False
                break
        ok += 1 if inside else 0
    return ok / len(pts)


__all__ = [
    # Front & HV
    "extract_pareto_front",
    "calculate_hypervolume_exact",
    "calculate_hypervolume_approx",
    "metric_hypervolume",
    # Curves
    "hv_curve_over_evals",
    "evals_to_reach_fraction",
    # Bench metrics
    "metric_spacing",
    "metric_delta_spread",
    "metric_diversity_measure",
    "metric_maximum_spread",
    "metric_radial_coverage",
    "metric_success_counting",
    # Binary indicator
    "epsilon_indicator",
    # Helpers
    "valid_rate",
]
