"""Objective utilities and rewards for multi-objective vehicle optimization.

Includes:
- Target-box helpers (membership, distance, score).
- Heuristic and Tchebycheff rewards.
- A unified `rl_reward` interface (+ `is_terminal`).
- NSGA-compatible reward: rank + normalized crowding distance.

Default convention: all objectives are minimized.
"""
from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional, List
import numpy as np

OBJECTIVES_DEFAULT = {"consumption": "min", "ela3": "min", "ela4": "min", "ela5": "min"}

_DEFAULT_S_HAT_INV = (1.0 / 0.7, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 11.0)
_KEYS = ("consumption", "ela3", "ela4", "ela5")


def sanitize_sim(sim: Tuple[float, float, float, float]) -> bool:
    """Return True iff values are not NaN and strictly positive (<= 0 is invalid)."""
    c, e3, e4, e5 = sim
    if any(np.isnan([c, e3, e4, e5])):
        return False
    if c <= 0 or e3 <= 0 or e4 <= 0 or e5 <= 0:
        return False
    return True


def _ranges_to_arrays(target_ranges: Dict[str, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert range dict to (mins, maxs) arrays in fixed key order."""
    mins, maxs = [], []
    for k in _KEYS:
        lo, hi = target_ranges[k]
        mins.append(float(lo))
        maxs.append(float(hi))
    return np.asarray(mins, dtype=float), np.asarray(maxs, dtype=float)


def in_target_box(
    sim: Tuple[float, float, float, float],
    target_ranges: Dict[str, Sequence[float]],
    require_all: bool = True,
    min_satisfied: Optional[int] = None,
) -> bool:
    """Return True if `sim` lies inside the target hyper-rectangle.

    If `require_all` is False, at least `min_satisfied` objectives must be inside.
    """
    y = np.asarray(sim, dtype=float)
    mins, maxs = _ranges_to_arrays(target_ranges)
    inside_each = (y >= mins) & (y <= maxs)
    if require_all:
        return bool(np.all(inside_each))
    thr = int(min_satisfied or len(_KEYS))
    return bool(np.sum(inside_each) >= thr)


def box_distance(
    sim: Tuple[float, float, float, float],
    target_ranges: Dict[str, Sequence[float]],
    s_hat_inv: Sequence[float] | None = None,
    norm: str = "l2",
) -> Tuple[float, float]:
    """Distance of `sim` to the target box.

    Returns:
        (box_dist, box_dist_scaled)
        - `box_dist`: L2 (or L1) distance to the projection onto the box.
        - `box_dist_scaled`: same after elementwise scaling by `s_hat_inv`.
        Inside the box both distances are 0.
    """
    y = np.asarray(sim, dtype=float)
    mins, maxs = _ranges_to_arrays(target_ranges)
    proj = np.clip(y, mins, maxs)
    diff = y - proj

    if norm.lower() in ("l1", "manhattan"):
        d_raw = float(np.sum(np.abs(diff)))
    else:
        d_raw = float(np.linalg.norm(diff, ord=2))

    s_hat_inv_arr = np.asarray(s_hat_inv if s_hat_inv is not None else _DEFAULT_S_HAT_INV, dtype=float)
    diff_scaled = diff * s_hat_inv_arr
    if norm.lower() in ("l1", "manhattan"):
        d_scaled = float(np.sum(np.abs(diff_scaled)))
    else:
        d_scaled = float(np.linalg.norm(diff_scaled, ord=2))

    return d_raw, d_scaled


def score_target_box(
    sim: Tuple[float, float, float, float],
    target_ranges: Dict[str, Sequence[float]],
    s_hat_inv: Sequence[float] | None = None,
    norm: str = "l2",
    eps: float = 1e-12,
) -> float:
    """Reward based on scaled distance to the target box: r = 1 / (eps + ||...||).

    Inside the box (distance 0) the score approaches 1/eps; no early stopping here.
    """
    if not sanitize_sim(sim):
        return 0.0
    _, d_scaled = box_distance(sim, target_ranges, s_hat_inv=s_hat_inv, norm=norm)
    return 1.0 / max(eps, d_scaled)


def reward_heuristic(
    sim: Tuple[float, float, float, float],
    w: Dict[str, float],
    invalid_penalty: float = -10.0,
) -> float:
    """Negative weighted sum (larger is better); returns `invalid_penalty` if invalid."""
    if not sanitize_sim(sim):
        return float(invalid_penalty)
    c, e3, e4, e5 = sim
    w_c = float(w.get("w_consumption", 1.0))
    w_e = float(w.get("w_elas_sum", 0.1))
    return float(-(w_c * c + w_e * (e3 + e4 + e5)))


def to_minimization(
    phen: Tuple[float, float, float, float],
    objectives: Dict[str, str] = OBJECTIVES_DEFAULT,
) -> Tuple[float, ...]:
    """Convert to minimization orientation by negating 'max' objectives."""
    out = []
    for k, v in zip(_KEYS, phen):
        out.append(float(v) if objectives.get(k, "min") == "min" else float(-v))
    return tuple(out)


_def_ranges = {
    "consumption": (3.0, 15.0),
    "ela3": (0.0, 10.0),
    "ela4": (0.0, 10.0),
    "ela5": (0.0, 10.0),
}


def _norm(val: float, lo: float, hi: float) -> float:
    """Linear normalization to [0, 1] over [lo, hi]."""
    return (val - lo) / max(1e-12, hi - lo)


def reward_tchebycheff(
    sim: Tuple[float, float, float, float],
    weights: Sequence[float] | None,
    ranges: Dict[str, Tuple[float, float]] | None = None,
    rho: float = 1e-3,
    invalid_penalty: float = -10.0,
    objectives: Dict[str, str] = OBJECTIVES_DEFAULT,
) -> float:
    """Augmented Tchebycheff scalarization (returned as negative so higher is better)."""
    if not sanitize_sim(sim):
        return float(invalid_penalty)
    r = ranges or _def_ranges
    g = to_minimization(sim, objectives)
    gtil = [_norm(gi, r[k][0], r[k][1]) for gi, k in zip(g, _KEYS)]
    d = len(gtil)
    if not weights:
        w = np.ones(d) / d
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or len(w) != d:
            w = np.ones(d) / d
        s = w.sum()
        w = w / s if s > 0 else np.ones(d) / d
    max_term = np.max(w * np.asarray(gtil))
    aug = max_term + rho * np.sum(w * np.asarray(gtil))
    return float(-aug)


def rl_reward(sim: Tuple[float, float, float, float], cfg: Dict[str, object]) -> float:
    """Unified reward interface: target-box, Tchebycheff, or heuristic."""
    t = str(cfg.get("type", "heuristic")).lower()
    invalid = float(cfg.get("invalid_penalty", -10.0))

    if t in ("target_box", "box", "box_distance"):
        tr = cfg.get("target_ranges")
        if not isinstance(tr, dict):
            return reward_heuristic(
                sim,
                w=cfg.get("heuristic_weights", {"w_consumption": 1.0, "w_elas_sum": 0.1}),
                invalid_penalty=invalid,
            )
        return score_target_box(
            sim,
            target_ranges=tr,
            s_hat_inv=cfg.get("s_hat_inv", _DEFAULT_S_HAT_INV),
            norm=str(cfg.get("norm", "l2")),
            eps=float(cfg.get("eps", 1e-12)),
        )

    if t == "tchebycheff":
        return reward_tchebycheff(
            sim,
            weights=cfg.get("tcheby_weights"),
            ranges=cfg.get("ranges"),
            rho=float(cfg.get("rho", 1e-3)),
            invalid_penalty=invalid,
            objectives=cfg.get("objectives", OBJECTIVES_DEFAULT),
        )

    return reward_heuristic(
        sim, w=cfg.get("heuristic_weights", {"w_consumption": 1.0, "w_elas_sum": 0.1}), invalid_penalty=invalid
    )


def is_terminal(sim: Tuple[float, float, float, float], cfg: Dict[str, object]) -> bool:
    """Return True iff the target box is satisfied (only for type 'target_box')."""
    t = str(cfg.get("type", "heuristic")).lower()
    if t in ("target_box", "box", "box_distance"):
        tr = cfg.get("target_ranges")
        if not isinstance(tr, dict):
            return False
        return in_target_box(
            sim,
            target_ranges=tr,
            require_all=bool(cfg.get("require_all", True)),
            min_satisfied=cfg.get("min_satisfied", None),
        )
    return False


def _nsga_obj_keys(objectives: Dict[str, str]) -> List[str]:
    """Return objective keys preserving the dict order."""
    return list(objectives.keys())


def _nsga_to_min_orientation(P: np.ndarray, objectives: Dict[str, str], keys: List[str]) -> np.ndarray:
    """Flip columns corresponding to 'max' objectives so that all are minimized."""
    Pm = P.copy()
    for i, k in enumerate(keys):
        if str(objectives.get(k, "min")).lower() == "max":
            Pm[:, i] *= -1.0
    return Pm


def _nsga_dominates_min(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b under minimization."""
    return bool(np.all(a <= b) and np.any(a < b))


def _nsga_fast_non_dominated_sort_min(Pm: np.ndarray) -> List[List[int]]:
    """Fast non-dominated sort on minimization-oriented points."""
    n = Pm.shape[0]
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _nsga_dominates_min(Pm[p], Pm[q]):
                S[p].append(q)
            elif _nsga_dominates_min(Pm[q], Pm[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        if not next_front:
            break
        fronts.append(next_front)

    return fronts


def _nsga_crowding_distance_min(Pm: np.ndarray, front_idx: List[int], eps: float = 1e-12) -> np.ndarray:
    """Raw crowding distance for a given front (minimization)."""
    cd = np.zeros(Pm.shape[0], dtype=float)
    if not front_idx:
        return cd
    if len(front_idx) <= 2:
        cd[front_idx] = np.inf
        return cd

    m = Pm.shape[1]
    F = np.array(front_idx, dtype=int)
    for j in range(m):
        vals = Pm[F, j]
        order = np.argsort(vals)
        F_sorted = F[order]
        vmin, vmax = float(vals[order[0]]), float(vals[order[-1]])
        cd[F_sorted[0]] = np.inf
        cd[F_sorted[-1]] = np.inf
        if vmax > vmin + eps:
            denom = vmax - vmin + eps
            for k in range(1, len(F_sorted) - 1):
                prev_v = Pm[F_sorted[k - 1], j]
                next_v = Pm[F_sorted[k + 1], j]
                cd[F_sorted[k]] += (next_v - prev_v) / denom
    return cd


def nsga_rank_cd_reward(
    archive_points,
    y: Sequence[float] | Dict[str, float],
    objectives: Dict[str, str],
    *,
    w_r: float = 0.7,
    w_c: float | None = None,
    eps: float = 1e-12,
) -> Tuple[float, int, float, float]:
    """NSGA-compatible reward without a reference point: r = w_r*(1/rank) + w_c*CD_tilde.

    Returns:
        (reward, rank, cd_raw, cd_tilde) for the newly added point.
        - `rank`: 1-based from fast non-dominated sort.
        - `CD_tilde`: normalized crowding distance on the first front (others → 0).
          Normalization: finite CD / max(finite CD) in front 1; boundary points (inf) → 1.0.
    """
    keys = _nsga_obj_keys(objectives)

    if hasattr(archive_points, "to_numpy"):
        A = archive_points[keys].to_numpy(dtype=float, copy=False)
    else:
        A = np.asarray(archive_points, dtype=float)
        if A.size and (A.ndim != 2 or A.shape[1] != len(keys)):
            raise ValueError(f"`archive_points` needs shape (N,{len(keys)}) with order {keys}")
    if A.size == 0:
        A = A.reshape(0, len(keys))

    if isinstance(y, dict):
        y_arr = np.array([float(y[k]) for k in keys], dtype=float)
    else:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape != (len(keys),):
            raise ValueError(f"`y` needs shape ({len(keys)},) in order {keys}")

    P = np.vstack([A, y_arr])
    Pm = _nsga_to_min_orientation(P, objectives, keys)

    fronts = _nsga_fast_non_dominated_sort_min(Pm)
    n = Pm.shape[0]
    ranks = np.zeros(n, dtype=int)
    for level, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = level + 1

    cd_raw = np.zeros(n, dtype=float)
    if fronts and fronts[0]:
        cd_raw = _nsga_crowding_distance_min(Pm, fronts[0], eps=eps)

    cd_tilde = np.zeros(n, dtype=float)
    if fronts and fronts[0]:
        f0 = fronts[0]
        if len(f0) <= 2:
            cd_tilde[f0] = 1.0
        else:
            finite_vals = [cd_raw[i] for i in f0 if np.isfinite(cd_raw[i])]
            if len(finite_vals) > 0:
                max_cd = max(finite_vals)
                for i in f0:
                    cd_tilde[i] = 1.0 if not np.isfinite(cd_raw[i]) else (cd_raw[i] / (max_cd + eps))
            else:
                cd_tilde[f0] = 1.0

    yi = n - 1
    w_r = float(np.clip(w_r, 0.0, 1.0))
    if w_c is None:
        w_c = 1.0 - w_r
    rank_y = int(max(1, ranks[yi]))
    cd_raw_y = float(cd_raw[yi]) if np.isfinite(cd_raw[yi]) else float("inf")
    cd_tilde_y = float(cd_tilde[yi])
    reward_y = float(w_r * (1.0 / rank_y) + w_c * cd_tilde_y)

    return reward_y, rank_y, cd_raw_y, cd_tilde_y
