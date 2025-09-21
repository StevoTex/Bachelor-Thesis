from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional, List
import numpy as np

# --- Default: ALLE Ziele werden minimiert ---
OBJECTIVES_DEFAULT = {"consumption": "min", "ela3": "min", "ela4": "min", "ela5": "min"}

# Standard-Skalierung \hat{s}^{-1} aus der Aufgabenstellung:
_DEFAULT_S_HAT_INV = (1.0 / 0.7, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 11.0)
_KEYS = ("consumption", "ela3", "ela4", "ela5")


def sanitize_sim(sim: Tuple[float, float, float, float]) -> bool:
    c, e3, e4, e5 = sim
    if any(np.isnan([c, e3, e4, e5])):
        return False
    if c <= 0 or e3 <= 0 or e4 <= 0 or e5 <= 0:
        return False
    return True


# --------------------------------------------------------------------------
# Target-BOX-Helfer für einheitliche Auswertung/Reward/Logging
# --------------------------------------------------------------------------
def _ranges_to_arrays(target_ranges: Dict[str, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Konvertiert ein Dict mit Intervallen in Arrays (Min/Max) in fester Reihenfolge."""
    mins = []
    maxs = []
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
    """
    True, wenn die Simulationswerte im Ziel-Hyperrechteck liegen.
    - require_all=True     -> alle 4 müssen im Intervall liegen
    - require_all=False    -> es reicht, wenn min_satisfied Ziele drin liegen (Default=4)
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
    """
    Abstand eines Punktes zur Ziel-BOX:
    - 'box_dist'        : euklidischer Abstand zum projizierten Punkt (ungewichtet)
    - 'box_dist_scaled' : Abstand nach komponentenweiser Skalierung (∘ s_hat_inv)
    Ist der Punkt in der BOX, sind beide Distanzen 0.
    """
    y = np.asarray(sim, dtype=float)
    mins, maxs = _ranges_to_arrays(target_ranges)
    proj = np.clip(y, mins, maxs)        # Projektion auf die Box
    diff = y - proj                      # 0, wenn in der Box

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
    """
    Reward/Score auf Basis der Distanz zur Ziel-BOX:
        r = 1 / (eps + || (y - Π_Box(y)) ∘ s_hat_inv ||)
    -> innen (Distanz==0) ergibt maximalen Score (=1/eps), aber KEIN Early-Stop.
    """
    if not sanitize_sim(sim):
        return 0.0
    _, d_scaled = box_distance(sim, target_ranges, s_hat_inv=s_hat_inv, norm=norm)
    return 1.0 / max(eps, d_scaled)


# --------------------------------------------------------------------------
# Weitere, bestehende Rewards / Utilities
# --------------------------------------------------------------------------
def reward_heuristic(sim: Tuple[float, float, float, float],
                     w: Dict[str, float],
                     invalid_penalty: float = -10.0) -> float:
    if not sanitize_sim(sim):
        return float(invalid_penalty)
    c, e3, e4, e5 = sim
    w_c = float(w.get("w_consumption", 1.0))
    w_e = float(w.get("w_elas_sum", 0.1))
    # Minimierung der Ziele ↔ Reward maximieren (negativ gewichtete Summe)
    return float(-(w_c * c + w_e * (e3 + e4 + e5)))


def to_minimization(phen: Tuple[float, float, float, float],
                    objectives: Dict[str, str] = OBJECTIVES_DEFAULT) -> Tuple[float, ...]:
    out = []
    for k, v in zip(_KEYS, phen):
        out.append(float(v) if objectives.get(k, "min") == "min" else float(-v))
    return tuple(out)


_def_ranges = {"consumption": (3.0, 15.0), "ela3": (0.0, 10.0), "ela4": (0.0, 10.0), "ela5": (0.0, 10.0)}


def _norm(val: float, lo: float, hi: float) -> float:
    return (val - lo) / max(1e-12, hi - lo)


def reward_tchebycheff(sim: Tuple[float, float, float, float],
                        weights: Sequence[float] | None,
                        ranges: Dict[str, Tuple[float, float]] | None = None,
                        rho: float = 1e-3,
                        invalid_penalty: float = -10.0,
                        objectives: Dict[str, str] = OBJECTIVES_DEFAULT) -> float:
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
    """
    Einheitliche Reward-Schnittstelle für RL & Co.
    Unterstützt:
      - type == "target_box": Distanz zur Ziel-BOX (empfohlen)
      - type == "tchebycheff"
      - sonst: heuristische Negativsumme
    """
    t = str(cfg.get("type", "heuristic")).lower()
    invalid = float(cfg.get("invalid_penalty", -10.0))

    if t in ("target_box", "box", "box_distance"):
        tr = cfg.get("target_ranges")
        if not isinstance(tr, dict):
            # Ohne Ranges fällt es auf Heuristik zurück
            return reward_heuristic(sim, w=cfg.get("heuristic_weights", {"w_consumption": 1.0, "w_elas_sum": 0.1}),
                                    invalid_penalty=invalid)
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


# --- NEU: Terminal-Kriterium für RL bei Treffer der Target-Box ----------------
def is_terminal(sim: Tuple[float, float, float, float], cfg: Dict[str, object]) -> bool:
    """
    True, wenn die aktuelle Simulation die definierte Target-Box trifft.
    (Nur aktiv für cfg['type'] == 'target_box'.)
    """
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


# ==========================================================================
# NSGA-Reward (ohne DC/Ref-Punkt): Rank + normalisierte Crowding-Distanz
# ==========================================================================

def _nsga_obj_keys(objectives: Dict[str, str]) -> List[str]:
    # Verwende die Reihenfolge des objectives-Dicts
    return list(objectives.keys())

def _nsga_to_min_orientation(P: np.ndarray, objectives: Dict[str, str], keys: List[str]) -> np.ndarray:
    Pm = P.copy()
    for i, k in enumerate(keys):
        if str(objectives.get(k, "min")).lower() == "max":
            Pm[:, i] *= -1.0
    return Pm

def _nsga_dominates_min(a: np.ndarray, b: np.ndarray) -> bool:
    # Minimierung: a dominiert b <=> a <= b (alle) und a < b (mind. eins)
    return bool(np.all(a <= b) and np.any(a < b))

def _nsga_fast_non_dominated_sort_min(Pm: np.ndarray) -> List[List[int]]:
    """NSGA-II Fast non-dominated sort auf Minimierungs-orientierten Punkten."""
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
    """Roh-Crowding-Distanz für die gegebene Front (Min-Orientierung)."""
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
        order = np.argsort(vals)  # aufsteigend (Minimierung)
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
    """
    NSGA-kompatibler Reward ohne Referenzpunkt:
        r = w_r * (1 / rank) + w_c * CD_tilde

    - 'rank'   : 1-basierter Rang aus Fast Non-Dominated Sort.
    - 'CD_tilde': normalisierte Crowding Distanz auf der ersten Front (Punkte nicht in Front 1 → 0).
                  Normalisierung: finite CD / max(finite CD) in Front 1; Randpunkte (inf) → 1.0.

    Rückgabe: (reward, rank, cd_raw, cd_tilde)
    """
    keys = _nsga_obj_keys(objectives)

    # Archiv in Array formen
    if hasattr(archive_points, "to_numpy"):  # pandas.DataFrame
        A = archive_points[keys].to_numpy(dtype=float, copy=False)
    else:
        A = np.asarray(archive_points, dtype=float)
        if A.size and (A.ndim != 2 or A.shape[1] != len(keys)):
            raise ValueError(f"`archive_points` needs shape (N,{len(keys)}) with order {keys}")
    if A.size == 0:
        A = A.reshape(0, len(keys))

    # y in Array
    if isinstance(y, dict):
        y_arr = np.array([float(y[k]) for k in keys], dtype=float)
    else:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape != (len(keys),):
            raise ValueError(f"`y` needs shape ({len(keys)},) in order {keys}")

    # Kombinieren & in Minimierungs-Orientierung bringen
    P = np.vstack([A, y_arr])
    Pm = _nsga_to_min_orientation(P, objectives, keys)

    # Fast non-dominated sort
    fronts = _nsga_fast_non_dominated_sort_min(Pm)
    n = Pm.shape[0]
    ranks = np.zeros(n, dtype=int)
    for level, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = level + 1  # 1-basiert

    # Crowding-Distanz (nur Front 1)
    cd_raw = np.zeros(n, dtype=float)
    if fronts and fronts[0]:
        cd_raw = _nsga_crowding_distance_min(Pm, fronts[0], eps=eps)

    # CD-Normalisierung zu [0,1] auf Front 1
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

    # Reward des neu hinzugefügten Punkts (letzter Index)
    yi = n - 1
    w_r = float(np.clip(w_r, 0.0, 1.0))
    if w_c is None:
        w_c = 1.0 - w_r
    rank_y = int(max(1, ranks[yi]))
    cd_raw_y = float(cd_raw[yi]) if np.isfinite(cd_raw[yi]) else float("inf")
    cd_tilde_y = float(cd_tilde[yi])
    reward_y = float(w_r * (1.0 / rank_y) + w_c * cd_tilde_y)

    return reward_y, rank_y, cd_raw_y, cd_tilde_y
