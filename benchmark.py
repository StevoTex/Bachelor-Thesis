# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Benchmark analysis & plots for multi-objective experiments.

Functional highlights:
- Works with per-algorithm *all_runs* CSVs (multiple seeds in one file).
- Robust HV curves (per-seed filtering, dedup per 'evaluation', strict bounds via metrics.py).
- HV(t) hard-cut at 100 evaluations, with padding per seed to reach 100.
- Spacing & Diversity (Coverage) plots (bar + box; bars/boxes sorted by mean desc).
- Report of the number of Pareto solutions per seed + union over seeds.
- Directed hypervolume-improvement heatmap (HVI) only (ε heatmap removed).
- Friedman (average ranks only) for HV, Spacing, Coverage, and Time.
- TIME: wall_time_s is the **total runtime per seed**. Per algorithm, the
  **average over 10 seeds** is computed as (sum of seed times) / 10 — without filtering.

Analysis extras:
- Overview table HV per seed (seeds = columns, algorithms = rows): hv_per_seed_matrix.csv
- Per algorithm a CSV with PF points per seed + counts “negativ” and “out_of_box”:
  <algo>/<algo>_pf_summary.csv
- Pareto profiles per seed (line plot) + union profiles per algorithm

Assumptions:
- CSV logs include at least: ["algo","seed","evaluation", objectives...].
- "phase" may include "update" rows; those are excluded in metrics.hv_curve_over_evals (only for HV(t)).
- Invalid simulator outputs (NaN/inf/negative/out-of-bounds) are filtered by metric helpers (not for time).
- Exactly **10 seeds** per algorithm are present.

Outputs go to: <results_dir>/<out_subdir>/
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics import (
    extract_pareto_front,
    metric_hypervolume,
    metric_spacing,
    metric_diversity_measure,
    metric_maximum_spread,
    hv_curve_over_evals,          # robust, dedups per evaluation, excludes 'update'
    evals_to_reach_fraction,
)

# ---- Global constants ----
# For profile plots use an explicit objective order:
ORDERED_OBJ_COLS = ["ela3", "ela4", "ela5", "consumption"]
OBJ_COLS = ["consumption", "ela3", "ela4", "ela5"]  # keep log column names
ALGO_ALIASES = {
    "GA": "NSGA2",
    "NSGA-II": "NSGA2",
    "NSGA_II": "NSGA2",
    "NSGA2": "NSGA2",
    "A3C": "A3C",
    "SAC": "SAC",
    "AL": "AL",
    "ACTIVELEARNING": "AL",
    "ACTIVE_LEARNING": "AL",
}
DEFAULT_ALGOS = ["A3C", "SAC", "AL", "NSGA2"]

# HV(t) cutoff & seeds
HV_EVALS_CUTOFF = 100
N_SEEDS_PER_ALGO = 10  # hard-coded


# ------------------------------- Config I/O ---------------------------------

def _load_common() -> Tuple[Dict[str, str], Dict[str, float], str, Optional[Dict[str, Tuple[float, float]]]]:
    """Read central settings including strict bounds from configs/common_config.json."""
    try:
        with open("configs/common_config.json", "r") as f:
            common = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read configs/common_config.json ({e}). Using defaults.")
        common = {}

    objectives = common.get(
        "objectives",
        {"consumption": "min", "ela3": "min", "ela4": "min", "ela5": "min"},
    )
    reference_point = common.get(
        "reference_point",
        {"consumption": 15.0, "ela3": 40.0, "ela4": 50.0, "ela5": 60.0},
    )
    lower_bounds = common.get("lower_bounds", None)
    if lower_bounds is None:
        lower_bounds = common.get("lower_bound", {"consumption": 0.0, "ela3": 0.0, "ela4": 0.0, "ela5": 0.0})
    hv_mode = str(common.get("hv_mode", "approx")).lower()

    bounds: Optional[Dict[str, Tuple[float, float]]] = {}
    try:
        for k in objectives.keys():
            lo = float(lower_bounds[k]) if lower_bounds is not None and k in lower_bounds else None
            hi = float(reference_point[k]) if reference_point is not None and k in reference_point else None
            bounds[k] = (lo, hi)
    except Exception:
        bounds = None

    return objectives, reference_point, hv_mode, bounds


# ------------------------------- File helpers -------------------------------

def _collect_csvs(results_dir: str) -> List[str]:
    """Recursively collect CSV files from results_dir."""
    csvs = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return sorted(csvs)


def _latest_by_algo_seed(csv_paths: List[str]) -> Dict[Tuple[str, int], str]:
    """
    Expand 'all_runs' CSVs: if a CSV contains multiple seeds, create one mapping per (algo, seed).
    Keep the latest file per (algo, seed) by mtime.
    """
    best: Dict[Tuple[str, int], Tuple[float, str]] = {}
    for p in csv_paths:
        algo_guess = os.path.basename(p).split("_")[0].upper()
        try:
            df_head = pd.read_csv(p, usecols=["algo", "seed"], dtype={"algo": str, "seed": int})
            if "algo" in df_head.columns and not df_head.empty:
                algo_guess = str(df_head["algo"].iloc[0]).upper()
        except Exception:
            df_head = None

        algo = ALGO_ALIASES.get(algo_guess, algo_guess)
        seeds: List[int] = []
        if df_head is not None and "seed" in df_head.columns:
            try:
                seeds = sorted(int(s) for s in df_head["seed"].dropna().astype(int).unique().tolist())
            except Exception:
                seeds = []
        if not seeds:
            m = re.search(r"seed(\d+)", os.path.basename(p), flags=re.IGNORECASE)
            if m:
                seeds = [int(m.group(1))]

        for sd in seeds:
            key = (algo, int(sd))
            mt = os.path.getmtime(p)
            if key not in best or mt > best[key][0]:
                best[key] = (mt, p)

    return {k: v for k, (_, v) in best.items()}


# ----------------------------- Data preparation -----------------------------

def _valid_eval_mask(df: pd.DataFrame, obj_cols: List[str], bounds: Optional[Dict[str, Tuple[float, float]]]) -> np.ndarray:
    """Valid rows: not phase=='update', finite objectives, >= 0, inside bounds."""
    mask = np.ones(len(df), dtype=bool)
    if "phase" in df.columns:
        mask &= df["phase"].astype(str).str.lower() != "update"
    if not set(obj_cols).issubset(df.columns):
        return np.zeros(len(df), dtype=bool)
    X = df[obj_cols].to_numpy(dtype=float, copy=True)
    mask &= np.isfinite(X).all(axis=1)
    mask &= (X >= 0.0).all(axis=1)
    if bounds:
        for j, c in enumerate(obj_cols):
            lo, hi = bounds.get(c, (None, None))
            if lo is not None:
                mask &= X[:, j] >= float(lo)
            if hi is not None:
                mask &= X[:, j] <= float(hi)
    return mask


# -------------------------- HV(t) alignment to 1..100 -----------------------

def _clip_and_pad_to_cutoff(s: pd.Series) -> pd.Series:
    """Clip HV series to [1..HV_EVALS_CUTOFF] and pad via ffill/bfill to exact length."""
    if s is None or s.empty:
        return pd.Series(index=pd.Index(range(1, HV_EVALS_CUTOFF + 1), name="evals"), dtype=float)

    try:
        s.index = s.index.astype(int)
    except Exception:
        pass

    if 1 not in s.index and 0 in s.index:
        s = s.copy()
        s.loc[1] = s.loc[0]
        s = s[~s.index.duplicated(keep="first")].sort_index()

    idx = pd.Index(range(1, HV_EVALS_CUTOFF + 1), name="evals")
    s = s[s.index <= HV_EVALS_CUTOFF]
    s = s.reindex(idx).ffill().bfill()
    return s


def _align_series_mean(series_by_seed: Dict[int, pd.Series]) -> Tuple[pd.Index, pd.Series, int, int]:
    """Align all seed series to 1..HV_EVALS_CUTOFF (no common-prefix truncation)."""
    idx = pd.Index(range(1, HV_EVALS_CUTOFF + 1), name="evals")
    if not series_by_seed:
        return idx, pd.Series(index=idx, dtype=float), HV_EVALS_CUTOFF, HV_EVALS_CUTOFF

    aligned = [_clip_and_pad_to_cutoff(s) for s in series_by_seed.values()]
    A = pd.concat(aligned, axis=1)
    mean = A.mean(axis=1)
    return idx, mean, HV_EVALS_CUTOFF, HV_EVALS_CUTOFF


# ------------------------------- Time handling ------------------------------

def _seed_total_time_s(df: pd.DataFrame) -> float:
    """Total runtime per seed (seconds) — no filtering/tricks. Uses max(wall_time_s)."""
    if "wall_time_s" not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df["wall_time_s"], errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.max())


# ------------------------------- Counting helpers ---------------------------

def _counts_neg_and_oob_unique(df: pd.DataFrame, bounds: Optional[Dict[str, Tuple[float, float]]]) -> Tuple[int, int]:
    """Over unique objective vectors in the seed: count negatives and out-of-bounds."""
    if not set(OBJ_COLS).issubset(df.columns):
        return 0, 0
    Z = df[OBJ_COLS].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    if Z.empty:
        return 0, 0
    neg_vectors_total = int((Z < 0).any(axis=1).sum())
    out_of_box_total = 0
    if bounds:
        oob = pd.Series(False, index=Z.index)
        for c in OBJ_COLS:
            lo, hi = bounds.get(c, (None, None)) if bounds else (None, None)
            if lo is not None:
                oob |= (Z[c] < float(lo))
            if hi is not None:
                oob |= (Z[c] > float(hi))
        out_of_box_total = int(oob.sum())
    return neg_vectors_total, out_of_box_total


# ------------------------------- Plot helpers -------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _bar_chart(values: Dict[str, float], ylabel: str, out_path: str) -> None:
    """Bar chart sorted by value desc; drops NaN/Inf."""
    pairs = [(k, float(v)) for k, v in values.items() if np.isfinite(v)]
    if not pairs:
        return
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in pairs]
    vals = [v for _, v in pairs]

    plt.figure(figsize=(7.8, 4.2), constrained_layout=True)
    plt.bar(labels, vals)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def _box_plot(series_dict: Dict[str, List[float]], ylabel: str, out_path: str) -> None:
    """Box plot sorted by mean desc; skips empty series."""
    stats = []
    for k, v in series_dict.items():
        arr = np.array(v, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        m = float(np.nanmean(arr))
        stats.append((k, m))
    if not stats:
        return
    labels_sorted = [k for k, _ in sorted(stats, key=lambda t: t[1], reverse=True)]
    data = [series_dict[k] for k in labels_sorted]

    plt.figure(figsize=(7.8, 4.2), constrained_layout=True)
    plt.boxplot(data, labels=labels_sorted, showmeans=True)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def _heatmap(matrix: pd.DataFrame, out_path: str, cbar_label: str = "", value_fmt: str = "%.2f") -> None:
    """Simple matrix heatmap with inline annotations."""
    plt.figure(figsize=(8.6, 6.2), constrained_layout=True)
    plt.imshow(matrix.values, cmap="viridis")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.gcf().subplots_adjust(left=0.24, bottom=0.18, right=0.98, top=0.98)

    mean_val = np.nanmean(matrix.values.astype(float))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iloc[i, j]
            if pd.isna(v):
                txt = "—"
            else:
                try:
                    txt = (value_fmt % float(v))
                except Exception:
                    txt = str(v)
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if (pd.notna(v) and float(v) > mean_val) else "black")
    if cbar_label:
        plt.colorbar(label=cbar_label)
    else:
        plt.colorbar()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def _plot_hv_all_seeds(algo: str, curves: Dict[int, pd.Series], out_dir: str) -> None:
    """HV(t) per seed aligned to 1..HV_EVALS_CUTOFF + bold black mean."""
    _ensure_dir(out_dir)
    idx, mean, _, _ = _align_series_mean(curves)
    plt.figure(figsize=(7.5, 4.3), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, sd in enumerate(sorted(curves.keys())):
        col = cmap(i % 10)
        s = _clip_and_pad_to_cutoff(curves[sd])
        plt.plot(idx, s.values, label=f"Seed {sd}", linewidth=1.2, color=col, alpha=0.9)
    if not mean.empty:
        plt.plot(idx, mean.values, color="black", linewidth=3.0, label="Mittelwert")
    plt.xlim(1, HV_EVALS_CUTOFF)
    plt.xlabel("Auswertungen")
    plt.ylabel("Hypervolumen")
    plt.legend(title=None, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{algo}_hv_all_seeds.png"), dpi=160, bbox_inches='tight')
    plt.close()


def _plot_hv_mean_only(algo: str, curves: Dict[int, pd.Series], out_dir: str) -> None:
    """Mean-only HV(t) curve aligned to 1..HV_EVALS_CUTOFF."""
    _ensure_dir(out_dir)
    idx, mean, _, _ = _align_series_mean(curves)
    plt.figure(figsize=(7.5, 4.0), constrained_layout=True)
    if not mean.empty:
        plt.plot(idx, mean.values, color="black", linewidth=3.0)
    plt.xlim(1, HV_EVALS_CUTOFF)
    plt.xlabel("Auswertungen")
    plt.ylabel("Hypervolumen")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{algo}_hv_mean.png"), dpi=160, bbox_inches='tight')
    plt.close()


def _plot_union_pareto_profiles(algo: str, pf_union: pd.DataFrame, out_dir: str) -> None:
    """Line-profile per union-Pareto point across ORDERED_OBJ_COLS (original scales)."""
    _ensure_dir(out_dir)
    obj_order = [c for c in ORDERED_OBJ_COLS if c in pf_union.columns]
    if not obj_order:
        return
    Z = pf_union[obj_order].apply(pd.to_numeric, errors="coerce").dropna()
    if Z.empty:
        return

    x = np.arange(len(obj_order))
    plt.figure(figsize=(7.8, 4.6), constrained_layout=True)
    for _, row in Z.iterrows():
        y = row.values.astype(float)
        plt.plot(x, y, linewidth=0.8, alpha=0.35)

    plt.xticks(x, obj_order)
    plt.xlabel("Zielvariablen")
    plt.ylabel("Werte")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{algo}_union_pareto_profiles.png"), dpi=160, bbox_inches='tight')
    plt.close()


def _plot_pareto_profile_for_seed(algo: str, seed: int, pf_seed: pd.DataFrame, out_dir: str) -> None:
    """Pareto profiles per seed: each PF solution as a line over ORDERED_OBJ_COLS."""
    _ensure_dir(out_dir)
    obj_order = [c for c in ORDERED_OBJ_COLS if c in pf_seed.columns]
    if not obj_order:
        return
    Z = pf_seed[obj_order].apply(pd.to_numeric, errors="coerce").dropna()
    if Z.empty:
        return

    x = np.arange(len(obj_order))
    plt.figure(figsize=(7.4, 4.4), constrained_layout=True)
    for _, row in Z.iterrows():
        y = row.values.astype(float)
        plt.plot(x, y, linewidth=0.9, alpha=0.5)
    plt.xticks(x, obj_order)
    plt.xlabel("Zielvariablen")
    plt.ylabel("Werte")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{algo}_seed{seed}_pareto_profile.png"), dpi=160, bbox_inches='tight')
    plt.close()


# -------------------------- Statistics (Friedman) ---------------------------

def _average_ranks(row: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Average ranks with ties (rank 1 = best)."""
    x = row.copy().astype(float)
    if higher_is_better:
        x = -x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)

    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inv, weights=ranks)
    avg = rank_sums / counts
    return avg[inv]


def _friedman_ranks_only(per_algo_seed_values: Dict[str, Dict[int, float]],
                         algos_order: List[str],
                         metric_name: str,
                         out_dir: str,
                         higher_is_better: bool = True) -> None:
    """
    Compute **average ranks only** (no test statistics). Writes CSV:
      friedman_ranks_<metric>.csv with columns [Algorithmus, Durchschnittsrang]
    over common seeds with finite values across all algorithms.
    """
    seed_sets = [set(per_algo_seed_values[a].keys()) for a in algos_order]
    common = set.intersection(*seed_sets) if seed_sets else set()

    def _finite(a: str, s: int) -> bool:
        v = per_algo_seed_values[a].get(s, float("nan"))
        return np.isfinite(v)
    common = sorted([s for s in common if all(_finite(a, s) for a in algos_order)])

    if len(common) < 2 or len(algos_order) < 2:
        print(f"[FRIEDMAN-RANKS] Not enough blocks/algorithms for {metric_name} after NaN filtering.")
        return

    X = np.array([[per_algo_seed_values[a][s] for a in algos_order] for s in common], dtype=float)
    ranks = np.vstack([_average_ranks(X[i, :], higher_is_better) for i in range(X.shape[0])])
    avg_ranks = np.mean(ranks, axis=0)

    ranks_df = pd.DataFrame({"Algorithmus": algos_order, "Durchschnittsrang": avg_ranks})
    ranks_df = ranks_df.sort_values("Durchschnittsrang")
    ranks_path = os.path.join(out_dir, f"friedman_ranks_{metric_name.lower()}.csv")
    ranks_df.to_csv(ranks_path, index=False, float_format="%.6f")
    print(f"[FRIEDMAN-RANKS] Saved ranks for {metric_name} -> {ranks_path}")


# ------------------------------- Benchmark core -----------------------------

def run_benchmarks(
    results_dir: str,
    modes: Optional[List[str]] = None,
    make_plots: bool = True,
    out_dir: Optional[str] = None,
    bins: int = 10,
    stride: int = 1,
    frac: float = 0.95,
    include_speed_plot: bool = True,
):
    """Build per-algorithm reports & plots as specified."""
    print(f"--- Running Benchmarks on Results in '{results_dir}' ---")

    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found at '{results_dir}'")
        return

    objectives, reference_point, hv_mode_cfg, bounds = _load_common()
    if not modes:
        modes = [hv_mode_cfg]
    modes = [str(m).lower() for m in modes]

    all_csvs = _collect_csvs(results_dir)
    if not all_csvs:
        print(f"No result files (.csv) found under '{results_dir}'.")
        return

    # Reduce to the latest file per (algo, seed) – expands all_runs CSVs
    latest_map = _latest_by_algo_seed(all_csvs)
    if not latest_map:
        print("No usable (algo, seed) CSVs found.")
        return

    # Prepare output dir
    if out_dir is None:
        out_dir = os.path.join(results_dir, "analysis")
    _ensure_dir(out_dir)

    # --- Per (algo, seed): compute HV curve, final HV, coverage, spacing, time ---
    per_algo_seed_curves: Dict[str, Dict[int, pd.Series]] = defaultdict(dict)
    per_algo_seed_hv: Dict[str, Dict[int, float]] = defaultdict(dict)
    per_algo_seed_cov: Dict[str, Dict[int, float]] = defaultdict(dict)
    per_algo_seed_spacing: Dict[str, Dict[int, float]] = defaultdict(dict)
    per_algo_seed_time: Dict[str, Dict[int, float]] = defaultdict(dict)
    per_algo_seed_evals95: Dict[str, Dict[int, int]] = defaultdict(dict)
    per_algo_seed_pfcount: Dict[str, Dict[int, int]] = defaultdict(dict)
    per_algo_seed_negcnt: Dict[str, Dict[int, int]] = defaultdict(dict)
    per_algo_seed_oobcnt: Dict[str, Dict[int, int]] = defaultdict(dict)

    # PF CSV rows per algorithm:
    pf_summary_rows_by_algo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Union-DF for HVI:
    per_algo_union_df: Dict[str, pd.DataFrame] = defaultdict(lambda: pd.DataFrame(columns=OBJ_COLS))

    for (algo, seed), path in sorted(latest_map.items()):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  - Could not read {path}: {e}")
            continue

        # Subset by seed
        if "seed" in df.columns and seed is not None:
            try:
                df = df[df["seed"].astype(int) == int(seed)].copy()
            except Exception:
                df = df[df["seed"] == seed].copy()

        # Normalize algo name
        if "algo" in df.columns and not df.empty:
            df.loc[:, "algo"] = ALGO_ALIASES.get(str(df["algo"].iloc[0]).upper(), str(df["algo"].iloc[0]).upper())

        if df.empty or not set(OBJ_COLS).issubset(df.columns):
            print(f"  - Skipping {path} (missing objectives or empty) for seed {seed}.")
            continue

        # Time (NO filtering)
        per_algo_seed_time[algo][seed] = _seed_total_time_s(df)

        # Negative / out-of-box counts (unique objective vectors)
        neg_cnt, oob_cnt = _counts_neg_and_oob_unique(df, bounds)
        per_algo_seed_negcnt[algo][seed] = neg_cnt
        per_algo_seed_oobcnt[algo][seed] = oob_cnt

        # HV(t) (excludes 'update' internally, dedup per evaluation)
        curve = hv_curve_over_evals(df, objectives, reference_point, hv_mode_cfg, stride=stride, bounds=bounds)
        if not curve.empty:
            try:
                curve.index = curve.index.astype(int)
            except Exception:
                pass
            curve = curve[curve.index <= HV_EVALS_CUTOFF]
        per_algo_seed_curves[algo][seed] = curve

        # PF & metrics (bounds-aware)
        pf = extract_pareto_front(df, objectives, bounds=bounds)
        hv_val = float(metric_hypervolume(pf, objectives, reference_point, hv_mode=modes[0], bounds=bounds))
        cov_val = float(metric_diversity_measure(pf, objectives, bounds=bounds))
        sp_val = float(metric_spacing(pf, objectives, bounds=bounds))
        per_algo_seed_hv[algo][seed] = hv_val
        per_algo_seed_cov[algo][seed] = cov_val
        per_algo_seed_spacing[algo][seed] = sp_val

        # PF count per seed
        try:
            pf_unique = pf[OBJ_COLS].drop_duplicates()
            pf_count = int(len(pf_unique))
        except Exception:
            pf_count = int(len(pf))
        per_algo_seed_pfcount[algo][seed] = pf_count

        # PF rows for per-algo CSV
        if pf is None or pf.empty:
            pf_summary_rows_by_algo[algo].append({
                "seed": int(seed),
                "consumption": np.nan, "ela3": np.nan, "ela4": np.nan, "ela5": np.nan,
                "neg_vectors_total": int(neg_cnt),
                "out_of_box_total": int(oob_cnt),
            })
        else:
            pf_clean = pf[["consumption", "ela3", "ela4", "ela5"]].copy()
            for _, r in pf_clean.iterrows():
                row = {
                    "seed": int(seed),
                    "consumption": float(r["consumption"]),
                    "ela3": float(r["ela3"]),
                    "ela4": float(r["ela4"]),
                    "ela5": float(r["ela5"]),
                    "neg_vectors_total": int(neg_cnt),
                    "out_of_box_total": int(oob_cnt),
                }
                pf_summary_rows_by_algo[algo].append(row)

        # Pareto profiles per seed
        algo_dir = os.path.join(out_dir, algo)
        _plot_pareto_profile_for_seed(algo, int(seed), pf, algo_dir)

        # Evals to x% HV
        try:
            evals95 = int(evals_to_reach_fraction(df, objectives, reference_point, hv_mode_cfg,
                                                  frac=frac, stride=stride, bounds=bounds))
        except Exception:
            evals95 = 0
        per_algo_seed_evals95[algo][seed] = evals95

        # Union DF for HVI
        per_algo_union_df[algo] = pd.concat([per_algo_union_df[algo], df[OBJ_COLS]], axis=0, ignore_index=True)

    # Algorithms present (keep order)
    algos_present = [a for a in DEFAULT_ALGOS if a in per_algo_seed_hv]
    if not algos_present:
        print("No recognized algorithms found in results.")
        return

    # PF CSV per algorithm
    for a in algos_present:
        algo_dir = os.path.join(out_dir, a)
        _ensure_dir(algo_dir)
        cols = ["seed"] + ["consumption", "ela3", "ela4", "ela5"] + ["neg_vectors_total", "out_of_box_total"]
        df_pf = pd.DataFrame(pf_summary_rows_by_algo[a], columns=cols)
        df_pf.to_csv(os.path.join(algo_dir, f"{a}_pf_summary.csv"), index=False)

    # HV-per-seed matrix
    all_seeds_sorted = sorted({s for a in algos_present for s in per_algo_seed_hv[a].keys()})
    hv_mat = pd.DataFrame(index=algos_present, columns=all_seeds_sorted, dtype=float)
    for a in algos_present:
        for s, v in per_algo_seed_hv[a].items():
            hv_mat.loc[a, s] = float(v)
    hv_mat.to_csv(os.path.join(out_dir, "hv_per_seed_matrix.csv"), float_format="%.8f")

    # Plots per algorithm
    for algo in algos_present:
        algo_dir = os.path.join(out_dir, algo)
        _plot_hv_all_seeds(algo, per_algo_seed_curves[algo], algo_dir)
        _plot_hv_mean_only(algo, per_algo_seed_curves[algo], algo_dir)

    # Aggregated metrics
    mean_hv         = {a: float(np.nanmean(list(per_algo_seed_hv[a].values()))) for a in algos_present}
    std_hv          = {a: float(np.nanstd (list(per_algo_seed_hv[a].values()), ddof=0)) for a in algos_present}
    var_hv          = {a: float(np.nanvar (list(per_algo_seed_hv[a].values()), ddof=0)) for a in algos_present}

    mean_cov        = {a: float(np.nanmean(list(per_algo_seed_cov[a].values()))) for a in algos_present}
    std_cov         = {a: float(np.nanstd (list(per_algo_seed_cov[a].values()), ddof=0)) for a in algos_present}
    var_cov         = {a: float(np.nanvar (list(per_algo_seed_cov[a].values()), ddof=0)) for a in algos_present}

    mean_spacing    = {a: float(np.nanmean(list(per_algo_seed_spacing[a].values()))) for a in algos_present}

    # Average runtime per run = (sum of seed times) / 10 (hard-coded)
    time_lists      = {a: list(per_algo_seed_time[a].values()) for a in algos_present}
    mean_time       = {a: float(np.nansum(time_lists[a]) / N_SEEDS_PER_ALGO) for a in algos_present}
    std_time        = {a: float(np.nanstd (time_lists[a], ddof=0)) for a in algos_present}
    var_time        = {a: float(np.nanvar (time_lists[a], ddof=0)) for a in algos_present}

    mean_evals95    = {a: float(np.nanmean(list(per_algo_seed_evals95[a].values()))) for a in algos_present}

    # Ø Pareto solutions / seed — robust: sum / 10 (hard-coded)
    pfcount_lists   = {a: list(per_algo_seed_pfcount[a].values()) for a in algos_present}
    mean_pfcount    = {a: float(np.nansum(pfcount_lists[a]) / N_SEEDS_PER_ALGO) for a in algos_present}

    # For plots
    hv_lists        = {a: list(per_algo_seed_hv[a].values())       for a in algos_present}
    coverage_lists  = {a: list(per_algo_seed_cov[a].values())      for a in algos_present}
    spacing_lists   = {a: list(per_algo_seed_spacing[a].values())  for a in algos_present}

    # Union Pareto per algorithm + profiles
    union_pf_counts: Dict[str, int] = {}
    for a in algos_present:
        df_all = per_algo_union_df[a]
        pf_union = extract_pareto_front(df_all, objectives, bounds=bounds)
        try:
            union_pf_counts[a] = int(len(pf_union[OBJ_COLS].drop_duplicates()))
        except Exception:
            union_pf_counts[a] = int(len(pf_union))
        _plot_union_pareto_profiles(a, pf_union, os.path.join(out_dir, a))

    # Aggregated plots (German axes; bars sorted by mean desc)
    _bar_chart(mean_cov,       "Abdeckung (Varianz)", os.path.join(out_dir, "coverage_bar.png"))
    _box_plot(coverage_lists,  "Abdeckung (Varianz)", os.path.join(out_dir, "coverage_box.png"))

    _bar_chart(mean_hv,        "Hypervolumen",        os.path.join(out_dir, "hv_bar.png"))
    _box_plot(hv_lists,        "Hypervolumen",        os.path.join(out_dir, "hv_box.png"))

    _bar_chart(mean_pfcount,   "Anzahl Pareto-Lösungen/Seed", os.path.join(out_dir, "pfcount_bar.png"))
    _box_plot(pfcount_lists,   "Anzahl Pareto-Lösungen",      os.path.join(out_dir, "pfcount_box.png"))

    _bar_chart(mean_time,      "Zeit (s)",            os.path.join(out_dir, "time_bar.png"))
    _box_plot(time_lists,      "Zeit (s)",            os.path.join(out_dir, "time_box.png"))
    if include_speed_plot:
        _bar_chart(mean_evals95, "Auswertungen",      os.path.join(out_dir, f"evals_to_{int(frac*100)}_bar.png"))

    _bar_chart(mean_spacing,   "Abstand (Spacing)",   os.path.join(out_dir, "spacing_bar.png"))
    _box_plot(spacing_lists,   "Abstand (Spacing)",   os.path.join(out_dir, "spacing_box.png"))

    # --- HVI (directed hypervolume improvement) heatmap (ε heatmap removed) ---
    def _normalize_df(df_vals: pd.DataFrame, objectives: Dict[str, str], bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        if df_vals is None or df_vals.empty:
            return pd.DataFrame(columns=OBJ_COLS)
        from src.metrics import _to_min_space as _to_min_space_priv, _bounds_to_min_space as _b2min_priv, _clean_objective_rows as _clean_priv
        df_clean = _clean_priv(df_vals[OBJ_COLS], objectives, bounds=bounds)
        if df_clean.empty:
            return pd.DataFrame(columns=OBJ_COLS)
        Fmin = _to_min_space_priv(df_clean, objectives)
        lo_min, hi_min = _b2min_priv(bounds, objectives, OBJ_COLS)
        denom = np.maximum(hi_min - lo_min, 1e-12)
        Z = (Fmin.to_numpy(dtype=float) - lo_min[None, :]) / denom[None, :]
        return pd.DataFrame(Z, columns=OBJ_COLS)

    pf_by_algo_norm: Dict[str, pd.DataFrame] = {}
    for a in algos_present:
        df_all = per_algo_union_df[a]
        pf_a = extract_pareto_front(df_all, objectives, bounds=bounds)
        pf_by_algo_norm[a] = _normalize_df(pf_a, objectives, bounds) if bounds else pf_a

    # Common objective columns
    keys_candidates = [set(df.columns) for df in pf_by_algo_norm.values() if not df.empty]
    keys_full = sorted(set.intersection(*keys_candidates)) if keys_candidates else ORDERED_OBJ_COLS
    if not keys_full:
        keys_full = ORDERED_OBJ_COLS

    # HVI
    obj_min_full = {k: "min" for k in keys_full}
    ref_norm_full = {k: 1.0 for k in keys_full}
    hv_norm_single = {}
    for a in algos_present:
        hv_norm_single[a] = metric_hypervolume(pf_by_algo_norm[a][keys_full], obj_min_full, ref_norm_full,
                                               hv_mode=modes[0], bounds=None)

    hvi_mat = pd.DataFrame(index=algos_present, columns=algos_present, dtype=float)
    for a in algos_present:
        for b in algos_present:
            if pf_by_algo_norm[a].empty or pf_by_algo_norm[b].empty:
                hvi = float("nan")
            else:
                union_pf = extract_pareto_front(pd.concat([pf_by_algo_norm[a][keys_full],
                                                           pf_by_algo_norm[b][keys_full]], axis=0, ignore_index=True),
                                                obj_min_full, bounds=None)
                hv_union = metric_hypervolume(union_pf, obj_min_full, ref_norm_full,
                                              hv_mode=modes[0], bounds=None)
                hvi = max(0.0, hv_union - hv_norm_single[b])
            hvi_mat.loc[a, b] = float(hvi) if np.isfinite(hvi) else np.nan
    _heatmap(hvi_mat, os.path.join(out_dir, "hvi_heatmap.png"), cbar_label="ΔHV", value_fmt="%.3f")
    hvi_mat.to_csv(os.path.join(out_dir, "hvi_matrix.csv"), float_format="%.6f")

    # --- Summary table (CSV + stdout) ---
    summary_rows: List[Dict[str, Any]] = []
    for a in algos_present:
        row = {
            "Algorithm": a,
            "Mean HV": mean_hv[a],
            "Std HV": std_hv[a],
            "Var HV": var_hv[a],
            "Mean Coverage (Diversity Var)": mean_cov[a],
            "Std Coverage": std_cov[a],
            "Var Coverage": var_cov[a],
            "Mean Total Time (s)": mean_time[a],   # per-run average = sum/10
            "Std Total Time (s)": std_time[a],
            "Var Total Time (s)": var_time[a],
            "Ø Pareto-Loesungen/Seed": mean_pfcount[a],
            "Pareto-Loesungen (Union)": union_pf_counts[a],
            f"Mean evals to {int(frac*100)}% HV": mean_evals95[a],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["Mean HV", "Mean Coverage (Diversity Var)"], ascending=[False, False])
    print("\n--- Summary (per algorithm, mean over seeds) ---")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    print(f"[SAVE] {os.path.join(out_dir, 'summary_metrics.csv')}")

    # --- Significance: Friedman (average ranks only) over common seeds ---
    _friedman_ranks_only(per_algo_seed_hv,      algos_present, "HV",       out_dir, higher_is_better=True)
    _friedman_ranks_only(per_algo_seed_spacing, algos_present, "Spacing",  out_dir, higher_is_better=False)
    _friedman_ranks_only(per_algo_seed_cov,     algos_present, "Coverage", out_dir, higher_is_better=True)
    _friedman_ranks_only(per_algo_seed_time,    algos_present, "Zeit",     out_dir, higher_is_better=False)


# ------------------------------- Entry point --------------------------------

def main(
    results_dir: str = "results",
    modes: Optional[List[str]] = None,
    make_plots: bool = True,
    out_subdir: str = "analysis",
    bins: int = 10,
    stride: int = 1,
    frac: float = 0.95,
):
    """Convenience wrapper: outputs to <results_dir>/<out_subdir>/."""
    out_dir = os.path.join(results_dir, out_subdir) if out_subdir else None
    run_benchmarks(
        results_dir=results_dir,
        modes=modes,
        make_plots=make_plots,
        out_dir=out_dir,
        bins=bins,
        stride=stride,
        frac=frac,
        include_speed_plot=True,
    )


if __name__ == "__main__":
    main(results_dir="results", modes=None, make_plots=True)
