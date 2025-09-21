# run_experiment.py
"""
Run multiple algorithm experiments against the car environment and persist logs.

Key decisions:
- TensorBoard is not used anymore; all logging is written to CSV.
- Algorithms do NOT receive hv_mode/reference_point/rl_reward; those are reserved
  for evaluation scripts (e.g., hypervolume computation in a separate pipeline).
- CSV output contains interaction-level logs plus minimal, useful run metadata
  (run_id, run_index, experiment name, seed, budget, executable, timestamps, wall_time_s).
- The script is robust to a slimmed-down common_config.json (no seeds/experiment_name required).

Expected files:
- configs/common_config.json        (env executable, budget, search_space, objectives, etc.)
- configs/algorithms/<algo>_config.json (per-algorithm defaults; optional but recommended)

Usage:
- python run_experiment.py
- or import main() and pass a subset of algorithms, e.g. main(["SAC","GA"])
"""

import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional

import pandas as pd

from src.environment.car_env import CarEnv
from src.algorithms.active_learning import ActiveLearningAlgorithm
from src.algorithms.soft_actor_critic import SoftActorCritic
from src.algorithms.a3c import A3CAlgorithm
from src.algorithms.nsga2 import Nsga2Algorithm
from src.utils.seed_utils import seed_everything


# --------------------------- I/O helpers ---------------------------

def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return as dict."""
    with open(path, "r") as f:
        return json.load(f)


# --------------------------- Algo helpers ---------------------------

def _algo_name_canonical(tag: str) -> str:
    """Normalize user-facing tags to canonical keys."""
    tag = (tag or "").upper()
    if tag in {"AL", "ACTIVE_LEARNING"}:
        return "AL"
    if tag in {"GA", "NSGA2", "NSGA-II"}:
        return "GA"
    if tag in {"SAC"}:
        return "SAC"
    if tag in {"A3C"}:
        return "A3C"
    raise ValueError(f"Unknown algorithm tag: {tag}")


def _load_algo_cfg(algo_key: str) -> Dict[str, Any]:
    """
    Load per-algorithm config with sensible fallbacks.

    Looks for:
      - configs/algorithms/<key>_config.json  (e.g., sac_config.json)
      - for GA, also try nsga2_config.json / ga_config.json
      - for AL, also try activelearning_config.json / al_config.json
    """
    key = _algo_name_canonical(algo_key)
    primary = f"configs/algorithms/{key.lower()}_config.json"
    fallback = []
    if key == "GA":
        fallback += ["configs/algorithms/nsga2_config.json", "configs/algorithms/ga_config.json"]
    if key == "AL":
        fallback += ["configs/algorithms/activelearning_config.json", "configs/algorithms/al_config.json"]

    for path in [primary, *fallback]:
        try:
            cfg = _load_json(path)
        except FileNotFoundError:
            cfg = None
        if cfg and isinstance(cfg, dict):
            return cfg
    return {"algorithm_params": {}}


def _build_algorithm(tag: str, env: CarEnv, search_space: List[Dict[str, Any]], algo_params: Dict[str, Any]):
    """Factory constructing an algorithm with its environment & search space."""
    tag = _algo_name_canonical(tag)
    if tag == "AL":
        return ActiveLearningAlgorithm(env=env, search_space=search_space, **algo_params)
    if tag == "GA":
        return Nsga2Algorithm(env=env, search_space=search_space, **algo_params)
    if tag == "SAC":
        return SoftActorCritic(env=env, search_space=search_space, **algo_params)
    if tag == "A3C":
        return A3CAlgorithm(env=env, search_space=search_space, **algo_params)
    raise RuntimeError("unreachable")


# --------------------------- Dataframe utils ---------------------------

def _set_or_insert(df: pd.DataFrame, idx: int, col: str, value) -> None:
    """
    Set or insert a constant column with `value`.
    If the column exists, the whole column is overwritten with the constant.
    Otherwise, the column is inserted at position `idx`.
    """
    if col in df.columns:
        df[col] = value
    else:
        df.insert(idx, col, value)


# --------------------------- Main runner ---------------------------

def main(algo_list: Optional[List[str]] = None) -> None:
    """
    Run a batch of experiments for each algorithm and write per-run and combined CSVs.

    - Algorithms default to ["AL", "GA", "SAC", "A3C"] unless overridden by `common["algorithms"]`
      or by passing `algo_list`.
    - Seeds are taken from `common["seeds"]` if present. Otherwise, derive from
      `common["seed"]` and `common["num_runs"]` (defaults: 42 and 10).
    """
    # ---- Load common config ----
    common_path = "configs/common_config.json"
    common = _load_json(common_path)
    print(f"[OK] Loaded common config: {common_path}")

    # Algorithms to run
    if algo_list is None:
        algo_list = common.get("algorithms", ["AL", "GA", "SAC", "A3C"])
    algo_list = [_algo_name_canonical(a) for a in algo_list]
    print(f"[INFO] Algorithms to run: {', '.join(algo_list)}")

    # Seeds (robust to slimmed config)
    seeds: Optional[List[int]] = common.get("seeds")
    if not seeds:
        base_seed = int(common.get("seed", 42))
        num_runs = int(common.get("num_runs", 10))
        seeds = [base_seed + i for i in range(num_runs)]
    print(f"[INFO] Seeds: {seeds}")

    # Paths & basics
    experiment_name = common.get("experiment_name", "default_experiment")
    output_root = common.get("output_dir", "results")
    search_space = common.get("search_space", [])
    budget = int(common.get("evaluation_budget", 500))
    executable_name = common.get("executable_name", "ConsumptionCar.exe")

    # Keep objectives in case downstream analysis wants them; not passed to algorithms
    objectives = common.get(
        "objectives",
        {"consumption": "min", "ela3": "min", "ela4": "min", "ela5": "min"},
    )

    # NOTE: We intentionally do NOT store hv_mode/reference_point in the per-row logs,
    # and we do NOT pass them to the algorithms. If needed for evaluation, load them
    # from the config in a separate analysis script.
    # ref_point = common.get("reference_point", {...})
    # hv_mode = str(common.get("hv_mode", "approx")).lower()

    for algo_tag in algo_list:
        # Load per-algorithm defaults (if present)
        algo_cfg = _load_algo_cfg(algo_tag)
        print(f"[OK] Loaded algo config for {algo_tag}")
        base_params: Dict[str, Any] = dict(algo_cfg.get("algorithm_params", {}))

        all_runs_rows: List[pd.DataFrame] = []

        for run_idx, seed in enumerate(seeds, start=1):
            # Reproducibility
            seed_everything(seed)

            # Run identifiers & output path
            run_id = str(uuid.uuid4())
            run_ts = time.strftime("%Y%m%d-%H%M%S")
            algo_dir = os.path.join(output_root, experiment_name, algo_tag)
            _ensure_dir(algo_dir)
            run_csv = os.path.join(algo_dir, f"{algo_tag}_seed{seed}_run{run_idx}_{run_ts}.csv")

            # Prepare algorithm parameters
            algo_params = dict(base_params)
            # Keep only minimal, common passthroughs:
            algo_params.update({
                "use_constraints": bool(common.get("use_constraints", True)),
                # Algorithms accept **kwargs; they will ignore unknowns gracefully.
                # We do NOT pass hv_mode/reference_point/rl_reward.
            })
            algo_params["seed"] = seed

            # Build env & algorithm
            env = CarEnv(exe_file_name=executable_name)
            algorithm = _build_algorithm(algo_tag, env, search_space, algo_params)

            # Run algorithm
            print(f"[RUN] {algo_tag} | seed={seed} | run={run_idx} | budget={budget}")
            t0 = time.perf_counter()
            try:
                algorithm.run(budget=budget)
            except Exception as e:
                print(f"[ERROR] Algorithm run failed: {e}")
                import traceback
                traceback.print_exc()
            wall_time_s = time.perf_counter() - t0
            end_ts_unix = time.time()

            # Collect results (robust fallback if no logs)
            if hasattr(algorithm, "results_list") and algorithm.results_list:
                df = pd.DataFrame(algorithm.results_list)
            else:
                df = pd.DataFrame(columns=[
                    "algo", "seed", "evaluation", "timestamp",
                    "p1_final_drive_ratio", "p2_roll_radius", "p3_gear3_diff", "p4_gear4_diff", "p5_gear5",
                    "consumption", "ela3", "ela4", "ela5", "reward"
                ])

            # Run-level metadata (constant columns)
            start_ts_unix = float(df["timestamp"].min()) if not df.empty and "timestamp" in df.columns else end_ts_unix - wall_time_s
            _set_or_insert(df, 0, "run_id", run_id)
            _set_or_insert(df, 1, "run_index", run_idx)
            _set_or_insert(df, 2, "experiment", experiment_name)
            _set_or_insert(df, 3, "algo", algo_tag)
            _set_or_insert(df, 4, "seed", seed)
            df["budget"] = budget
            df["executable_name"] = executable_name
            df["start_ts"] = start_ts_unix
            df["end_ts"] = end_ts_unix
            df["wall_time_s"] = wall_time_s

            # NOTE: No hv_mode/ref_point columns are added here by design.

            # Persist this run
            df.to_csv(run_csv, index=False)
            print(f"[SAVE] {run_csv}  ({len(df)} rows)")

            all_runs_rows.append(df)

        # Write aggregated CSV per algorithm
        if all_runs_rows:
            combined = pd.concat(all_runs_rows, axis=0, ignore_index=True)
            combined_csv = os.path.join(output_root, experiment_name, algo_tag, f"{algo_tag}_all_runs.csv")
            combined.to_csv(combined_csv, index=False)
            print(f"[SAVE] {combined_csv}  ({len(combined)} rows total)")

    print("\n--- All experiments finished. ---")


if __name__ == "__main__":
    main()
