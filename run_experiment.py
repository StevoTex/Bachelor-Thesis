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

    PARAMETERS
    ----------
    algo_list : Optional[List[str]]
        Which algorithms to run in this invocation. Tags are case-insensitive and
        normalized to the following canonical keys:
          - "AL"    (aliases: "ACTIVE_LEARNING")
          - "GA"    (aliases: "NSGA2", "NSGA-II")
          - "SAC"
          - "A3C"
        If `None`, the runner uses (in order of precedence):
          1) `algorithms` from configs/common_config.json (if present), or
          2) the default ["AL", "GA", "SAC", "A3C"].

    CONFIGURATION (configs/common_config.json)
    -----------------------------------------
    The runner expects a JSON file with (at least) the following keys. All keys are optional
    unless marked as *required*. Sensible defaults are applied when omitted.

    Required for a meaningful run:
      - "search_space": List[Dict]          *required*
          Parameter dictionaries understood by the algorithms (min/max/name etc.).
      - "evaluation_budget": int            *required*
          Number of valid environment evaluations per run.

    Common/optional settings:
      - "experiment_name": str              (default: "default_experiment")
      - "output_dir": str                   (default: "results")
      - "executable_name": str              (default: "ConsumptionCar.exe")
      - "use_constraints": bool             (default: true)
      - "objectives": Dict[str,"min"|"max"] (default: {"consumption":"min","ela3":"min","ela4":"min","ela5":"min"})
      - "algorithms": List[str]             (optional; overrides default set when `algo_list` is None)
      - Seeding (choose one of):
          * "seeds": List[int]              (exact seeds to run)
          * or "seed": int + "num_runs": int   (default: 42 + 10) → seeds = seed + [0..num_runs-1]

    IMPORTANT (intentionally ignored by the runner — used only by analysis scripts):
      - "reference_point"
      - "hv_mode"
      - Any RL reward shaping for analysis
      These are *not* passed into algorithms and are not written to per-row logs.

    PER-ALGORITHM CONFIGS (configs/algorithms/*_config.json)
    --------------------------------------------------------
    Optional JSONs to set constructor kwargs for each algorithm. The runner will search:
      - AL :  activelearning_config.json  or  al_config.json
      - GA :  nsga2_config.json           or  ga_config.json
      - SAC:  sac_config.json
      - A3C:  a3c_config.json
    Expected structure:
      {
        "algorithm_params": {
          "...": <value>,           # forwarded as **kwargs to the algorithm’s constructor
          # Common pass-through added by this runner:
          #   "use_constraints": bool
          #   "seed": int
          # Unknown keys are safely ignored by the algorithm implementations.
        }
      }

    OUTPUTS
    -------
    For each algorithm and seed, a per-run CSV:
      <output_dir>/<experiment_name>/<ALGO>/<ALGO>_seed<seed>_run<run_idx>_<YYYYmmdd-HHMMSS>.csv

    Each CSV contains:
      - Interaction-level logs produced by the respective algorithm (e.g., actions, objectives, reward, diagnostics).
      - Run-level metadata columns (constant per file):
          run_id (uuid4), run_index (1..N), experiment, algo, seed, budget,
          executable_name, start_ts (unix), end_ts (unix), wall_time_s (seconds).

    Additionally, a combined CSV per algorithm:
      <output_dir>/<experiment_name>/<ALGO>/<ALGO>_all_runs.csv

    USAGE
    -----
    CLI:
      $ python run_experiment.py
        → Uses algorithms from common_config.json (or the default set) and the seeding scheme as configured.

    Programmatic:
      >>> from run_experiment import main
      >>> main()                          # use config/default algorithms
      >>> main(["SAC", "GA"])             # run only SAC and NSGA-II
      >>> main(["al","a3c"])              # tags are case-insensitive; aliases allowed

    NOTES & BEHAVIOR
    ----------------
    - One environment instance (CarEnv) is created per run.
    - If an algorithm raises an exception, the stack trace is printed and the runner proceeds
      with the next run/seed (the failed run may still create an empty/partial CSV).
    - `wall_time_s` is measured as end-to-end wall-clock for the run (no filtering).
    - The runner does not accept CLI flags; configure via JSON or the `algo_list` parameter.
    - No TensorBoard is used; all outputs are CSV.

    RETURNS
    -------
    None. Results are written to disk.

    ERRORS
    ------
    - ValueError if `algo_list` contains unknown tags (see accepted tags above).
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
