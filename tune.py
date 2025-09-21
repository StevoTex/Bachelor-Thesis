# -*- coding: utf-8 -*-
import os, json, time, shutil, hashlib, glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna

from src.environment.car_env import CarEnv
from src.algorithms.active_learning import ActiveLearningAlgorithm
from src.algorithms.soft_actor_critic import SoftActorCritic
from src.algorithms.a3c import A3CAlgorithm
from src.algorithms.nsga2 import Nsga2Algorithm

from src.metrics import extract_pareto_front, metric_hypervolume

OBJ_COLS = ["consumption", "ela3", "ela4", "ela5"]


# ---------- kleine I/O-Helfer ----------
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _save_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _hash_params(params: Dict[str, Any]) -> str:
    j = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(j.encode("utf-8"), digest_size=6).hexdigest()

def _trial_dir(root: str, algo_key: str, trial_number: int, params: Dict[str, Any]) -> str:
    h = _hash_params(params)
    label = f"trial{int(trial_number):03d}__{h}"
    return os.path.join(root, algo_key, label)

def _trial_dir_by_number(root: str, algo_key: str, trial_number: int) -> Optional[str]:
    prefix = os.path.join(root, algo_key, f"trial{int(trial_number):03d}__")
    matches = glob.glob(prefix + "*")
    return matches[0] if matches else None

def _project_root() -> Path:
    here = Path(__file__).resolve()
    # Kandidaten: repo-root (mit "configs"), scripts/, src/
    for p in [here.parent, here.parent.parent, here.parent.parent.parent, Path.cwd()]:
        if (p / "configs" / "common_config.json").exists():
            return p
    # Fallback: aktuelle Arbeitsumgebung
    return Path.cwd()

def _load_json(relpath: str) -> Dict[str, Any]:
    root = _project_root()
    full = root / relpath
    if not full.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {full}")
    with open(full, "r") as f:
        return json.load(f)

def _optuna_storage_dir(storage_url: Optional[str]) -> Optional[str]:
    if not storage_url or not storage_url.startswith("sqlite:///"):
        return None
    db_path = storage_url.replace("sqlite:///", "")
    return os.path.dirname(db_path) if db_path else None


# ---------- Algo-Factory ----------
def _algo_name_canonical(tag: str) -> str:
    t = (tag or "").upper()
    if t in {"AL", "ACTIVELEARNING", "ACTIVE_LEARNING"}: return "AL"
    if t in {"NSGA2", "NSGA-II", "GA"}:                  return "NSGA2"
    if t in {"SAC"}:                                     return "SAC"
    if t in {"A3C"}:                                     return "A3C"
    raise ValueError(f"Unknown algorithm tag: {tag}")

def _build_algo(algo_key: str, env, algo_kwargs: Dict[str, Any]):
    key = _algo_name_canonical(algo_key)
    if key == "AL":    return ActiveLearningAlgorithm(env=env, search_space=algo_kwargs.pop("search_space"), **algo_kwargs)
    if key == "SAC":   return SoftActorCritic(env=env, search_space=algo_kwargs.pop("search_space"), **algo_kwargs)
    if key == "A3C":   return A3CAlgorithm(env=env, search_space=algo_kwargs.pop("search_space"), **algo_kwargs)
    if key == "NSGA2": return Nsga2Algorithm(env=env, search_space=algo_kwargs.pop("search_space"), **algo_kwargs)
    raise RuntimeError("unreachable")


# ---------- HV-Helfer ----------
def _compute_hv_from_run_df(run_df: pd.DataFrame,
                            objectives: Dict[str, str],
                            ref_point: Dict[str, float],
                            hv_mode: str = "approx",
                            bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> float:
    if run_df is None or run_df.empty or not set(OBJ_COLS).issubset(run_df.columns):
        return 0.0
    pareto_df = extract_pareto_front(run_df, objectives, bounds=bounds)
    if pareto_df.empty:
        return 0.0
    return float(metric_hypervolume(pareto_df, objectives, ref_point, hv_mode=hv_mode, bounds=bounds))


# ---------- Suggest-Helper ----------
def _suggest_params(trial: optuna.Trial,
                    typed_space: Dict[str, Dict[str, Any]],
                    defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Wenn trial.number==0 -> exakt defaults; sonst: normal samplen."""
    if trial.number == 0:
        trial.set_user_attr("is_default", True)
        return dict(defaults)

    out: Dict[str, Any] = {}
    for k, spec in (typed_space or {}).items():
        t = str(spec.get("type", "")).lower()
        if t == "int":
            out[k] = trial.suggest_int(k, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif t == "float":
            out[k] = trial.suggest_float(k, float(spec["low"]), float(spec["high"]))
        elif t == "logfloat":
            out[k] = trial.suggest_float(k, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "categorical":
            choices = spec.get("choices", [])
            if choices and isinstance(choices[0], (list, tuple)):
                labels = ["x".join(str(int(v)) for v in c) for c in choices]
                label = trial.suggest_categorical(k, labels)
                out[k] = [int(x) for x in str(label).split("x")]
            else:
                out[k] = trial.suggest_categorical(k, choices)
        else:
            out[k] = defaults.get(k, None)
    return out


# ---------- Hauptprogramm ----------
def main(algo_subset: Optional[List[str]] = None) -> None:
    root = _project_root()

    # --- Common Config laden ---
    common = _load_json("configs/common_config.json")
    objectives   = common.get("objectives", {"consumption":"min","ela3":"min","ela4":"min","ela5":"min"})
    ref_point    = common.get("reference_point", {"consumption":15.0,"ela3":15.0,"ela4":15.0,"ela5":15.0})
    lower_bounds = common.get("lower_bounds", None)
    hv_mode      = str(common.get("hv_mode","approx")).lower()
    exe          = common.get("executable_name", "ConsumptionCar.exe")
    search_space = common.get("search_space", [])
    budget       = int(common.get("evaluation_budget", 100))

    # --- Bounds für Normalisierung: (min, max) pro Ziel ---
    hv_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    if lower_bounds and ref_point:
        hv_bounds = {k: (float(lower_bounds[k]), float(ref_point[k]))
                     for k in objectives.keys() if k in lower_bounds and k in ref_point}

    # --- Hyperparameter-Config laden ---
    hp = _load_json("configs/hyperparams.json")

    # --- AL-Config laden (feste Schrittweiten & optionale Defaults) ---
    al_cfg = _load_json("configs/algorithms/al_config.json")
    al_fixed = dict(al_cfg.get("algorithm_params", {}))  # kann init/batch etc. enthalten
    al_step_widths = al_fixed.get("step_widths", None)
    if al_step_widths is None or not isinstance(al_step_widths, dict) or not al_step_widths:
        raise ValueError("al_config.json: 'algorithm_params.step_widths' fehlt oder ist leer.")

    # --- Trials-Auswahl ---
    trials_map = {_algo_name_canonical(k): int(v) for k, v in hp.get("trials", {}).items()}
    if algo_subset is None:
        algo_list = list(trials_map.keys()) or ["AL", "NSGA2", "SAC", "A3C"]
    else:
        algo_list = [_algo_name_canonical(a) for a in algo_subset]

    g = hp.get("global", {})
    seeds_list = g.get("seeds_for_trial")
    if isinstance(seeds_list, list) and seeds_list:
        seeds_for_trial = [int(s) for s in seeds_list]
    else:
        base_seed = int(g.get("seed", 42))
        n_seeds   = int(g.get("seeds_per_trial", 10))
        seeds_for_trial = [base_seed + i for i in range(n_seeds)]
    optuna_seed = int(g.get("seed", 42))

    # (Optional) Storage-Verzeichnis für sqlite anlegen, falls gesetzt
    storage_url = g.get("optuna", {}).get("storage")
    storage_dir = _optuna_storage_dir(storage_url)
    if storage_dir:
        _ensure_dir(storage_dir)

    out_root = str(root / "results" / "tune")
    _ensure_dir(out_root)

    for algo_key in algo_list:
        n_trials = int(trials_map.get(algo_key, 0))
        if n_trials <= 0:
            print(f"[SKIP] No trials for {algo_key}.")
            continue

        algo_hp = hp.get(algo_key, {})
        defaults = dict(algo_hp.get("defaults", {}))
        typed_space = dict(algo_hp.get("search_space", {})) or {}

        # Safety: step_widths darf nicht in typed_space auftauchen.
        if "step_widths" in typed_space:
            print("[WARN] 'step_widths' in Search-Space gefunden; wird entfernt (nicht tunen).")
            typed_space.pop("step_widths", None)

        print(f"[TUNE] {algo_key}: trials={n_trials}, budget={budget}, seeds={len(seeds_for_trial)}, hv_mode={hv_mode}")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=optuna_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1),
            # Storage wird bewusst NICHT gesetzt, um Pfadprobleme zu vermeiden.
            # Falls Persistenz gewünscht: storage=storage_url, study_name=f"HV_{algo_key}", load_if_exists=True
        )

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, typed_space, defaults)

            trial_dir = _trial_dir(out_root, algo_key, trial.number, params)
            runs_dir  = os.path.join(trial_dir, "runs")
            _ensure_dir(runs_dir)
            _save_json(os.path.join(trial_dir, "params.json"), params)

            hvs: List[float] = []
            run_times: List[float] = []

            for s in seeds_for_trial:
                env = CarEnv(exe_file_name=exe)

                # Algo-kwargs zusammenstellen
                algo_kwargs: Dict[str, Any] = {
                    **params,                      # getunte Parameter
                    "seed": int(s),
                    "search_space": search_space,  # Designraum (immer gleich)
                    "use_constraints": bool(common.get("use_constraints", True)),
                    "objectives": objectives
                }

                # AL: feste Schrittweiten + optionale fixe Defaults aus al_config (params haben Vorrang)
                if _algo_name_canonical(algo_key) == "AL":
                    for k, v in al_fixed.items():
                        if k == "step_widths":
                            continue
                        algo_kwargs.setdefault(k, v)
                    algo_kwargs["step_widths"] = al_step_widths  # zwingend

                algo = _build_algo(algo_key, env, algo_kwargs)

                t0 = time.perf_counter()
                try:
                    algo.run(budget=budget)
                except Exception as e:
                    print(f"[WARN] {algo_key} seed={s}: run failed: {e}")
                dt = time.perf_counter() - t0
                run_times.append(dt)

                # Logs einsammeln + speichern
                if hasattr(algo, "results_list") and algo.results_list:
                    df = pd.DataFrame(algo.results_list)
                else:
                    df = pd.DataFrame(columns=["algo","seed","evaluation","timestamp", *OBJ_COLS])

                if "seed" not in df.columns:
                    df.insert(0, "seed", s)
                else:
                    df["seed"] = s
                df_path = os.path.join(runs_dir, f"{algo_key}_seed{s}.csv")
                df.to_csv(df_path, index=False)

                # HV im (optional) normalisierten Raum
                hv = _compute_hv_from_run_df(df, objectives, ref_point, hv_mode=hv_mode, bounds=hv_bounds)
                hvs.append(hv)

            median_hv = float(np.median(hvs)) if hvs else 0.0
            mean_hv   = float(np.mean(hvs))   if hvs else 0.0
            std_hv    = float(np.std(hvs, ddof=0)) if len(hvs) >= 2 else 0.0

            _save_json(os.path.join(trial_dir, "hvs_per_seed.json"),
                       {"seeds": seeds_for_trial, "hvs": hvs})

            _save_json(os.path.join(trial_dir, "score.json"), {
                "algo": algo_key,
                "trial_number": trial.number,
                "median_hv": median_hv,
                "mean_hv": mean_hv,
                "std_hv": std_hv,
                "hv_mode": hv_mode,
                "budget": budget,
                "seeds": seeds_for_trial,
                "run_time_mean_s": float(np.mean(run_times)) if run_times else 0.0,
                "run_time_std_s": float(np.std(run_times, ddof=0)) if len(run_times) >= 2 else 0.0,
                "is_default": bool(trial.user_attrs.get("is_default", False))
            })

            return median_hv

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
        print(f"[BEST] {algo_key}: median_HV={study.best_value:.6f} @ trial #{study.best_trial.number}")

        src = _trial_dir_by_number(out_root, algo_key, study.best_trial.number)
        if src and os.path.isdir(src):
            dst = os.path.join(out_root, algo_key, "best")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"[SAVE] Best trial copied to: {dst}")
        else:
            print("[WARN] Best trial folder not found; skip copy.")


if __name__ == "__main__":
    # nur AL laufen lassen (wie bei dir)
    main(['AL'])
