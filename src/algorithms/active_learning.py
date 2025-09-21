# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Optional, Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.utils.constraints import enforce_gear_descending
from src.utils.objectives import nsga_rank_cd_reward  # Rank + normalized crowding (no ref point)


class ActiveLearningAlgorithm:
    """
    Pool-based Active Learning with a small committee of NN surrogate models.

    Änderungen (Validierung):
    - Negative Simulatorwerte sind ERLAUBT.
    - Verworfen werden NUR nicht-endliche Werte (NaN/Inf).
      → keine Logs, keine Labels, kein Archiveintrag, kein eval_count++
      → Aktion wird gecached (deterministisches Modell) und künftig übersprungen.

    Pool:
    - Kandidatenpool = kartesisches GRID aus Bounds (common_config.search_space)
      und per Config vorgegebenen Schrittweiten `step_widths[name]`.
    - `pool_size` wird automatisch bestimmt; Tuning der Schrittweiten ist ausgeschlossen.
    """

    def __init__(self, env, search_space, **kwargs):
        self.env = env
        self.search_space = search_space
        self.results_list: List[Dict[str, Any]] = []

        # Reproducibility
        self.seed: int = int(kwargs.get("seed", 0))
        self.random_state = np.random.RandomState(self.seed or 42)

        # Algorithm meta
        self.algo_name: str = "AL"

        # AL cycle configuration
        self.initial_label_count: int = int(kwargs.get("initial_label_count", 100))
        self.num_cycles: int = int(kwargs.get("num_cycles", 20))
        self.batch_size: int = int(kwargs.get("batch_size", 20))

        # Surrogate committee (simple MLPs)
        self.num_models: int = int(kwargs.get("num_surrogate_models", 3))
        self.lr: float = float(kwargs.get("nn_learning_rate", 1e-3))
        self.nn_hidden_sizes: Tuple[int, ...] = tuple(kwargs.get("nn_hidden_sizes", [32, 32, 16]))

        # Exploit/Explore split
        ef = kwargs.get("exploit_fraction", 0.5)
        try:
            ef = float(ef)
        except Exception:
            ef = 0.5
        self.exploit_fraction: float = max(0.0, min(1.0, ef))

        # Constraints
        self.use_constraints: bool = bool(kwargs.get("use_constraints", True))

        # Schrittweiten (erforderlich)
        self.step_widths: Dict[str, float] = dict(kwargs.get("step_widths", {}))
        if not self.step_widths:
            raise ValueError("ActiveLearningAlgorithm: 'step_widths' müssen in der AL-Config gesetzt sein.")
        for p in self.search_space:
            name = p["name"]
            if name not in self.step_widths or float(self.step_widths[name]) <= 0.0:
                raise ValueError(f"ActiveLearningAlgorithm: fehlende/ungültige step_width für '{name}'.")

        # NSGA reward parameters (alle Ziele minimiert)
        self.objectives: Dict[str, str] = kwargs.get(
            "objectives",
            {"consumption": "min", "ela3": "min", "ela4": "min", "ela5": "min"},
        )
        self.obj_keys: List[str] = list(self.objectives.keys())

        w_r_val = float(kwargs.get("nsga_rank_weight", 0.75))
        self.w_r: float = float(np.clip(w_r_val, 0.0, 1.0))
        self.w_c: float = 1.0 - self.w_r
        self.nsga_eps: float = float(kwargs.get("nsga_eps", 1e-12))

        # Kandidatenpool aus Schrittweiten
        self.X_pool: np.ndarray = self._build_grid_from_steps(self.search_space, self.step_widths)
        self.pool_size: int = int(self.X_pool.shape[0])

        # Labels & Masken
        self.y_pool: np.ndarray = np.full(self.pool_size, np.nan, dtype=float)  # NaN = unlabeled
        self.invalid_actions: Set[Tuple[float, ...]] = set()
        self.invalid_mask: np.ndarray = np.zeros(self.pool_size, dtype=bool)

        # Surrogate committee
        self.models: List[Sequential] = [self._build_nn_model() for _ in range(self.num_models)]

        # Book-keeping
        self.eval_count: int = 0                   # zählt GÜLTIGE Evaluations
        self.best_reward_so_far: float = float("-inf")

        # Archiv gültiger Zielvektoren (für NSGA-Reward)
        self.archive_all_df: pd.DataFrame = pd.DataFrame(columns=self.obj_keys)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _action_key(x_env: np.ndarray, decimals: int = 6) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(x_env, dtype=np.float64), decimals=decimals).tolist())

    @staticmethod
    def _is_valid_sim(sim: Tuple[float, float, float, float]) -> bool:
        """Erlaubt negative Zielwerte; verwirft nur NaN/Inf."""
        arr = np.asarray(sim, dtype=np.float64)
        return np.isfinite(arr).all()

    def _grid_values(self, lo: float, hi: float, step: float) -> np.ndarray:
        """Inklusive Gitterwerte [lo, hi] mit Schrittweite `step` (robust gegen Rundung)."""
        n = int(np.floor((hi - lo) / step + 1e-12)) + 1
        vals = lo + np.arange(n, dtype=np.float64) * step
        vals = np.round(vals, 6)
        vals = vals[vals <= hi + 1e-12]
        return vals

    def _build_grid_from_steps(self, search_space: List[Dict[str, Any]], step_widths: Dict[str, float]) -> np.ndarray:
        """Kartesisches Produkt aus Gitterwerten je Dimension; Constraints anwenden; Duplikate entfernen."""
        axes: List[np.ndarray] = []
        names: List[str] = []
        for p in search_space:
            name = p["name"]
            lo, hi = float(p["min"]), float(p["max"])
            step = float(step_widths[name])
            axes.append(self._grid_values(lo, hi, step))
            names.append(name)

        # Kartesisches Produkt
        mesh = np.meshgrid(*axes, indexing="ij")
        grid = np.stack([m.reshape(-1) for m in mesh], axis=1)  # [N, D]

        # Constraints anwenden + Duplikate entfernen (deterministisch runden)
        uniq: List[np.ndarray] = []
        seen: Set[Tuple[float, ...]] = set()
        for x in grid:
            xx = enforce_gear_descending(x, self.search_space) if self.use_constraints else x
            key = self._action_key(xx)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(xx.astype(np.float64))
        if not uniq:
            return np.zeros((0, len(search_space)), dtype=np.float64)
        X = np.vstack(uniq)
        return X

    def _build_nn_model(self) -> Sequential:
        """MLP-Regression mit Input-Normalisierung, adaptiert auf X_pool."""
        normalizer = Normalization(axis=-1)
        normalizer.adapt(self.X_pool)

        layers = [Input(shape=(self.X_pool.shape[1],)), normalizer]
        for u in self.nn_hidden_sizes:
            layers.append(Dense(int(u), activation="relu"))
        layers.append(Dense(1, activation="linear"))

        model = Sequential(layers)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss="mse")
        return model

    def _fit_surrogates(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits each surrogate model on the currently labeled set."""
        for m in self.models:
            m.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)],
            )

    def _predict_committee(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns mean/std across committee predictions."""
        preds = np.array([m.predict(X, verbose=0).flatten() for m in self.models])
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0) if preds.shape[0] > 1 else np.zeros_like(mean)
        return mean, std

    def _select_query_indices(
        self, unlabeled_idx: np.ndarray, y_mean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Teilt Batch in exploit (Top-Reward) und explore (random)."""
        n_total_possible = len(unlabeled_idx)
        n_exploit = int(round(self.batch_size * self.exploit_fraction))
        n_exploit = max(0, min(n_exploit, min(self.batch_size, n_total_possible)))
        n_explore = max(0, min(self.batch_size - n_exploit, n_total_possible - n_exploit))

        if n_exploit > 0:
            top_order = np.argsort(y_mean)[-n_exploit:]
            exploit_idx = unlabeled_idx[top_order]
        else:
            exploit_idx = np.array([], dtype=int)

        if n_explore > 0:
            mask = np.ones(n_total_possible, dtype=bool)
            if exploit_idx.size > 0:
                sel_positions = np.where(np.in1d(unlabeled_idx, exploit_idx))[0]
                mask[sel_positions] = False
            remaining_positions = np.where(mask)[0]
            explore_positions = (
                self.random_state.choice(remaining_positions, size=n_explore, replace=False)
                if remaining_positions.size > 0
                else np.array([], dtype=int)
            )
            explore_idx = unlabeled_idx[explore_positions]
        else:
            explore_idx = np.array([], dtype=int)

        query_idx = np.unique(np.concatenate((exploit_idx, explore_idx)))
        return query_idx, exploit_idx, explore_idx

    def _log_record(
        self,
        x: np.ndarray,
        sim: Tuple[float, float, float, float],
        reward: float,
        rank: int,
        cd_raw: float,
        cd_tilde: float,
        *,
        phase: str,
        cycle: int,
        select_mode: str,
        t_env_ms: float,
        wall_time_s: float,
        y_pred: Optional[float],
        y_pred_std: Optional[float],
    ) -> None:
        rec: Dict[str, Any] = {
            "algo": self.algo_name,
            "seed": self.seed,
            "evaluation": int(self.eval_count),
            "timestamp": time.time(),
            # timing
            "t_env_ms": float(t_env_ms),
            "wall_time_s": float(wall_time_s),
            # action (after constraints)
            "p1_final_drive_ratio": float(x[0]),
            "p2_roll_radius": float(x[1]),
            "p3_gear3_diff": float(x[2]),
            "p4_gear4_diff": float(x[3]),
            "p5_gear5": float(x[4]),
            # objectives
            "consumption": float(sim[0]),
            "ela3": float(sim[1]),
            "ela4": float(sim[2]),
            "ela5": float(sim[3]),
            # reward diagnostics
            "reward": float(reward),
            "rank": int(rank),
            "cd_raw": float(cd_raw),
            "cd_tilde": float(cd_tilde),
            # AL context
            "phase": phase,
            "cycle": int(cycle),
            "select_mode": str(select_mode),
            # surrogate predictions at query time
            "y_pred": None if y_pred is None else float(y_pred),
            "y_pred_std": None if y_pred_std is None else float(y_pred_std),
        }
        self.results_list.append(rec)

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------
    def run(self, budget: int) -> None:
        """
        Läuft, bis `budget` GÜLTIGE Evaluierungen erreicht sind.
        Ungültig = nur nicht-endliche Simulationsoutputs (NaN/Inf).
        """
        # ------------------ Initial labeling ------------------
        init_targets = min(self.initial_label_count, self.pool_size)
        # Nur Kandidaten, die (noch) nicht invalid markiert sind
        candidates = np.where(~self.invalid_mask)[0]
        init_order = self.random_state.permutation(candidates)
        init_added = 0

        for i in init_order:
            if self.eval_count >= budget or init_added >= init_targets:
                break
            if self.invalid_mask[i] or not np.isnan(self.y_pool[i]):
                continue

            step_t0 = time.perf_counter()
            xx = self.X_pool[i]
            key = self._action_key(xx)

            # Bekannter Invalid?
            if key in self.invalid_actions:
                self.invalid_mask[i] = True
                continue

            t_env0 = time.perf_counter()
            sim = self.env.step(xx)
            t_env_ms = (time.perf_counter() - t_env0) * 1000.0

            if not self._is_valid_sim(sim):
                # Nur NaN/Inf verwerfen
                self.invalid_actions.add(key)
                self.invalid_mask[i] = True
                continue

            # --- gültig ---
            y_point = {"consumption": float(sim[0]), "ela3": float(sim[1]),
                       "ela4": float(sim[2]), "ela5": float(sim[3])}

            reward, rank, cd_raw, cd_tilde = nsga_rank_cd_reward(
                self.archive_all_df, y_point, self.objectives,
                w_r=self.w_r, w_c=self.w_c, eps=self.nsga_eps
            )

            self.archive_all_df = pd.concat([self.archive_all_df, pd.DataFrame([y_point])], ignore_index=True)
            self.y_pool[i] = reward

            self.eval_count += 1
            init_added += 1
            self.best_reward_so_far = max(self.best_reward_so_far, reward)

            wall_time_s = time.perf_counter() - step_t0
            self._log_record(
                x=xx, sim=sim, reward=float(reward), rank=int(rank),
                cd_raw=float(cd_raw), cd_tilde=float(cd_tilde),
                phase="init", cycle=0, select_mode="random",
                t_env_ms=float(t_env_ms), wall_time_s=float(wall_time_s),
                y_pred=None, y_pred_std=None
            )

        # ------------------ Active Learning cycles ------------------
        for cycle in range(1, self.num_cycles + 1):
            if self.eval_count >= budget:
                break

            labeled_mask = ~np.isnan(self.y_pool)
            if not np.any(labeled_mask):
                break

            X_train = self.X_pool[labeled_mask]
            y_train = self.y_pool[labeled_mask]
            self._fit_surrogates(X_train, y_train)

            unlabeled_idx = np.where((~labeled_mask) & (~self.invalid_mask))[0]
            if unlabeled_idx.size == 0:
                break

            X_unlabeled = self.X_pool[unlabeled_idx]
            y_mean, y_std = self._predict_committee(X_unlabeled)

            query_idx, exploit_idx, explore_idx = self._select_query_indices(unlabeled_idx, y_mean)
            if query_idx.size == 0:
                break
            exploit_set = set(map(int, exploit_idx.tolist()))

            for i in query_idx:
                if self.eval_count >= budget:
                    break
                if self.invalid_mask[i] or not np.isnan(self.y_pool[i]):
                    continue

                step_t0 = time.perf_counter()
                xx = self.X_pool[i]
                key = self._action_key(xx)

                if key in self.invalid_actions:
                    self.invalid_mask[i] = True
                    continue

                t_env0 = time.perf_counter()
                sim = self.env.step(xx)
                t_env_ms = (time.perf_counter() - t_env0) * 1000.0

                if not self._is_valid_sim(sim):
                    # Nur NaN/Inf verwerfen
                    self.invalid_actions.add(key)
                    self.invalid_mask[i] = True
                    continue

                # --- gültig ---
                y_point = {"consumption": float(sim[0]), "ela3": float(sim[1]),
                           "ela4": float(sim[2]), "ela5": float(sim[3])}

                reward, rank, cd_raw, cd_tilde = nsga_rank_cd_reward(
                    self.archive_all_df, y_point, self.objectives,
                    w_r=self.w_r, w_c=self.w_c, eps=self.nsga_eps
                )

                self.archive_all_df = pd.concat([self.archive_all_df, pd.DataFrame([y_point])], ignore_index=True)
                self.y_pool[i] = reward

                self.eval_count += 1
                self.best_reward_so_far = max(self.best_reward_so_far, reward)

                wall_time_s = time.perf_counter() - step_t0

                # Vorhersagen-Position
                pos = np.where(unlabeled_idx == i)[0]
                y_hat = float(y_mean[pos[0]]) if pos.size > 0 else None
                y_hat_std = float(y_std[pos[0]]) if pos.size > 0 else None

                self._log_record(
                    x=xx, sim=sim, reward=float(reward), rank=int(rank),
                    cd_raw=float(cd_raw), cd_tilde=float(cd_tilde),
                    phase="cycle", cycle=int(cycle),
                    select_mode="exploit" if int(i) in exploit_set else "explore",
                    t_env_ms=float(t_env_ms), wall_time_s=float(wall_time_s),
                    y_pred=y_hat, y_pred_std=y_hat_std
                )
