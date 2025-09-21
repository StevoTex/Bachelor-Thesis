# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from src.utils.constraints import enforce_gear_descending
from src.utils.objectives import nsga_rank_cd_reward  # Central NSGA-style reward: rank + normalized crowding

# Use double precision throughout
tf.keras.backend.set_floatx("float64")


class A3CAlgorithm:
    """
    Asynchronous Advantage Actor-Critic (A3C) für das Fahrzeug-Optimierungsproblem.

    WICHTIG (Validierung):
    - Simulator liefert 4 Ziele (consumption, ela3, ela4, ela5), alle zu minimieren.
    - Änderung: Negative Zielwerte sind ERLAUBT und zählen ganz normal.
    - Verworfen wird NUR, wenn ein Zielwert nicht endlich ist (NaN/Inf):
        * kein Log-Eintrag
        * kein Archiveintrag
        * keine Trainingsnutzung
        * Budget-Zähler wird zurückgerollt
        * Aktion wird in einer globalen "invalid actions"-Menge zwischengespeichert,
          so dass dieselbe Parameterkombination zukünftig übersprungen wird.
    """

    def __init__(self, env, search_space, **kwargs):
        # Factory für per-Worker-Umgebungen
        self.env_creator = lambda: type(env)(exe_file_name=env.exe_file_name)

        self.search_space = search_space
        self.seed = int(kwargs.get("seed", 0))
        self.algo_name = "A3C"

        # Action/Observation
        self.action_low = np.array([p["min"] for p in self.search_space], dtype=np.float64)
        self.action_high = np.array([p["max"] for p in self.search_space], dtype=np.float64)
        self.action_shape = (len(self.search_space),)
        self.observation_shape = (4,)  # simulator outputs 4 objectives

        # Flags
        self.use_constraints = bool(kwargs.get("use_constraints", True))

        # NSGA reward parameters (alle Ziele minimiert)
        self.objectives: Dict[str, str] = kwargs.get(
            "objectives",
            {"consumption": "min", "ela3": "min", "ela4": "min", "ela5": "min"},
        )
        self.obj_keys: List[str] = list(self.objectives.keys())
        w_r_cfg = kwargs.get("nsga_rank_weight", kwargs.get("rank_weight", 0.8))
        try:
            w_r_val = float(w_r_cfg)
        except Exception:
            w_r_val = 0.8
        self.w_r: float = float(np.clip(w_r_val, 0.0, 1.0))
        self.w_c: float = 1.0 - self.w_r
        self.nsga_eps: float = float(kwargs.get("nsga_eps", 1e-12))

        # Hyperparameter
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.lr = float(kwargs.get("lr", 1e-4))
        self.t_max = int(kwargs.get("t_max", 5))
        self.num_workers = int(kwargs.get("num_workers", 4))
        self.actor_critic_units = tuple(kwargs.get("actor_critic_units", [128, 128]))
        self.value_loss_factor = float(kwargs.get("value_loss_factor", 0.5))
        self.entropy_beta = float(kwargs.get("entropy_beta", 0.01))

        # Globales Netz & Optimizer
        self.global_model = self._build_actor_critic(units=self.actor_critic_units)
        self.global_optimizer = Adam(learning_rate=self.lr)

        # Thread-Synchronisation
        self.opt_lock = threading.Lock()     # serialize optimizer updates
        self.budget_lock = threading.Lock()  # serialize budget reservations

        # Optimizer-Slots aufwärmen
        zero_grads = [tf.zeros_like(v) for v in self.global_model.trainable_variables]
        self.global_optimizer.apply_gradients(zip(zero_grads, self.global_model.trainable_variables))

        # Shared counters/state
        self.global_eval_count = tf.Variable(0, dtype=tf.int32)  # zählt gültige, reservierte Evaluations
        self.global_budget = tf.Variable(0, dtype=tf.int32)

        # Shared results buffer
        self.results_list: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()

        # Shared archive aller gültigen Zielvektoren
        self.archive_all_df = pd.DataFrame(columns=self.obj_keys)
        self.archive_lock = threading.Lock()

        # Shared cache: invalid actions (deterministisches Modell)
        self.invalid_actions: Set[Tuple[float, ...]] = set()
        self.invalid_lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Networks
    # -------------------------------------------------------------------------
    def _build_actor_critic(self, units: Tuple[int, ...]) -> Model:
        """Shared trunk + Policy(Logits [mean, log_std]) + Value."""
        s = Input(shape=self.observation_shape)
        x = s
        for u in units:
            x = Dense(int(u), activation="relu")(x)
        policy_logits = Dense(self.action_shape[0] * 2)(x)  # [mean, log-std] je Aktionsdimension
        value = Dense(1)(x)
        return Model(inputs=s, outputs=[policy_logits, value])

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------
    def run(self, budget: int) -> None:
        """Startet Worker und blockiert bis das globale Budget an GÜLTIGEN Auswertungen erreicht ist."""
        self.global_budget.assign(int(budget))

        # Worker erzeugen
        workers = [
            Worker(
                env_creator=self.env_creator,
                global_model=self.global_model,
                global_optimizer=self.global_optimizer,
                opt_lock=self.opt_lock,
                budget_lock=self.budget_lock,
                results_lock=self.results_lock,
                archive_lock=self.archive_lock,
                invalid_lock=self.invalid_lock,
                # shared state
                global_eval_count=self.global_eval_count,
                global_budget=self.global_budget,
                results_list=self.results_list,
                archive_all_df=self.archive_all_df,
                invalid_actions=self.invalid_actions,
                # identifiers & config
                worker_id=i,
                search_space=self.search_space,
                gamma=self.gamma,
                t_max=self.t_max,
                value_loss_factor=self.value_loss_factor,
                entropy_beta=self.entropy_beta,
                actor_critic_units=self.actor_critic_units,
                action_low=self.action_low,
                action_high=self.action_high,
                use_constraints=self.use_constraints,
                # NSGA reward
                objectives=self.objectives,
                nsga_w_r=self.w_r,
                nsga_w_c=self.w_c,
                nsga_eps=self.nsga_eps,
                algo_name=self.algo_name,
                seed=self.seed,
            )
            for i in range(self.num_workers)
        ]

        for w in workers:
            w.start()

        # Einfaches Fortschritts-Polling
        while self.global_eval_count.numpy() < self.global_budget.numpy():
            time.sleep(0.25)

        for w in workers:
            w.join()


# ============================================================================
# Worker thread
# ============================================================================

class Worker(threading.Thread):
    """Ein A3C-Worker (eigene Env, lokales Netz, kurze Rollouts, globales Update)."""

    def __init__(
        self,
        env_creator,
        global_model: Model,
        global_optimizer: Adam,
        opt_lock: threading.Lock,
        budget_lock: threading.Lock,
        results_lock: threading.Lock,
        archive_lock: threading.Lock,
        invalid_lock: threading.Lock,
        global_eval_count: tf.Variable,
        global_budget: tf.Variable,
        results_list: List[Dict[str, Any]],
        archive_all_df: pd.DataFrame,
        invalid_actions: Set[Tuple[float, ...]],
        worker_id: int,
        **kwargs,
    ):
        super().__init__()

        # Env & globale Handles
        self.env = env_creator()
        self.global_model = global_model
        self.global_optimizer = global_optimizer
        self.opt_lock = opt_lock
        self.budget_lock = budget_lock
        self.results_lock = results_lock
        self.archive_lock = archive_lock
        self.invalid_lock = invalid_lock
        self.global_eval_count = global_eval_count
        self.global_budget = global_budget
        self.worker_id = int(worker_id)

        # Config
        self.search_space = kwargs["search_space"]
        self.gamma = float(kwargs["gamma"])
        self.t_max = int(kwargs["t_max"])
        self.value_loss_factor = float(kwargs["value_loss_factor"])
        self.entropy_beta = float(kwargs["entropy_beta"])
        self.actor_critic_units = tuple(kwargs["actor_critic_units"])
        self.action_low = kwargs["action_low"]
        self.action_high = kwargs["action_high"]
        self.use_constraints = bool(kwargs.get("use_constraints", True))

        # NSGA reward
        self.objectives: Dict[str, str] = kwargs["objectives"]
        self.w_r: float = float(kwargs.get("nsga_w_r", 0.7))
        self.w_c: float = float(kwargs.get("nsga_w_c", 0.3))
        self.nsga_eps: float = float(kwargs.get("nsga_eps", 1e-12))
        self.obj_keys: List[str] = list(self.objectives.keys())

        # Logging & shared buffers
        self.algo_name = kwargs.get("algo_name", "A3C")
        self.seed = int(kwargs.get("seed", 0))
        self.results_list: List[Dict[str, Any]] = results_list

        # Shared archive (nur gültige Punkte)
        self.archive_all_df: pd.DataFrame = archive_all_df

        # Shared invalid actions cache
        self.invalid_actions: Set[Tuple[float, ...]] = invalid_actions

        # Lokales Netz
        self.action_shape = (len(self.search_space),)
        self.observation_shape = (4,)
        self.local_model = self._build_local_from_global()

        # Rollout counter
        self.rollout_id = 0

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _build_local_from_global(self) -> Model:
        local = clone_model(self.global_model)
        local.build((None,) + self.observation_shape)
        local.set_weights(self.global_model.get_weights())
        return local

    @staticmethod
    def _map_to_bounds(a_tanh: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        a_bound = (high - low) / 2.0
        a_shift = (high + low) / 2.0
        return a_tanh * a_bound + a_shift

    def _apply_constraints(self, x: np.ndarray) -> np.ndarray:
        return enforce_gear_descending(x, self.search_space) if self.use_constraints else x

    @staticmethod
    def _is_valid_sim(sim: Tuple[float, float, float, float]) -> bool:
        """
        Änderung: nur Endlichkeit prüfen, negative Werte sind erlaubt.
        """
        arr = np.asarray(sim, dtype=np.float64)
        return np.isfinite(arr).all()

    @staticmethod
    def _action_key(x: np.ndarray, decimals: int = 6) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(x, dtype=np.float64), decimals=decimals).tolist())

    def _log_step(
        self,
        *,
        action_vec: np.ndarray,
        sim: Tuple[float, float, float, float],
        reward: float,
        rank: int,
        cd_raw: float,
        cd_tilde: float,
        t_env_ms: float,
        wall_time_s: float,
        evaluation: int,
        phase: str,
    ) -> None:
        rec = {
            "algo": self.algo_name,
            "seed": self.seed,
            "evaluation": int(evaluation),
            "timestamp": time.time(),
            # timing
            "t_env_ms": float(t_env_ms),
            "wall_time_s": float(wall_time_s),
            # action (after constraints)
            "p1_final_drive_ratio": float(action_vec[0]),
            "p2_roll_radius": float(action_vec[1]),
            "p3_gear3_diff": float(action_vec[2]),
            "p4_gear4_diff": float(action_vec[3]),
            "p5_gear5": float(action_vec[4]),
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
            # context
            "phase": phase,
            "worker_id": self.worker_id,
            "rollout_id": self.rollout_id,
        }
        with self.results_lock:
            self.results_list.append(rec)

    def _log_update(
        self,
        *,
        loss_actor: float,
        loss_critic: float,
        entropy_mean: float,
        total_loss: float,
        update_time_ms: float,
    ) -> None:
        rec = {
            "algo": self.algo_name,
            "seed": self.seed,
            "timestamp": time.time(),
            "phase": "update",
            "worker_id": self.worker_id,
            "rollout_id": self.rollout_id,
            "loss_actor": float(loss_actor),
            "loss_critic": float(loss_critic),
            "entropy_mean": float(entropy_mean),
            "total_loss": float(total_loss),
            "update_time_ms": float(update_time_ms),
        }
        with self.results_lock:
            self.results_list.append(rec)

    # ---------------------------------------------------------------------
    # Thread main
    # ---------------------------------------------------------------------
    def run(self) -> None:
        current_state = np.zeros(self.observation_shape, dtype=np.float64)

        while True:
            if self.global_eval_count.numpy() >= self.global_budget.numpy():
                break

            # Sync lokale Gewichte
            self.local_model.set_weights(self.global_model.get_weights())
            self.rollout_id += 1
            memory: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, Dict[str, Any]]] = []

            # -------- Interaction (bis t_max oder Budget) --------
            for _ in range(self.t_max):
                if self.global_eval_count.numpy() >= self.global_budget.numpy():
                    break

                step_t0 = time.perf_counter()

                # Policy forward
                logits, _ = self.local_model(np.expand_dims(current_state, axis=0))
                logits = tf.squeeze(logits)  # shape [2*A]
                action_mean, action_log_std = tf.split(logits, 2, axis=-1)
                action_std = tf.nn.softplus(action_log_std)
                dist = tfp.distributions.Normal(loc=action_mean, scale=action_std)

                # Mehrfach probieren, bekannte "invalid actions" zu vermeiden
                max_resamples = 20
                attempt = 0
                final_action = None
                while attempt < max_resamples:
                    sampled = dist.sample()
                    a_tanh = tf.tanh(sampled).numpy()
                    cand = self._map_to_bounds(a_tanh, self.action_low, self.action_high)
                    cand = self._apply_constraints(cand)
                    key = self._action_key(cand)
                    with self.invalid_lock:
                        known_bad = key in self.invalid_actions
                    if not known_bad:
                        final_action = cand
                        break
                    attempt += 1
                if final_action is None:
                    # Fallback: uniforme Aktion (ebenfalls Constraints)
                    cand = np.random.uniform(self.action_low, self.action_high, size=self.action_shape[0])
                    final_action = self._apply_constraints(cand)

                # Reserviere Evaluation-Token
                with self.budget_lock:
                    if self.global_eval_count.numpy() >= self.global_budget.numpy():
                        break
                    self.global_eval_count.assign_add(1)
                    step_idx = int(self.global_eval_count.numpy())

                # Environment step
                t_env0 = time.perf_counter()
                sim = self.env.step(final_action)
                t_env_ms = (time.perf_counter() - t_env0) * 1000.0

                # Validierung
                if not self._is_valid_sim(sim):
                    # Aktion als ungültig merken & Budget rückgängig machen
                    key = self._action_key(final_action)
                    with self.invalid_lock:
                        self.invalid_actions.add(key)
                    with self.budget_lock:
                        # Schutz vor negativen Werten
                        cur = int(self.global_eval_count.numpy())
                        if cur > 0:
                            self.global_eval_count.assign(cur - 1)
                    # Nichts loggen, nicht trainieren, kein Archiveintrag
                    continue

                # --- Ab hier: gültiger Punkt ---
                point = {
                    "consumption": float(sim[0]),
                    "ela3": float(sim[1]),
                    "ela4": float(sim[2]),
                    "ela5": float(sim[3]),
                }

                # NSGA-Reward + Archive-Update
                with self.archive_lock:
                    reward, rank, cd_raw, cd_tilde = nsga_rank_cd_reward(
                        self.archive_all_df, point, self.objectives,
                        w_r=self.w_r, w_c=self.w_c, eps=self.nsga_eps
                    )
                    self.archive_all_df.loc[len(self.archive_all_df)] = point

                next_state = np.array(sim[:4], dtype=np.float64)
                done = False  # keine Terminal-Condition

                wall_time_s = time.perf_counter() - step_t0

                # Transition in den Rollout-Speicher
                memory.append((
                    current_state.copy(),
                    final_action.copy(),
                    float(reward),
                    next_state.copy(),
                    done,
                    {"t_env_ms": float(t_env_ms), "evaluation": step_idx,
                     "rank": int(rank), "cd_raw": float(cd_raw), "cd_tilde": float(cd_tilde)}
                ))

                # Logging: nur gültige Punkte
                self._log_step(
                    action_vec=final_action,
                    sim=sim,
                    reward=float(reward),
                    rank=int(rank),
                    cd_raw=float(cd_raw),
                    cd_tilde=float(cd_tilde),
                    t_env_ms=float(t_env_ms),
                    wall_time_s=float(wall_time_s),
                    evaluation=step_idx,
                    phase="rollout",
                )

                # Nächster Zustand
                current_state = next_state

            if not memory:
                continue

            # -------- A3C-Update --------
            _, bootstrap_value_t = self.local_model(np.expand_dims(memory[-1][3], axis=0))
            bootstrap_value = float(tf.squeeze(bootstrap_value_t).numpy())

            rewards = [m[2] for m in memory]
            dones = [m[4] for m in memory]
            returns: List[float] = []
            R = bootstrap_value
            for r, _ in zip(reversed(rewards), reversed(dones)):
                R = r + self.gamma * R
                returns.append(R)
            returns.reverse()

            states = np.array([m[0] for m in memory], dtype=np.float64)        # [T, S]
            taken_actions = np.array([m[1] for m in memory], dtype=np.float64)  # [T, A]

            # Inverse tanh (für Log-Prob)
            a_bound = (self.action_high - self.action_low) / 2.0
            a_shift = (self.action_high + self.action_low) / 2.0
            tanh_actions = (taken_actions - a_shift) / a_bound
            raw_actions = np.arctanh(np.clip(tanh_actions, -0.999, 0.999))  # [T, A]

            update_t0 = time.perf_counter()
            with tf.GradientTape() as tape:
                logits, values = self.local_model(states)         # values: [T,1]
                values = tf.squeeze(values)                       # [T]

                action_mean, action_log_std = tf.split(logits, 2, axis=-1)  # [T, A]
                action_std = tf.nn.softplus(action_log_std)
                dist = tfp.distributions.Normal(loc=action_mean, scale=action_std)

                # Log-prob der pre-squash Actions
                log_probs = dist.log_prob(raw_actions)            # [T, A]
                log_probs = tf.reduce_sum(log_probs, axis=1)      # [T]

                returns_t = tf.convert_to_tensor(returns, dtype=tf.float64)  # [T]
                advantage = returns_t - values                    # [T]

                # Verluste
                actor_loss_vec = -log_probs * tf.stop_gradient(advantage)
                critic_loss_vec = tf.square(advantage)
                entropy_vec = tf.reduce_sum(dist.entropy(), axis=1)

                loss_actor = tf.reduce_mean(actor_loss_vec)
                loss_critic = tf.reduce_mean(critic_loss_vec) * self.value_loss_factor
                entropy_term = -self.entropy_beta * tf.reduce_mean(entropy_vec)

                total_loss = loss_actor + loss_critic + entropy_term

            grads = tape.gradient(total_loss, self.local_model.trainable_variables)

            # Apply Gradients auf GLOBAL (thread-safe)
            pairs: List[Tuple[tf.Tensor, tf.Variable]] = []
            for g, gv in zip(grads, self.global_model.trainable_variables):
                if g is not None:
                    pairs.append((g, gv))

            with self.opt_lock:
                if pairs:
                    self.global_optimizer.apply_gradients(pairs)

            update_time_ms = (time.perf_counter() - update_t0) * 1000.0

            # Update-Logging
            self._log_update(
                loss_actor=float(loss_actor.numpy()),
                loss_critic=float((tf.reduce_mean(critic_loss_vec) * self.value_loss_factor).numpy()),
                entropy_mean=float(tf.reduce_mean(entropy_vec).numpy()),
                total_loss=float(total_loss.numpy()),
                update_time_ms=float(update_time_ms),
            )
