# -*- coding: utf-8 -*-
"""Soft Actor-Critic (SAC) for vehicle design with NSGA-style reward.

The simulator returns four objectives to minimize (consumption, ela3, ela4, ela5).
Negative values are allowed; only non-finite outputs are discarded and the
corresponding actions are cached to avoid re-evaluation. TensorFlow uses float64.
"""
from __future__ import annotations

import time
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.utils.constraints import enforce_gear_descending
from src.utils.objectives import nsga_rank_cd_reward

tf.keras.backend.set_floatx("float64")


class SoftActorCritic:
    """Soft Actor-Critic with an NSGA-compatible reward (rank + normalized crowding)."""

    def __init__(self, env, search_space, **kwargs):
        self.env = env
        self.search_space = search_space

        self.tb = kwargs.get("tb", None)

        self.seed = int(kwargs.get("seed", 0))
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.algo_name = "SAC"

        self.action_low = np.array([p["min"] for p in self.search_space], dtype=np.float64)
        self.action_high = np.array([p["max"] for p in self.search_space], dtype=np.float64)
        self.action_shape: Tuple[int, ...] = (len(self.search_space),)
        self.observation_shape: Tuple[int, ...] = (4,)

        self.use_constraints: bool = bool(kwargs.get("use_constraints", True))

        self.objectives: Dict[str, str] = kwargs.get(
            "objectives",
            {"consumption": "min", "ela3": "min", "ela4": "min", "ela5": "min"},
        )
        self.obj_keys: List[str] = list(self.objectives.keys())
        w_r_val = float(kwargs.get("nsga_rank_weight", 0.7))
        self.w_r: float = float(np.clip(w_r_val, 0.0, 1.0))
        self.w_c: float = 1.0 - self.w_r
        self.nsga_eps: float = float(kwargs.get("nsga_eps", 1e-12))

        self.gamma: float = float(kwargs.get("gamma", 0.99))
        self.tau: float = float(kwargs.get("tau", 0.005))
        self.batch_size: int = int(kwargs.get("batch_size", 64))
        self.memory_capacity: int = int(kwargs.get("memory_capacity", 100000))
        self.initial_random_steps: int = int(kwargs.get("initial_random_steps", 100))

        actor_units = tuple(kwargs.get("actor_units", [256, 256]))
        critic_units = tuple(kwargs.get("critic_units", [256, 256]))
        lr_actor = float(kwargs.get("lr_actor", 3e-4))
        lr_critic = float(kwargs.get("lr_critic", 3e-4))
        lr_alpha = float(kwargs.get("lr_alpha", 3e-4))

        self.auto_alpha: bool = bool(kwargs.get("auto_alpha", True))
        alpha_init = float(kwargs.get("alpha", 0.2))

        self.memory: deque = deque(maxlen=self.memory_capacity)
        self.action_bound = (self.action_high - self.action_low) / 2.0
        self.action_shift = (self.action_high + self.action_low) / 2.0
        self.log_std_min, self.log_std_max = -20.0, 2.0

        self.actor: Model = self._build_actor(actor_units)
        self.critic_1: Model = self._build_critic(critic_units)
        self.critic_2: Model = self._build_critic(critic_units)
        self.critic_target_1: Model = self._build_critic(critic_units)
        self.critic_target_2: Model = self._build_critic(critic_units)

        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.critic_optimizer_1 = Adam(learning_rate=lr_critic)
        self.critic_optimizer_2 = Adam(learning_rate=lr_critic)

        self._update_target_weights(self.critic_1, self.critic_target_1, tau=1.0)
        self._update_target_weights(self.critic_2, self.critic_target_2, tau=1.0)

        if self.auto_alpha:
            self.target_entropy = -float(np.prod(self.action_shape))
            self.log_alpha = tf.Variable(0.0, dtype=tf.float64)
            self.alpha = tf.exp(self.log_alpha)
            self.alpha_optimizer = Adam(learning_rate=lr_alpha)
        else:
            self.log_alpha = None
            self.alpha = tf.Variable(alpha_init, dtype=tf.float64)

        self.eval_count: int = 0
        self.results_list: List[Dict[str, Any]] = []
        self._recent_actions: deque = deque(maxlen=1000)

        self.archive_all_df: pd.DataFrame = pd.DataFrame(columns=self.obj_keys)

        self.invalid_actions: Set[Tuple[float, ...]] = set()

        if self.tb:
            self.tb.text("SAC/Info", "SoftActorCritic initialized (NSGA rank+CD reward).", step=0)
            self.tb.scalar("SAC/NSGA/w_r", float(self.w_r), step=0)
            self.tb.scalar("SAC/NSGA/w_c", float(self.w_c), step=0)

        print(f"SAC initialized — w_r={self.w_r:.3f}, w_c={self.w_c:.3f}")

    # ---------------------------------------------------------------------
    # Network builders & utils
    # ---------------------------------------------------------------------
    def _build_actor(self, units: Tuple[int, ...]) -> Model:
        """Gaussian policy network returning mean and log-std per action dimension."""
        s = Input(shape=self.observation_shape)
        x = Dense(units[0], activation="relu")(s)
        for u in units[1:]:
            x = Dense(u, activation="relu")(x)
        mean = Dense(self.action_shape[0])(x)
        log_std = Dense(self.action_shape[0])(x)
        return Model(inputs=s, outputs=[mean, log_std])

    def _build_critic(self, units: Tuple[int, ...]) -> Model:
        """State-action Q-network."""
        s_in = Input(shape=self.observation_shape)
        a_in = Input(shape=self.action_shape)
        x = Concatenate(axis=-1)([s_in, a_in])
        x = Dense(units[0], activation="relu")(x)
        for u in units[1:]:
            x = Dense(u, activation="relu")(x)
        q = Dense(1)(x)
        return Model(inputs=[s_in, a_in], outputs=q)

    def _update_target_weights(self, model: Model, target_model: Model, tau: float) -> None:
        """Polyak averaging: target ← tau * online + (1 - tau) * target."""
        w = model.get_weights()
        tw = target_model.get_weights()
        for i in range(len(tw)):
            tw[i] = w[i] * tau + tw[i] * (1.0 - tau)
        target_model.set_weights(tw)

    @staticmethod
    def _is_valid_sim(sim: Tuple[float, float, float, float]) -> bool:
        """Return True iff all values are finite; negatives are allowed."""
        arr = np.asarray(sim, dtype=np.float64)
        return np.isfinite(arr).all()

    @staticmethod
    def _action_key(x: np.ndarray, decimals: int = 6) -> Tuple[float, ...]:
        """Stable key for caching actions with limited precision."""
        return tuple(np.round(np.asarray(x, dtype=np.float64), decimals=decimals).tolist())

    def _process_actions(
        self, mean: tf.Tensor, log_std: tf.Tensor, eps: float = 1e-6
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Sample Tanh-squashed actions, return env-scaled actions and log-probs."""
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.exp(log_std)
        raw = mean + tf.random.normal(shape=tf.shape(mean), dtype=tf.float64) * std
        log_prob_u = tfp.distributions.Normal(loc=mean, scale=std).log_prob(raw)
        a_tanh = tf.tanh(raw)
        log_prob = tf.reduce_sum(
            log_prob_u - tf.math.log(1.0 - a_tanh ** 2 + eps), axis=1, keepdims=True
        )
        actions = a_tanh * self.action_bound + self.action_shift
        return actions, log_prob, a_tanh

    def _apply_constraints(self, x: np.ndarray) -> np.ndarray:
        """Apply problem-specific constraints if enabled."""
        return enforce_gear_descending(x, self.search_space) if self.use_constraints else x

    def _log_record(
        self,
        action_vec: np.ndarray,
        sim: Tuple[float, float, float, float],
        reward: float,
        rank: int,
        cd_raw: float,
        cd_tilde: float,
        *,
        t_env_ms: float,
        wall_time_s: float,
        loss_critic: Optional[float] = None,
        loss_actor: Optional[float] = None,
        alpha: Optional[float] = None,
        update_time_ms: Optional[float] = None,
        loss_critic1: Optional[float] = None,
        loss_critic2: Optional[float] = None,
        phase: str = "interaction",
    ) -> None:
        """Append a single interaction/update record to the results buffer."""
        rec: Dict[str, Any] = {
            "algo": self.algo_name,
            "seed": self.seed,
            "evaluation": int(self.eval_count),
            "timestamp": time.time(),
            "wall_time_s": float(wall_time_s),
            "t_env_ms": float(t_env_ms),
            "p1_final_drive_ratio": float(action_vec[0]),
            "p2_roll_radius": float(action_vec[1]),
            "p3_gear3_diff": float(action_vec[2]),
            "p4_gear4_diff": float(action_vec[3]),
            "p5_gear5": float(action_vec[4]),
            "consumption": float(sim[0]),
            "ela3": float(sim[1]),
            "ela4": float(sim[2]),
            "ela5": float(sim[3]),
            "reward": float(reward),
            "rank": int(rank),
            "cd_raw": float(cd_raw),
            "cd_tilde": float(cd_tilde),
            "phase": phase,
        }
        if loss_critic is not None:
            rec["loss_critic"] = float(loss_critic)
        if loss_actor is not None:
            rec["loss_actor"] = float(loss_actor)
        if alpha is not None:
            rec["alpha"] = float(alpha)
        if update_time_ms is not None:
            rec["update_time_ms"] = float(update_time_ms)
        if loss_critic1 is not None:
            rec["loss_critic1"] = float(loss_critic1)
        if loss_critic2 is not None:
            rec["loss_critic2"] = float(loss_critic2)
        self.results_list.append(rec)

    # ---------------------------------------------------------------------
    # Acting, memory, learning
    # ---------------------------------------------------------------------
    def act(self, state: np.ndarray, use_random: bool = False) -> Tuple[np.ndarray, Optional[tf.Tensor]]:
        """Return an action (random during warmup) and its log-prob (if policy-based)."""
        s = np.expand_dims(state, axis=0)
        if use_random:
            a = np.random.uniform(self.action_low, self.action_high, size=self.action_shape)
            return np.expand_dims(a, axis=0), None
        mean, log_std = self.actor.predict(s, verbose=0)
        mean = tf.convert_to_tensor(mean, dtype=tf.float64)
        log_std = tf.convert_to_tensor(log_std, dtype=tf.float64)
        actions, log_prob, _ = self._process_actions(mean, log_std)
        return actions, log_prob

    def remember(self, state: np.ndarray, action: np.ndarray, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.memory.append([state, action, reward, next_state, done])

    def replay(self) -> Optional[Dict[str, float]]:
        """One SAC update step from a random minibatch; returns scalar metrics."""
        if len(self.memory) < self.batch_size:
            return None

        samples = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = [np.vstack(s) for s in zip(*samples)]

        states = tf.convert_to_tensor(states, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float64)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float64)
        dones = tf.convert_to_tensor(dones, dtype=tf.float64)

        update_t0 = time.perf_counter()
        with tf.GradientTape(persistent=True) as tape:
            n_mean, n_log_std = self.actor(next_states)
            next_actions, n_logp, _ = self._process_actions(n_mean, n_log_std)
            q1_next = self.critic_target_1([next_states, next_actions])
            q2_next = self.critic_target_2([next_states, next_actions])
            q_next_min = tf.minimum(q1_next, q2_next)
            target_v = q_next_min - self.alpha * n_logp
            target_q = rewards + self.gamma * (1.0 - dones) * target_v

            q1 = self.critic_1([states, actions])
            q2 = self.critic_2([states, actions])
            critic_loss_1 = tf.reduce_mean(0.5 * tf.square(q1 - target_q))
            critic_loss_2 = tf.reduce_mean(0.5 * tf.square(q2 - target_q))

            a_mean, a_log_std = self.actor(states)
            new_actions, logp, _ = self._process_actions(a_mean, a_log_std)
            q1_pi = self.critic_1([states, new_actions])
            q2_pi = self.critic_2([states, new_actions])
            q_pi_min = tf.minimum(q1_pi, q2_pi)
            actor_loss = tf.reduce_mean(self.alpha * logp - q_pi_min)

            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(logp + self.target_entropy))

        self.critic_optimizer_1.apply_gradients(
            zip(tape.gradient(critic_loss_1, self.critic_1.trainable_variables),
                self.critic_1.trainable_variables)
        )
        self.critic_optimizer_2.apply_gradients(
            zip(tape.gradient(critic_loss_2, self.critic_2.trainable_variables),
                self.critic_2.trainable_variables)
        )
        self.actor_optimizer.apply_gradients(
            zip(tape.gradient(actor_loss, self.actor.trainable_variables),
                self.actor.trainable_variables)
        )
        if self.auto_alpha:
            self.alpha_optimizer.apply_gradients(
                zip(tape.gradient(alpha_loss, [self.log_alpha]), [self.log_alpha])
            )
            self.alpha = tf.exp(self.log_alpha)

        update_time_ms = (time.perf_counter() - update_t0) * 1000.0
        del tape

        loss_c1 = float(critic_loss_1.numpy())
        loss_c2 = float(critic_loss_2.numpy())
        loss_c_mean = 0.5 * (loss_c1 + loss_c2)
        metrics = {
            "loss_critic1": loss_c1,
            "loss_critic2": loss_c2,
            "loss_critic": float(loss_c_mean),
            "loss_actor": float(actor_loss.numpy()),
            "alpha": float(self.alpha.numpy()) if isinstance(self.alpha, tf.Tensor) else float(self.alpha),
            "log_prob_mean": float(tf.reduce_mean(logp).numpy()),
            "q_pi_min_mean": float(tf.reduce_mean(q_pi_min).numpy()),
            "update_time_ms": float(update_time_ms),
        }
        if self.auto_alpha:
            metrics["loss_alpha"] = float(alpha_loss.numpy())
        return metrics

    def run(self, budget: int) -> None:
        """Interact with the environment until `budget` valid evaluations are collected."""
        current_state = np.zeros(self.observation_shape, dtype=np.float64)

        while self.eval_count < budget:
            step_t0 = time.perf_counter()

            use_random = self.eval_count < self.initial_random_steps
            max_resamples = 20
            attempt = 0
            action = None
            while attempt < max_resamples:
                a_tensor, _ = self.act(current_state, use_random=use_random)
                cand = a_tensor[0].numpy() if tf.is_tensor(a_tensor) else a_tensor[0]
                cand = self._apply_constraints(cand)
                key = self._action_key(cand)
                if key not in self.invalid_actions:
                    action = cand
                    break
                attempt += 1
                use_random = True
            if action is None:
                cand = np.random.uniform(self.action_low, self.action_high, size=self.action_shape)
                action = self._apply_constraints(cand)

            t_env0 = time.perf_counter()
            sim_results = self.env.step(action)
            t_env_ms = (time.perf_counter() - t_env0) * 1000.0

            if not self._is_valid_sim(sim_results):
                self.invalid_actions.add(self._action_key(action))
                time.sleep(0.001)
                continue

            self.eval_count += 1

            point = {
                "consumption": float(sim_results[0]),
                "ela3": float(sim_results[1]),
                "ela4": float(sim_results[2]),
                "ela5": float(sim_results[3]),
            }

            reward, rank, cd_raw, cd_tilde = nsga_rank_cd_reward(
                self.archive_all_df, point, self.objectives,
                w_r=self.w_r, w_c=self.w_c, eps=self.nsga_eps
            )
            self.archive_all_df.loc[len(self.archive_all_df)] = point

            next_state = np.array(sim_results[:4], dtype=np.float64)
            done = False

            self.remember(current_state, action, reward, next_state, done)
            train_metrics = self.replay()

            self._update_target_weights(self.critic_1, self.critic_target_1, self.tau)
            self._update_target_weights(self.critic_2, self.critic_target_2, self.tau)
            self._recent_actions.append(action)

            if self.tb:
                step = self.eval_count
                self.tb.scalar("SAC/Reward", float(reward), step=step)
                self.tb.scalar("Obj/consumption", float(sim_results[0]), step=step)
                self.tb.scalar("Obj/ela3", float(sim_results[1]), step=step)
                self.tb.scalar("Obj/ela4", float(sim_results[2]), step=step)
                self.tb.scalar("Obj/ela5", float(sim_results[3]), step=step)
                self.tb.scalar("SAC/Rank", float(rank), step=step)
                self.tb.scalar("SAC/CD_tilde", float(cd_tilde), step=step)
                if train_metrics:
                    self.tb.scalar("SAC/Loss/Critic", train_metrics["loss_critic"], step=step)
                    self.tb.scalar("SAC/Loss/Actor", train_metrics["loss_actor"], step=step)
                    self.tb.scalar("SAC/Alpha", train_metrics["alpha"], step=step)
                    self.tb.scalar("SAC/UpdateTimeMs", train_metrics["update_time_ms"], step=step)
                    self.tb.scalar("SAC/LogProb/BatchMean", train_metrics["log_prob_mean"], step=step)
                    self.tb.scalar("SAC/QpiMin/BatchMean", train_metrics["q_pi_min_mean"], step=step)

            wall_time_s = time.perf_counter() - step_t0

            self._log_record(
                action_vec=action,
                sim=sim_results,
                reward=float(reward),
                rank=int(rank),
                cd_raw=float(cd_raw),
                cd_tilde=float(cd_tilde),
                t_env_ms=float(t_env_ms),
                wall_time_s=float(wall_time_s),
                loss_critic=train_metrics["loss_critic"] if train_metrics else None,
                loss_actor=train_metrics["loss_actor"] if train_metrics else None,
                alpha=train_metrics["alpha"] if train_metrics else None,
                update_time_ms=train_metrics["update_time_ms"] if train_metrics else None,
                loss_critic1=train_metrics["loss_critic1"] if train_metrics else None,
                loss_critic2=train_metrics["loss_critic2"] if train_metrics else None,
                phase="interaction",
            )

            current_state = next_state
