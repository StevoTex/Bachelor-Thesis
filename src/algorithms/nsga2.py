"""NSGA-II for vehicle design (delta re-parametrization).

Evaluates candidates via an external simulator. All objectives are minimized.
Negative objective values are allowed; only non-finite results are discarded.
Invalid actions are cached to avoid re-use. Rank and crowding distance are logged.
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Tuple, Set
import numpy as np

from src.utils.constraints import enforce_gear_descending
from src.utils.objectives import rl_reward, is_terminal
from ..toolbox_genetic_algorithms.evolution import EvolutionStrategy, Allele, Individual, Population


class FinalDrive(Allele):
    """Allele for final-drive ratio (min/max)."""
    _min = 3.0
    _max = 5.5


class RollRadius(Allele):
    """Allele for roll radius (min/max)."""
    _min = 0.2
    _max = 0.4


class Gear5(Allele):
    """Allele for 5th gear ratio (min/max)."""
    _min = 0.5
    _max = 2.25


class Delta54(Allele):
    """Allele for delta(5→4) (min/max)."""
    _min = 0.1
    _max = 0.5


class Delta43(Allele):
    """Allele for delta(4→3) (min/max)."""
    _min = 0.1
    _max = 0.5


class Car(Individual):
    """Individual blueprint using delta re-parametrization."""
    _Blueprint = {
        "genotype": [FinalDrive, RollRadius, Gear5, Delta54, Delta43],
        "genotype_labels": ["Final Drive", "Roll Radius", "Gear 5", "Delta 5→4", "Delta 4→3"],
        "goals": ["minimize", "minimize", "minimize", "minimize"],
        "phenotype_labels": ["Consumption", "Elasticity 3", "Elasticity 4", "Elasticity 5"],
    }

    def _calculate_phenotype(self):
        """Placeholder; evaluation is handled externally via `eval_function`."""
        return tuple([0.0] * len(self._Blueprint["goals"]))

    def _enforce_constraints(self):
        """No genotype-side constraints; mapping is handled in `eval_function`."""
        return


class Nsga2(EvolutionStrategy):
    """Core NSGA-II strategy."""

    def __init__(self, population: Population, eval_function, on_update_metrics=None, **kwargs: dict):
        """Initialize with population and evaluation callback.

        Args:
            population: Initial population.
            eval_function: Callable (individual, genotype_values) -> phenotype tuple.
            on_update_metrics: Optional callback receiving individuals after rank/crowding update.
        """
        super().__init__(population, **kwargs)
        self.eval_function = eval_function
        self._on_update_metrics = on_update_metrics
        if not self.eval_function:
            raise ValueError("NSGA-II Strategy requires an 'eval_function'.")

        self.eta_crossover = kwargs.get("eta_crossover", 15)
        self.eta_mutation = kwargs.get("eta_mutation", 20)
        self.pmut = kwargs.get("pmut", 0.1)
        self.crossover_prob = kwargs.get("crossover_prob", 0.9)

        blueprint = self.population._IndividualClass._Blueprint["genotype"]
        self.bounds_low = np.array([allele_type._min for allele_type in blueprint], dtype=float)
        self.bounds_high = np.array([allele_type._max for allele_type in blueprint], dtype=float)

        for ind in self.population:
            if not getattr(ind, "is_evaluated", False):
                ind.phenotype = self.eval_function(ind, ind.get_genotype(transform=True))
                ind.is_evaluated = True

        initial_fronts = self._fast_non_dominated_sort(list(self.population))
        for front in initial_fronts:
            self._calculate_crowding_distance(front)

        self._notify_update(list(self.population))

    def _notify_update(self, individuals: List[Individual]):
        """Invoke the metrics callback if provided."""
        if callable(self._on_update_metrics):
            self._on_update_metrics(individuals)

    def _selection(self) -> Population:
        """Binary tournament selection on (rank, crowding)."""
        selected_parents_list = []
        population_list = list(self.population)
        for _ in range(self.population_size):
            p1, p2 = np.random.choice(population_list, size=2, replace=True)
            if (p1.rank < p2.rank) or (p1.rank == p2.rank and p1.crowding_distance > p2.crowding_distance):
                selected_parents_list.append(p1)
            else:
                selected_parents_list.append(p2)
        return Population(individuals=selected_parents_list, individual_class=self.population._IndividualClass)

    def _recombination(self, parents: Population) -> Population:
        """SBX recombination."""
        offspring_list = []
        parent_list = list(parents)
        genotype_blueprint = self.population._IndividualClass._Blueprint["genotype"]

        for i in range(0, self.population_size, 2):
            parent1 = parent_list[i]
            parent2 = parent_list[i + 1] if i + 1 < len(parent_list) else parent_list[0]

            geno1_values = list(parent1.get_genotype(transform=True))
            geno2_values = list(parent2.get_genotype(transform=True))
            offspring1_geno_values = list(geno1_values)
            offspring2_geno_values = list(geno2_values)

            if np.random.rand() < self.crossover_prob:
                for j in range(len(geno1_values)):
                    y1, y2 = geno1_values[j], geno2_values[j]
                    yl, yu = self.bounds_low[j], self.bounds_high[j]
                    if abs(y1 - y2) < 1e-14:
                        c1 = y1
                        c2 = y2
                    else:
                        if y1 > y2:
                            y1, y2 = y2, y1
                        rand = np.random.rand()
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                        alpha = 2.0 - beta ** -(self.eta_crossover + 1)
                        if rand <= 1.0 / alpha:
                            betaq = (rand * alpha) ** (1.0 / (self.eta_crossover + 1))
                        else:
                            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_crossover + 1))
                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                        rand = np.random.rand()
                        beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                        alpha = 2.0 - beta ** -(self.eta_crossover + 1)
                        if rand <= 1.0 / alpha:
                            betaq = (rand * alpha) ** (1.0 / (self.eta_crossover + 1))
                        else:
                            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_crossover + 1))
                        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                        c1 = np.clip(c1, yl, yu)
                        c2 = np.clip(c2, yl, yu)

                    offspring1_geno_values[j] = c1
                    offspring2_geno_values[j] = c2

            offspring1 = self.population._IndividualClass(*[genotype_blueprint[k](v) for k, v in enumerate(offspring1_geno_values)])
            offspring2 = self.population._IndividualClass(*[genotype_blueprint[k](v) for k, v in enumerate(offspring2_geno_values)])

            offspring_list.extend([offspring1, offspring2])

        return Population(individuals=offspring_list, individual_class=self.population._IndividualClass)

    def _mutation(self, offspring: Population) -> Population:
        """Polynomial mutation."""
        mutated_offspring_list = []
        genotype_blueprint = self.population._IndividualClass._Blueprint["genotype"]

        for individual in offspring:
            mutated_genotype_values = list(individual.get_genotype(transform=True))
            for i in range(len(mutated_genotype_values)):
                if np.random.random() <= self.pmut:
                    y = mutated_genotype_values[i]
                    yl, yu = self.bounds_low[i], self.bounds_high[i]
                    if yl == yu:
                        continue
                    delta1, delta2 = (y - yl) / (yu - yl), (yu - y) / (yu - yl)
                    rand = np.random.random()
                    mut_pow = 1.0 / (self.eta_mutation + 1.0)
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_mutation + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_mutation + 1.0))
                        deltaq = 1.0 - val ** mut_pow
                    y = y + deltaq * (yu - yl)
                    y = np.clip(y, yl, yu)
                    mutated_genotype_values[i] = y

            mutated_individual = individual.__class__(
                *[genotype_blueprint[j](val) for j, val in enumerate(mutated_genotype_values)]
            )
            mutated_offspring_list.append(mutated_individual)

        return Population(individuals=mutated_offspring_list, individual_class=self.population._IndividualClass)

    def _create_new_generation(self, offspring: Population) -> Population:
        """Environmental selection to form the next generation."""
        combined_population = list(self.population) + list(offspring)

        for ind in combined_population:
            if not getattr(ind, "is_evaluated", False) or ind.phenotype is None:
                ind.phenotype = self.eval_function(ind, ind.get_genotype(transform=True))
                ind.is_evaluated = True

        fronts = self._fast_non_dominated_sort(combined_population)
        next_generation_individuals = []
        current_front_index = 0

        while current_front_index < len(fronts) and len(next_generation_individuals) + len(fronts[current_front_index]) <= self.population_size:
            self._calculate_crowding_distance(fronts[current_front_index])
            next_generation_individuals.extend(fronts[current_front_index])
            current_front_index += 1

        if current_front_index < len(fronts):
            remaining_front = fronts[current_front_index]
            self._calculate_crowding_distance(remaining_front)
            remaining_front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            num_to_add = self.population_size - len(next_generation_individuals)
            next_generation_individuals.extend(remaining_front[:num_to_add])

        self._notify_update(next_generation_individuals)
        return Population(individuals=next_generation_individuals, individual_class=self.population._IndividualClass)

    def _is_dominated(self, p_phenotype, q_phenotype):
        """Return True if p is dominated by q (minimization)."""
        return all(p_val >= q_val for p_val, q_val in zip(p_phenotype, q_phenotype)) and any(p_val > q_val for p_val, q_val in zip(p_phenotype, q_phenotype))

    def _fast_non_dominated_sort(self, population_list: List[Individual]) -> List[List[Individual]]:
        """Efficient non-dominated sorting producing Pareto fronts."""
        fronts = [[]]
        for p in population_list:
            p.domination_count = 0
            p.dominated_solutions = []
            if p.phenotype is None:
                continue
            p_phen = list(p.phenotype)
            for q in population_list:
                if p == q or q.phenotype is None:
                    continue
                q_phen = list(q.phenotype)
                if self._is_dominated(p_phen, q_phen):
                    p.domination_count += 1
                elif self._is_dominated(q_phen, p_phen):
                    p.dominated_solutions.append(q)
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if not next_front:
                break
            fronts.append(next_front)
        return fronts

    def _calculate_crowding_distance(self, front: List[Individual]):
        """Compute crowding distance within a front."""
        if not front:
            return
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        num_objectives = len(front[0].phenotype)
        for ind in front:
            ind.crowding_distance = 0.0

        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.phenotype[m])
            f_min, f_max = front[0].phenotype[m], front[-1].phenotype[m]
            if f_max == f_min:
                continue
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            for i in range(1, len(front) - 1):
                distance = front[i + 1].phenotype[m] - front[i - 1].phenotype[m]
                front[i].crowding_distance += distance / (f_max - f_min)


class Nsga2Algorithm:
    """High-level NSGA-II wrapper with unified reward/validity handling."""

    def __init__(self, env, search_space, **kwargs):
        self.env = env
        self.search_space = search_space
        self.results_list: List[Dict[str, Any]] = []

        self.seed = int(kwargs.get("seed", 0))
        self.algo_name = "GA"

        self.pop_size = int(kwargs.get("pop_size", 40))
        self.generations_hint = int(kwargs.get("generations", 0))
        self.use_constraints = bool(kwargs.get("use_constraints", True))

        self.rl_reward_cfg: Dict[str, Any] = kwargs.get("rl_reward", {"type": "heuristic"})

        self.eval_count: int = 0
        self._generation: int = 0

        self.invalid_actions: Set[Tuple[float, ...]] = set()

        blueprint = Car._Blueprint["genotype"]
        self.bounds_low = np.array([allele_type._min for allele_type in blueprint], dtype=float)
        self.bounds_high = np.array([allele_type._max for allele_type in blueprint], dtype=float)

        def _is_valid_sim(sim: Tuple[float, float, float, float]) -> bool:
            """Return True iff all objective values are finite; negatives are allowed."""
            arr = np.asarray(sim, dtype=np.float64)
            return np.isfinite(arr).all()

        def _action_key(x_env: np.ndarray, decimals: int = 6) -> Tuple[float, ...]:
            """Stable key for caching actions with limited precision."""
            return tuple(np.round(np.asarray(x_env, dtype=np.float64), decimals=decimals).tolist())

        def _resample_genotype(max_tries: int = 1000) -> np.ndarray:
            """Uniform resampling within bounds that avoids known invalid actions."""
            for _ in range(max_tries):
                x = np.random.uniform(self.bounds_low, self.bounds_high).astype(np.float64)
                xx = enforce_gear_descending(x, self.search_space) if self.use_constraints else x
                if _action_key(xx) not in self.invalid_actions:
                    return x
            return np.random.uniform(self.bounds_low, self.bounds_high).astype(np.float64)

        def eval_fn(ind: Individual, genotype_values: List[float]):
            """Evaluation with strict invalid filtering and unified reward logging."""
            attempts = 0
            max_attempts = 1000

            while True:
                if attempts == 0:
                    x = np.array(genotype_values, dtype=np.float64)
                else:
                    x = _resample_genotype()

                x_env = enforce_gear_descending(x, self.search_space) if self.use_constraints else x
                key = _action_key(x_env)

                if key in self.invalid_actions:
                    attempts += 1
                    if attempts >= max_attempts:
                        attempts = 0
                    continue

                t0 = time.perf_counter()
                consumption, e3, e4, e5 = self.env.step(x_env)
                t_env_ms = (time.perf_counter() - t0) * 1000.0

                if not _is_valid_sim((consumption, e3, e4, e5)):
                    self.invalid_actions.add(key)
                    attempts += 1
                    if attempts >= max_attempts:
                        attempts = 0
                    continue

                rew = rl_reward((consumption, e3, e4, e5), self.rl_reward_cfg)
                valid_flag = bool(is_terminal((consumption, e3, e4, e5), self.rl_reward_cfg))

                self.eval_count += 1
                log_idx = len(self.results_list)
                setattr(ind, "_last_log_idx", log_idx)

                self.results_list.append({
                    "algo": self.algo_name,
                    "seed": self.seed,
                    "evaluation": int(self.eval_count),
                    "timestamp": time.time(),
                    "t_env_ms": float(t_env_ms),
                    "p1_final_drive_ratio": float(x_env[0]),
                    "p2_roll_radius": float(x_env[1]),
                    "p3_gear3_diff": float(x_env[2]),
                    "p4_gear4_diff": float(x_env[3]),
                    "p5_gear5": float(x_env[4]),
                    "consumption": float(consumption),
                    "ela3": float(e3),
                    "ela4": float(e4),
                    "ela5": float(e5),
                    "reward": float(rew),
                    "phase": "init" if self._generation == 0 else "generation",
                    "generation": int(self._generation),
                    "valid": valid_flag,
                    "rank": None,
                    "crowding_distance": None,
                })

                return (consumption, e3, e4, e5)

        def on_update_metrics(individuals: List[Individual]):
            """Write rank and crowding distance back into the latest logs."""
            for ind in individuals:
                idx = getattr(ind, "_last_log_idx", None)
                if idx is None:
                    continue
                if 0 <= idx < len(self.results_list):
                    self.results_list[idx]["rank"] = int(getattr(ind, "rank", -1)) if hasattr(ind, "rank") else None
                    cd = getattr(ind, "crowding_distance", None)
                    self.results_list[idx]["crowding_distance"] = float(cd) if cd is not None else None

        self.population = Population(size=self.pop_size, individual_class=Car)

        passthrough = {
            k: v for k, v in kwargs.items()
            if k not in {"pop_size", "generations", "seed", "use_constraints", "rl_reward"}
        }

        self._generation = 0
        self.core = Nsga2(
            population=self.population,
            eval_function=eval_fn,
            on_update_metrics=on_update_metrics,
            **passthrough,
        )

    def run(self, budget: int):
        """Run evolution for as many full generations as fit into the budget."""
        if budget <= 0:
            return
        already = len(self.results_list)
        remaining = max(0, budget - already)
        generations = remaining // self.pop_size
        if self.generations_hint > 0:
            generations = min(generations, self.generations_hint)

        for g in range(generations):
            self._generation = g + 1
            self.core.evolve(generations=1)
