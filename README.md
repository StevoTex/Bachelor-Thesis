# Comparison of ML Algorithms for the Identification of Pareto Fronts in High-Dimensional Design Spaces

## Overview

End-to-end pipeline for discovering and evaluating Pareto fronts in a car design optimization task using four algorithmic approaches:

## Algorithms (Core Behaviour)

### NSGA-II (`src/algorithms/nsga2.py`)
- Uses SBX crossover and polynomial mutation.
- Environmental selection based on non-dominated sorting and crowding distance.
- Evaluation applies a unified validity rule and logs rank/crowding information.

### SAC (`src/algorithms/soft_actor_critic.py`)
- Double critics and target networks with a stochastic squashed Gaussian policy.
- Temperature parameter \( \alpha \) is learned during training.
- Reward is based on NSGA rank and normalized crowding distance against a growing archive of valid solutions.

### A3C (`src/algorithms/a3c.py`)
- Multi-threaded rollouts with a global optimizer and lock-protected updates.
- Same NSGA-style reward as SAC.
- Invalid actions are avoided via resampling.

### Active Learning (`src/algorithms/active_learning.py`)
- Fixed Cartesian candidate grid from per-dimension step sizes.
- Small neural network committee (with input normalization) predicts rewards.
- Batch split into exploit (top mean prediction) and explore (random).
- Labels use NSGA rank + crowding reward; only valid simulator outputs are used.

## Environment

- Simulation model provided as an executable in `executables/`.
- Environment interface: `src/environment/car_env.py`
  - Exposes a `step()` function that takes a **decision vector** and returns a **target vector**.

## Setup & Installation

1. Place `ConsumptionCar.exe` in the project root (or adjust the path in `car_env.py`).
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. *(macOS/Linux only)* Install `wine` to run the Windows simulation binary:
    ```bash
    brew install --cask wine-stable
    ```

## Running an Experiment

The workflow consists of three stages:

### 1. Hyperparameter Tuning

- **Purpose**: Optimize algorithm-specific hyperparameters.
- **Configuration**: `configs/hyperparams.json`
  - Define `seeds_per_trial`, the seed list, search space, and Optuna sampler (e.g. `"TPE"`).
- **Usage**:
  - Run the main function in `src/tune.py`:
    ```python
    if __name__ == '__main__':
        main(["SAC", "A3C"])  # or ["AL"], ["GA"], etc.
    ```
  - You can pass a list of algorithms:  
    **`["SAC"]`**, **`["A3C"]`**, **`["AL"]`**, **`["GA"]`**  
    If no list is passed, **all four algorithms** will be tuned.
- **Results**: Saved in `results/tune/`.

### 2. Run Experiment

- **Purpose**: Execute the actual optimization runs.
- **Configuration**:
  - Global config: `configs/common_config.json` (seeds, eval budget, bounds, HV mode).
  - Algorithm-specific config: `configs/algorithms/`.
  - For Active Learning: configure candidate pool.
- **Usage**:
  - Run the main function in `run_experiment.py`:
    ```python
    if __name__ == '__main__':
        main(["AL", "GA"])
    ```
  - Possible values:  
    **`["SAC"]`**, **`["A3C"]`**, **`["AL"]`**, **`["GA"]`**  
    If no list is passed, **all four algorithms** will be executed.
- **Results**: Stored in `results/default_experiment/`.

### 3. Benchmarking & Analysis

- **Purpose**: Compare all algorithms quantitatively.
- **Usage**:
  - Run the main function in `benchmark.py`:
    ```python
    if __name__ == '__main__':
        main()
    ```
- **Visualizations**:
  - HV, spacing, and variance as boxplots & bar charts
  - Pareto fronts per run
  - HV curves per seed
  - Mean runtime comparison
- **Metrics**:
  - Summary stats (mean, variance, std)
  - Friedman ranking
- **Results**: Saved in `results/analysis/`.

