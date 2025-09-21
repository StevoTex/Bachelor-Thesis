# Comparison of ML Algorithms for the Identification of Pareto Fronts in High-Dimensional Design Spaces.

This project provides a framework for the systematic comparison of different optimization algorithms for the multi-objective parameter optimization of a simulated vehicle. A modular framework was developed to enable analysis of Genetic Algorithms, Reinforcement Learning, and Active Learning.

## Key Features

- **Modular Framework:** A central script (`run_experiment.py`) controls all experiments, allowing for easy swapping of algorithms.
- **Unified Simulation Interface:** A `car_env` class encapsulates the interaction with the external `.exe` simulation file, providing an interface.
- **JSON-based Configuration:** All experiments are controlled via `.json` files. Each algorithms has its own `.json` file. Shared parameters are centralized in a `common_config.json` to avoid redundancy.
- **Implemented Algorithms:**
    - **Genetic Algorithm:** NSGA-II (in progress)
    - **Reinforcement Learning:** Soft Actor-Critic (SAC) and A3C (the latter is in progress)
    - **Active Learning** 
- **Integrated Benchmarking:** A script for evaluation and comparison (in progress)

## Setup & Installation

1. Ensure the `ConsumptionCar.exe` file is in the project's root directory (or adjust the path in `car_env.py`).
2. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
6.  (macOS/Linux Only) Install `wine`:
    ```bash
    brew install wine-stable 
    ```
    
## Running an Experiment

- All experiments are started via the `run_experiment.py` script. To control which algorithm runs, you only need to adapt a single line at the end of the file.

    ```python
    if __name__ == '__main__':
        # Change "SAC" to "GA", "AL", or "A3C" to start a different experiment.
        main("SAC")
    ```

- The system uses a central `configs/common_config.json` to define parameters that are shared across all algorithms.
- Specific hyperparameters for an algorithm are defined in the corresponding file (`configs/sac_config.json`, etc.).