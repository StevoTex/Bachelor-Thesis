# File: /your_project/src/environment/car_env.py

import os
import subprocess
import platform
from typing import List, Tuple, Union
import numpy as np
from flatbuffers.flexbuffers import String


class CarEnv:
    """
    An environment class for the car simulation model.

    This class calls to the .exe file and provides a
    'step' method in order to run the simulation.
    """

    def __init__(self, exe_file_name: String):
        # NOTE: The .exe file must be in the /executables/ folder (hardcoded)!
        self.exe_file_name = exe_file_name
        self.simulation_path = self._get_simulation_path()
        self.command_prefix = self._get_command_prefix()

    def _get_simulation_path(self) -> str:
        """Determines the absolute path to the simulation EXE."""
        # Navigate two levels up from the 'src/environment' folder to the project root.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "executables", str(self.exe_file_name))

    def _get_command_prefix(self) -> List[str]:
        """Determines the command prefix based on the operating system (e.g., 'wine64')."""
        os_name = platform.system()
        if os_name == "Windows":
            return []
        elif os_name == "Darwin" or os_name == "Linux":  # For Mac & Linux
            if self._is_wine_installed():
                return ["wine"]
            else:
                raise RuntimeError("Wine64 is required to run on macOS/Linux but was not found.")
        else:
            raise NotImplementedError(f"Unsupported operating system: {os_name}")

    def _is_wine_installed(self) -> bool:
        """Checks if 'wine64' is available in the system's PATH."""
        try:
            subprocess.run(["wine", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def step(self, parameters: Union[List[float], np.ndarray]) -> Tuple[float, float, float, float]:
        """
        Executes a simulation step with the given parameters.

        Args:
            parameters: A list or NumPy array with the 5 input parameters.
                        Order: [final_drive_ratio, roll_radius, gear3, gear4, gear5]

        Returns:
            A tuple with the 4 objective values (consumption, elasticity 3, elasticity 4, elasticity 5).
        """
        # Convert all parameters to strings for the command line.
        str_params = [str(p) for p in parameters]

        # Assemble the complete command for execution.
        command = self.command_prefix + [self.simulation_path] + str_params

        try:
            completed_process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        except FileNotFoundError:
            raise RuntimeError(f"Simulation executable not found at: {self.simulation_path}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"The simulation could not be executed. Error: {e.stderr}")

        try:
            simulation_output = completed_process.stdout.strip().split("\n")[-1].split("  ")
            # Filter out empty strings that might result from multiple spaces.
            simulation_output = [item for item in simulation_output if item]
            consumption, _, _, ela_3, ela_4, ela_5 = simulation_output

            return float(consumption), float(ela_3), float(ela_4), float(ela_5)

        except (ValueError, IndexError) as e:
            raise ValueError(
                f"The simulation output could not be processed. Output: '{completed_process.stdout}'. Error: {e}")


# Example of usage to test.
if __name__ == '__main__':
    env = CarEnv("ConsumptionCar.exe")

    test_params = [3.0, 0.3, 1.5, 1.0, 0.8]

    print(f"Starting simulation with parameters: {test_params}")
    results = env.step(test_params)
    print(f"Received results (Consumption, E3, E4, E5): {results}")