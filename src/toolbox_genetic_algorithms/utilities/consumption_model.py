"""
This module contains the functions to run the consumption simulation.
"""
import os
import subprocess
import platform

cache = {}


def is_wine_installed():
    """
    :return: True if wine64 is installed, False otherwise.
    :rtype: bool
    """

    # check if the result is already cached
    if 'is_wine_installed' in cache:
        return cache['is_wine_installed']

    try:
        # try to run wine64 --version
        subprocess.run(["wine64", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        result = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # catch the error if wine64 is not installed and return False
        result = False

    # cache and return the result
    cache['is_wine_installed'] = result
    return result


def run_simulation(final_drive_ratio, roll_radius, gear3, gear4, gear5):
    """
    Runs the simulation with the given parameters and returns the results.
    :param final_drive_ratio: The final drive ratio.
    :type final_drive_ratio: float or int
    :param roll_radius: The roll radius.
    :type roll_radius: float or int
    :param gear3: The gear ratio of gear 3.
    :type gear3: float or int
    :param gear4: The gear ratio of gear 4.
    :type gear4: float or int
    :param gear5: The gear ratio of gear 5.
    :type gear5: float or int
    :return: consumption, elasticity 3, elasticity 4, elasticity 5
    :rtype: tuple(float, float, float, float)
    """

    # initialize the command
    command = []

    # check the operating system and modify the command accordingly
    os_name = platform.system()
    if os_name == "Windows":
        pass  # nothing to do here
    elif os_name == "Darwin":  # Mac OS
        # try to run using wine64
        if is_wine_installed():
            command.append("wine64")
        else:
            raise Exception("Wine64 is not installed. Please install it using homebrew: brew install wine")
    else:
        raise NotImplementedError(f"Unsupported operating system: {os_name}")

    # create the command to run the simulation
    path_to_simulation = os.path.join(os.path.dirname(__file__), "ConsumptionCar.exe")
    command += [path_to_simulation, str(final_drive_ratio), str(roll_radius), str(gear3), str(gear4), str(gear5)]

    # try to run the simulation
    try:
        completed_process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                           check=True)
    except subprocess.CalledProcessError:
        raise Exception("The simulation could not be run. Please make sure that the simulation is located in "
                        "utilities/ConsumptionCar.exe and that it is executable.")

    # try to convert the values to floats
    try:
        simulation_output = completed_process.stdout.split("\n")[-2].split("  ")
        consumption, _, _, ela_3, ela_4, ela_5 = simulation_output
        consumption, ela_3, ela_4, ela_5 = float(consumption), float(ela_3), float(ela_4), float(ela_5)
    except ValueError:
        raise Exception("The simulation output could not be converted to floats.")

    # return the results
    return consumption, ela_3, ela_4, ela_5
