import json
import math
import os
import random as rdm

import numpy as np
from matplotlib import pyplot as plt


class Benchmark:
    """
    A pseudo simulation that can be used to test evolution algorithms.
    """

    def __init__(self, m=1, n=1, complexity=1, seed=None):
        """
        :param m: Number of input parameters.
        :type m: int
        :param n: Number of output parameters.
        :type n: int
        :param complexity: Scaling factor influencing the complexity of the simulation.
        Expected to be in the range [0, 1].
        :type complexity: float or int
        :param seed: Seed for the random number generator.
        :type seed: int or None
        """

        self._m = m
        self._n = n

        # ensure that complexity is in range [0, 1]
        complexity = min(max(complexity, 0), 1)

        # set the seed for the random number generator
        if seed is None:
            seed = rdm.randint(0, 2 ** 32 - 1)
        rdm_original_state = rdm.getstate()
        rdm.seed(seed)

        # set a to a nxm matrix of random values between 0 and 6
        # a is used to scale the frequency of local extrema
        self._a = [[complexity * rdm.random() * 6 for _ in range(m)] for _ in range(n)]

        # set b to a nxm matrix of random values between 0 and .5
        # b is used to weight the global and local extrema
        self._b = [[complexity * rdm.random() / 2 for _ in range(m)] for _ in range(n)]

        # set c to a nxm matrix of random values between -0.5 and 1.5
        # c is used to set the position of the global maximum
        self._c = [[(rdm.random() - 0.5) * 2 for _ in range(m)] for _ in range(n)]

        # set weights to a nxm matrix of random values between 0 and 1
        # weights are used to weight the influence of each input parameter on each output parameter
        self._weights = [[0 for _ in range(m)] for _ in range(n)]  # initialize all weights to 0
        nonzero_weights_per_row = 1 + round((m - 1) * complexity)  # calculate the number of nonzero weights per row
        for j in range(n):
            indexes = [(j, k) for k in range(m)]
            rdm.shuffle(indexes)
            for i in range(nonzero_weights_per_row):
                self._weights[indexes[i][0]][indexes[i][1]] = rdm.random()
        
        # reset the seed for the random number generator
        rdm.setstate(rdm_original_state)

    def export(self, name="simulation", directory="pseudo_simulations"):
        """
        Exports the simulation to a directory.
        :param directory: Directory to save the simulation in.
        :type directory: str
        :param name: Name of the simulation.
        :type name: str
        """
        # create the path to the target directory
        path = os.path.join(directory, name)

        # check if the target directory exists
        if not os.path.exists(path):
            os.makedirs(path)

        # save the simulation to a .json file
        with open(os.path.join(path, 'settings.json'), 'w') as file:
            json.dump(self.__dict__, file)

        # save the visualization of the simulation
        self.plot(directory=os.path.join(path, 'results'))

    @classmethod
    def load(cls, name="simulation", directory="pseudo_simulations"):
        """
        Imports a simulation from a directory which must contain a settings.json file.
        :param name: Name of the simulation.
        :type name: str
        :param directory: Directory to load the simulation from.
        :type directory: str
        """
        # create the path to the source directory
        path = os.path.join(directory, name)

        # check if the source directory exists
        path = os.path.join(path, 'settings.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file {path}")

        # load the settings.json file and create the simulation
        with open(path, 'r') as file:
            data = json.load(file)
        sim = cls()
        sim.__dict__ = data
        return sim

    @staticmethod
    def global_max_wave(x):
        """
        Creates a cosine wave, which is designed to have a maximum value of 1 at x = 0. It's used as the foundational
        waveform for the pseudo-simulation.
        :param x: The input parameter for the wave function.
        :type x: float or int
        :return: The cosine wave value at the input parameter x.
        :rtype: float
        """
        return 1 + math.cos(math.pi * x)

    @staticmethod
    def local_extrema_overlay(a, x):
        """
        Overlays additional local extrema onto the base wave.
        :param a: Scaling factor influencing the frequency of local extrema.
        :type a: float or int
        :param x: The input parameter for the wave function.
        :type x: float or int
        :return: The value of the wave function at the input parameter x.
        :rtype: float
        """
        return 1 + math.cos(a * x * math.pi)

    @classmethod
    def evaluate(cls, a, b, c, x):
        """
        Evaluates the pseudo-simulation with the given parameters using the global_max_wave and local_extrema_overlay.
        :param a: Scaling factor influencing the frequency of local extrema.
        :type a: float or int
        :param b: Weighting factor influencing the weighting of global and local extrema.
        :type b: float or int
        :param c: Offset factor setting the position of the global maximum of the wave.
        :type c: float or int
        :param x: The input parameter for the wave function, expected to be in the range [0, 1].
        :type x: float or int
        :return: The value of the pseudo-simulation at the input parameter x.
        :rtype: float
        """
        return ((1 - b) * cls.global_max_wave(x - c) + b * cls.local_extrema_overlay(a, x - c)) / 2

    def plot(self, directory="results", resolution=100):
        """
        Visualizes the simulation for each output dimension using 2D and 3D results and saves them in the given directory.
        :param resolution: Number of points along each axis in the results.
        :type resolution: int
        :param directory: Directory to save the results in.
        :type directory: str
        """

        print(f"Plotting simulation... ")

        # check if the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        for output_dim in range(self._n):
            plot_number = 1
            for i in range(self._m):
                for j in range(self._m):
                    if i <= j:  # Avoid duplicate results
                        # Create a new figure
                        fig = plt.figure(figsize=(12, 12))

                        if i == j:  # Plot 2D for single parameters
                            ax = fig.add_subplot(111)
                            plot_number += 1

                            x = np.linspace(0, 1, resolution)
                            y = np.array([self.__call__(
                                *[xk if k == i else 0.5 for k in range(self._m)],
                                i=output_dim
                            ) for xk in x])

                            ax.plot(x, y)
                            ax.set_xlabel(f'Parameter {i + 1}')
                            ax.set_ylabel(f'Output {output_dim + 1}')
                            filename = os.path.join(directory, f'single-output_{output_dim + 1}-param_{i + 1}.png')

                        else:  # Plot 3D for pairs of parameters
                            ax = fig.add_subplot(111, projection='3d')
                            plot_number += 1

                            x = np.linspace(0, 1, resolution)
                            y = np.linspace(0, 1, resolution)
                            X, Y = np.meshgrid(x, y)
                            Z = np.array([self.__call__(
                                *[xi if k == i else yi if k == j else 0.5 for k in range(self._m)],
                                i=output_dim
                            ) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
                            Z = Z.reshape(X.shape)

                            ax.plot_surface(X, Y, Z, cmap='viridis')
                            ax.set_xlabel(f'Parameter {i + 1}')
                            ax.set_ylabel(f'Parameter {j + 1}')
                            ax.set_zlabel(f'Output {output_dim + 1}')
                            filename = os.path.join(directory, f'pair-output_{output_dim + 1}-param_{i + 1}_{j + 1}.png')

                        # Save the figure
                        plt.tight_layout()
                        fig.savefig(filename)
                        plt.close(fig)  # Close the figure to avoid memory leaks
        print(f"Done! Saved results to {directory}")

    def __call__(self, *x, i=None):
        """
        Performs a pseudo simulation with the given parameters.
        :param x: Input parameters as floats.
        :type x: float
        :param i: Index of the output parameter to return.
        :type i: int or None
        :return: The output of the simulation.
        :rtype: float or tuple
        """
        # check if the number of arguments is correct
        if len(x) != self._m:
            raise TypeError(f"Expected {self._m} arguments, got {len(x)}")

        # check if all arguments are numbers
        for arg in x:
            if not (isinstance(arg, float) or isinstance(arg, int)):
                raise TypeError(f"Expected float or int, got {type(arg)}")

        # warn if any arguments are out of range [0, 1]
        for arg in x:
            if not 0 <= arg <= 1:
                print(f"Warning: argument {arg} is out of range [0, 1]")

        # calculate the result
        result = [0] * self._n
        for j in range(self._n):
            for k in range(self._m):
                result[j] += self._weights[j][k] * self.evaluate(self._a[j][k], self._b[j][k], self._c[j][k], x[k])

        # return the result
        if i is not None:
            return result[i]
        if self._n == 1:
            return result[0]
        else:
            return tuple(result)
