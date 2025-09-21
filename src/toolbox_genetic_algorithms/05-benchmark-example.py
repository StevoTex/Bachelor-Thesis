"""
In this example, we will create a pseudo class as benchmark and evolve it for 10 generations.
"""
import evolution as evo
from car import Car
from utilities import plot_generations

if __name__ == '__main__':
    # create a pseudo class as benchmark
    PseudoCar = Car.pseudo_class(seed=42, complexity=.5)

    # create the initial population of pseudo cars
    initial_population = evo.Population(PseudoCar, 50, seed=42)

    # evolve the initial population for 10 generations using the default strategy
    generations = initial_population.evolve(20)

    # plot the generations
    plot_generations(generations, name="generations", directory="results/05-benchmark-example")

    # plot the entire simulation
    PseudoCar.get_sim().plot("results/05-benchmark-example/simulation")