"""
In this script, we compare the different generation functions for the car problem to deside which one to use.
"""
import evolution as evo
from car import Car

if __name__ == '__main__':
    seed = 42
    pop_size = 50

    uniform_pop = evo.Population(Car, pop_size, gen_funct=evo.GEN_UNIFORM, seed=seed)
    uniform_pop.plot(name="uniform", directory="results/03-comparing-generation-functions-car")
    print(f"GEN_UNIFORM diversity: {uniform_pop.get_diversity()}")

    halton_pop = evo.Population(Car, pop_size, gen_funct=evo.GEN_HALTON, seed=seed)
    halton_pop.plot(name="halton", directory="results/03-comparing-generation-functions-car")
    print(f"GEN_HALTON diversity: {halton_pop.get_diversity()}")

    sobol_halton_pop = evo.Population(Car, pop_size, gen_funct=evo.GEN_SOBOL_HALTON, seed=seed)
    sobol_halton_pop.plot(name="sobol_halton", directory="results/03-comparing-generation-functions-car")
    print(f"GEN_SOBOL_HALTON diversity: {sobol_halton_pop.get_diversity()}")

    grid_halton_pop = evo.Population(Car, pop_size, gen_funct=evo.GEN_GRID_HALTON, seed=seed)
    grid_halton_pop.plot(name="grid_halton", directory="results/03-comparing-generation-functions-car")
    print(f"GEN_GRID_HALTON diversity: {grid_halton_pop.get_diversity()}")