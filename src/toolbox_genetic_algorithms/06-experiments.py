"""
This script contains the experiments that are used for the analysis and evaluation of the evolutionary algorithms in
chapter 5 of the thesis.
"""
import time

import evolution as evo
from car import Car
from utilities import export_generations_to_csv

if __name__ == '__main__':
    # create a pseudo class
    PseudoCar = Car.pseudo_class(seed=42, complexity=.5)

    # determine the variables
    seeds = [42, 1234, 2024]
    gen_functions = [evo.GEN_HALTON, evo.GEN_GRID_HALTON]
    eval_functions = [evo.EVAL_FITNESS, evo.EVAL_PARETO]
    recombination_functions = [evo.REC_CROSS_ARITHMETIC, evo.REC_CROSS_POINT]
    classes = [PseudoCar, Car]
    dimensions = [(10, 10), (50, 20)]  # (population size, number of generations)

    current = 0
    total = (len(seeds) * len(classes) * len(gen_functions) * len(eval_functions) * len(recombination_functions) *
             len(dimensions))

    for seed in seeds:
        for gen_funct in gen_functions:
            for eval_funct in eval_functions:
                for rec_funct in recombination_functions:
                    for cls in classes:
                        for n_pop, n_gen in dimensions:
                            # generate a key for the experiment name
                            key = (f"{cls.__name__}-{gen_funct.__name__}-{eval_funct.__name__}-{rec_funct.__name__}-"
                                   f"{n_pop}-{n_gen}-{seed}")

                            # print the progress
                            current += 1
                            print(f"({current}/{total}): {key}")


                            # create the strategy
                            def STRAT_CUSTOM(population, i, n):
                                aep = (i + 1) / (n + 1)
                                return population.next_generation(
                                    aep=aep,
                                    eval_funct=eval_funct,
                                    recombination_funct=rec_funct,
                                )


                            # create the initial population
                            initial_population = evo.Population(
                                cls,
                                size=n_pop,
                                gen_funct=gen_funct,
                                seed=seed
                            )

                            # evolve the initial population
                            generations = initial_population.evolve(n_gen, STRAT_CUSTOM)

                            # export the generations to csv
                            export_generations_to_csv(
                                generations,
                                name=key,
                                directory="results/06-experiments"
                            )

                            # clear all instances of the class to prevent interference with the next experiment
                            cls.reset()

                            # sleep a second to keep the console output readable
                            time.sleep(1)

                            print()  # print an empty line for better readability

    # inform the user that the script is done
    print("Done!")
