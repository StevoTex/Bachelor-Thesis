"""
In this script, we will plot the results of the experiments conducted in 06-experiments.py. We will import the
generations from the csv files and plot them using the plot_generations function from utilities.

All plots will be comparable, as fitness values are collected relatively across all experiments and lie on a scale from
0 to 1.
"""

import evolution as evo
from car import Car
from utilities import import_generations_from_csv, plot_generations


if __name__ == '__main__':
    # define where to store the results
    directory = "results/07-plot-experiments"

    # recreate the pseudo class
    PseudoCar = Car.pseudo_class(seed=42, complexity=.5)

    # determine the variables
    seeds = [42, 1234, 2024]
    gen_functions = [evo.GEN_HALTON, evo.GEN_GRID_HALTON]
    eval_functions = [evo.EVAL_FITNESS, evo.EVAL_PARETO]
    recombination_functions = [evo.REC_CROSS_ARITHMETIC, evo.REC_CROSS_POINT]
    classes = [PseudoCar, Car]
    dimensions = [(10, 10), (50, 20)]  # (population size, number of generations)

    # import the generations evolved in 06-experiments.py
    print("Importing the generations ...")
    imports = {}
    for seed in seeds:
        for gen_funct in gen_functions:
            for eval_funct in eval_functions:
                for rec_funct in recombination_functions:
                    for cls in classes:
                        for n_pop, n_gen in dimensions:# generate a key for the experiment name
                            key = (f"{cls.__name__}-{gen_funct.__name__}-{eval_funct.__name__}-{rec_funct.__name__}-"
                                   f"{n_pop}-{n_gen}-{seed}")

                            # skip the halton grid experiment for cars
                            if cls.__name__ == "Car" and gen_funct.__name__ == "GEN_GRID_HALTON":
                                print(f"Skipped '{key}' because inconsistencies in the simulation are caused by edge "
                                      f"cases through GEN_GRID.")
                                continue

                            # import the generations
                            generations = import_generations_from_csv(
                                cls,
                                name=f"{key}.csv",
                                directory="results/06-experiments"
                            )

                            # store the generations
                            imports[key] = generations

    # determine the total number of experiments
    current = 0
    total = (len(seeds) * len(classes) * len(gen_functions) * len(eval_functions) * len(recombination_functions) *
             len(dimensions))

    # plot the generations
    print("Plotting the results ...")
    for key, generations in imports.items():
        # print the progress
        current += 1
        print(f"({current}/{total}): {key}")

        # plot the generations
        plot_generations(generations, name=key, directory=directory)

        print()  # print an empty line for better readability

    # inform the user that the script is finished
    print(f"Done! Results can be found in '{directory}'.")
