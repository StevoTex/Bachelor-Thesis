"""

"""
import os

import evolution as evo
from car import Car
from utilities import import_generations_from_csv, plot_generations

if __name__ == '__main__':
    # define where to store the results
    directory = "results/08-ranking"

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
                        for n_pop, n_gen in dimensions:  # generate a key for the experiment name
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

    for cls in classes:
        # determine the total number of experiments and inform the user about the progress
        print("Calculating the scores for", cls.__name__, "...")
        count = 0
        total = len(gen_functions) * len(eval_functions) * len(recombination_functions)

        # extract the last generation of each experiment
        last_generation = {}
        for key, generations in imports.items():
            last_generation[key] = generations[-1]

        # calculate a score for the best car in each generation
        scores_best = {}
        scores_avg = {}
        scores_diversity = {}
        for gen_funct in gen_functions:
            for eval_funct in eval_functions:
                for rec_funct in recombination_functions:
                    # print the progress
                    count += 1
                    print(f"({count}/{total}): {cls.__name__}-{gen_funct.__name__}-{eval_funct.__name__}-"
                          f"{rec_funct.__name__}")

                    # begin to calculate the score
                    score_key = f"{cls.__name__}-{gen_funct.__name__}-{eval_funct.__name__}-{rec_funct.__name__}"
                    fitness_max = []
                    fitness_avg = []
                    diversity = []
                    for seed in seeds:
                        for n_pop, n_gen in dimensions:
                            # get the key of the experiments name
                            key = (f"{cls.__name__}-{gen_funct.__name__}-{eval_funct.__name__}-{rec_funct.__name__}-"
                                   f"{n_pop}-{n_gen}-{seed}")

                            # skip the halton grid experiment for cars
                            if cls.__name__ == "Car" and gen_funct.__name__ == "GEN_GRID_HALTON":
                                continue

                            # get the last generation
                            last_gen = last_generation[key]

                            # append the objectives to the lists
                            fitness_max.append(last_gen.get_fitness(metric=evo.METRIC_MAX))
                            fitness_avg.append(last_gen.get_fitness(metric=evo.METRIC_AVG))
                            diversity.append(last_gen.get_diversity())

                    # skip the score if there are no values (e.g. because of the halton grid experiment)
                    if len(fitness_max) == 0:
                        continue

                    # set the score to the average
                    scores_best[score_key] = sum(fitness_max) / len(fitness_max)
                    scores_avg[score_key] = sum(fitness_avg) / len(fitness_avg)
                    scores_diversity[score_key] = sum(diversity) / len(diversity)

        # normalize the scores to make them easier to compare
        for score in [scores_best, scores_avg, scores_diversity]:
            # get the minimum and maximum
            minimum = min(score.values())
            maximum = max(score.values())

            # normalize the scores
            for key, value in score.items():
                if maximum == minimum:
                    score[key] = .5
                else:
                    score[key] = (value - minimum) / (maximum - minimum)

        # calculate the total score
        scores_total = {}
        for key in scores_best.keys():
            scores_total[key] = scores_best[key] + scores_avg[key] + scores_diversity[key]

        # normalize the total score
        minimum = min(scores_total.values())
        maximum = max(scores_total.values())
        for key, value in scores_total.items():
            if maximum == minimum:
                scores_total[key] = .5
            else:
                scores_total[key] = (value - minimum) / (maximum - minimum)

        # inform the user about the progress
        print("Exporting the scores ...")

        # create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # create ranking lists for each score and export them to csv
        for score in [scores_best, scores_avg, scores_diversity, scores_total]:
            # create a ranking list
            ranking = []
            for key, value in score.items():
                ranking.append((key, value))

            # sort the ranking
            ranking.sort(key=lambda x: x[1], reverse=True)

            # determine the name of the score
            score_name = "best" if score == scores_best else "avg" if score == scores_avg else "diversity" \
                if score == scores_diversity else "total"
            name = f"{cls.__name__}-{score_name}"

            # export the ranking to csv
            with open(f"{directory}/ranking-{name}.csv", "w") as file:
                # write the header
                file.write("experiment,score\n")

                for key, value in ranking:
                    file.write(f"{key},{value}\n")

    # inform the user that the script is finished
    print(f"Done! Results can be found in '{directory}'.")