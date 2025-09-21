"""
In this script will generate a .tex file containing the last generations of cars evolved in 06-experiments.py.
"""
import os

import evolution as evo
from car import Car
from utilities import import_generations_from_csv

if __name__ == '__main__':
    # define where to store the results
    directory = "results/09-final-gens-to-latex"

    # determine the variables
    seeds = [42]  # removed 1234 and 2024 because this is only for a rough overview
    gen_functions = [evo.GEN_HALTON]  # removed GEN_GRID_HALTON because of edge cases
    gen_function_labels = ['Halton']
    eval_functions = [evo.EVAL_FITNESS, evo.EVAL_PARETO]
    eval_function_labels = ['Fitness', 'Pareto']
    rec_functions = [evo.REC_CROSS_ARITHMETIC, evo.REC_CROSS_POINT]
    rec_function_labels = ['Arithmetic', 'Point']
    classes = [Car]  # removed PseudoCar because we only want to show the final generations of real cars
    dimensions = [(10, 10), (50, 20)]  # (population size, number of generations)

    # import the generations evolved in 06-experiments.py
    print("Importing the generations ...")
    imports = {}
    titles = {}
    for seed in seeds:
        for ig, gen_funct in enumerate(gen_functions):
            for ie, eval_funct in enumerate(eval_functions):
                for ir, rec_funct in enumerate(rec_functions):
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

                            # create the title
                            titles[key] = f"{eval_function_labels[ie]}-Eval., " \
                                          f"{rec_function_labels[ir]}-Cross., " \
                                          f"Size: {n_pop}, " \
                                          f"{n_gen} Gens."

    # create the .tex file
    print("Creating the .tex file ...")

    # create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # open the file
    with open(os.path.join(directory, "final-generations.tex"), "w") as file:
        # write the preamble
        file.write("\\documentclass{article}\n")
        file.write("\\usepackage{graphicx}\n")
        file.write("\\usepackage{subcaption}\n")
        file.write("\\usepackage{geometry}\n")
        file.write("\\geometry{a4paper, margin=1in}\n")
        file.write("\\begin{document}\n")

        # add some empty lines
        file.write("\n\n")

        # write the content
        for key, generations in imports.items():

            # get the final generation and the blueprint
            pop = generations[-1]
            blueprint = pop[0].get_blueprint()

            # shorten the labels
            blueprint['genotype_labels'] = ['FD', 'RR', 'G3', 'G4', 'G5']
            blueprint['phenotype_labels'] = ['cons', 'els3', 'els4', 'els5']

            # open the wrapper
            file.write("\\begin{table}[H]\n")
            file.write("\\begin{tabularx}{\\textwidth}{")
            for i in range(len(blueprint['genotype']) + len(blueprint['goals']) + 1):
                file.write("| X ")
            file.write("|}\n")
            file.write("\\hline\n")

            # write the header
            file.write("\\textbf{Rank}")
            for gene in blueprint['genotype_labels']:
                file.write(" & \\textbf{" + gene + "}")
            for objective in blueprint['phenotype_labels']:
                file.write(" & \\textbf{" + objective + "}")
            file.write("\\\\ \\hline\n")

            # write the generations
            for i, ind in enumerate(pop.order()):
                # max. 10 individuals
                if i >= 10:
                    break

                # write the index
                file.write(f"\\textbf{{{i + 1}}}")

                # write the genotype
                for gene in ind.get_genotype():
                    file.write(" & " + str(round(gene.get(), 3)))

                # write the goals
                for objective in ind.get_phenotype():
                    file.write(" & " + str(round(objective, 3)))

                # close the row
                file.write("\\\\ \\hline\n")

            # close the wrapper
            file.write("\\end{tabularx}\n")
            slug = key.lower().replace(" ", "-").replace(",", "").replace(":", "")
            file.write("\\label{tab:" + slug + "}\n")
            file.write("\\caption{" + titles[key] + "}\n")
            file.write("\\end{table}\n")

            # add some empty lines
            file.write("\n\n")

        # write the end of the document
        file.write("\\end{document}\n")

    # inform the user that the script is done
    print("Done! Stored the .tex file in", directory, ".")
