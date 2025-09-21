"""
This module contains some helper functions that are used in the project.
"""
import os

from matplotlib import pyplot as plt

import evolution as evo


def export_generations_to_csv(generations, name="generations", directory=None, header=True, delimiter=";"):
    """
    Exports the given generations to a csv file.
    :param generations: The generations to export.
    :type generations: list[Population]
    :param name: The name of the file.
    :param name: str
    :param directory: The directory to save the file in.
    :type directory: str
    :param header: Flag whether to include a header in the file.
    :type header: bool
    :param delimiter: The delimiter to use.
    :type delimiter: str
    :return: The path to the file.
    :rtype: str
    :return:
    """

    # set the default directory to directory of the script where this function is called
    if directory is None:
        directory = os.path.dirname(os.path.realpath(__file__))

    # create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # remove the file extension if it is given
    if name.endswith(".csv"):
        name = name[:-4]

    # check if the file already exists and if so, add a number to the name
    name_tmp = name
    i = 1
    while os.path.exists(os.path.join(directory, f"{name_tmp}.csv")):
        i += 1
        name_tmp = f"{name} ({i})"
    name = name_tmp + ".csv"

    # get the blueprint of the individuals
    ind_cls = generations[0][0].__class__
    blueprint = ind_cls.get_blueprint()

    # create the file
    # create the file
    with open(os.path.join(directory, name), 'w') as f:
        # write the header
        if header:
            f.write('generation')
            for allele_label in blueprint['genotype_labels']:
                f.write(f'{delimiter}{allele_label}')
            for phenotype_label in blueprint['phenotype_labels']:
                f.write(f'{delimiter}{phenotype_label}')
            f.write('\n')

        # write the data
        for i, pop in enumerate(generations):
            for ind in pop.get_individuals():
                f.write(f'{i}')
                for allele in ind.get_genotype():
                    f.write(f'{delimiter}{allele.get()}')
                for objective in ind.get_phenotype():
                    f.write(f'{delimiter}{objective}')
                f.write('\n')

    # print and return the path to the file
    print(f"Exported {len(generations)} generations to {os.path.join(directory, name)}")
    return os.path.join(directory, name)


def import_generations_from_csv(cls, name="generations", directory=None, header=True, delimiter=";"):
    """
    Imports the generations from a csv file.
    :param cls: The class of the individuals.
    :type cls: Individual
    :param name: The name of the file.
    :param name: str
    :param directory: The directory to load the file from.
    :type directory: str
    :param header: Flag whether the file contains a header.
    :type header: bool
    :param delimiter: The used delimiter.
    :type delimiter: str
    :return: The generations.
    :return: list[Population]
    """

    # remove the file extension if it is given
    if name.endswith(".csv"):
        name = name[:-4]

    # set the default directory to directory of the script where this function is called
    if directory is None:
        directory = os.path.dirname(os.path.realpath(__file__))

    # determine the path to the file
    path = os.path.join(directory, f"{name}.csv")

    # check if the file exists
    if not os.path.exists(os.path.join(directory, f"{name}.csv")):
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    # get the blueprint of the individuals
    blueprint = cls.get_blueprint()

    # read the file
    with open(path, 'r') as f:
        # skip the header
        if header:
            f.readline()

        # create the generations
        generations = []
        for line in f.readlines():
            # split the line
            line = line.split(delimiter)

            # extract the generation number
            generation_number = int(line[0])

            # extract the genotype
            start = 1
            end = len(blueprint['genotype']) + 1
            genotype = [float(value) for value in line[start:end]]

            # extract the phenotype
            start = end
            end = start + len(blueprint['goals'])
            phenotype = [float(value) for value in line[start:end]]

            # create a new population if necessary
            if generation_number >= len(generations):
                generations.append(evo.Population(cls, size=0))

            # create the Alleles
            alleles = []
            for i, allele_cls in enumerate(blueprint['genotype']):
                alleles.append(allele_cls(genotype[i]))

            # create the individual
            ind = cls(*alleles)

            # set the phenotype
            ind._phenotype = tuple(phenotype)  # this is a hack to prevent the phenotype from being calculated again

            # add the individual to the population
            generations[generation_number].add(ind)

    # return the generations
    return generations


def plot_generations(generations, name='generations-plot', directory=None):
    """
    Plot the generations of an experiment.
    :param generations: The generations to plot.
    :type generations: list[Population]
    :param name: The name of the experiment.
    :type name: str
    :param directory: The directory to save the results to.
    :type directory: str
    """

    print(f"Plotting generations...")

    # set the default directory to the directory of the script where the method is called
    if directory is None:
        directory = os.path.dirname(os.path.realpath(__file__))

    # create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # add an index to the name if the directory already exists
    i = 1
    name_tmp = name
    while os.path.exists(os.path.join(directory, f'{name_tmp}')):
        i += 1
        name_tmp = f'{name} ({i})'
    name = name_tmp

    # create the directory (name)
    os.makedirs(os.path.join(directory, name))

    # get the class of the individuals
    individual_class = generations[-1][0].__class__

    # initialize the lists to store the data
    obj = []

    # process phenotype data
    for i, goal in enumerate(individual_class.get_goals()):
        objectives = [[ind.get_phenotype()[i] for ind in pop.get_individuals()] for pop in generations]
        objectives_max = [max(objectives[j]) for j in range(len(objectives))]
        objectives_min = [min(objectives[j]) for j in range(len(objectives))]
        objectives_avg = [sum(objectives[j]) / len(objectives[j]) for j in range(len(objectives))]
        obj.append((objectives_max, objectives_min, objectives_avg))

    blueprint = individual_class.get_blueprint()
    switcher = {}
    for i, goal in enumerate(blueprint['phenotype_labels']):
        switcher[i] = goal

    # create all results on one figure and save them as png
    fig, axes = plt.subplots(ncols=len(obj), nrows=1, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.plot(obj[i][0], label='MAX', color='black')
        ax.plot(obj[i][1], label='MIN', color='black')
        ax.plot(obj[i][2], label='AVG', color='black')
        ax.set_ylabel(switcher[i])
        ax.set_xlabel('Generation')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, name, 'phenotype.png'), dpi=600)
    plt.close(fig)

    # process trend (the average fitness of the pareto front)
    p_fronts = [pop.get_pareto_front() for pop in generations]
    p_front_fitness = []
    for i, p_front in enumerate(p_fronts):
        if len(p_front) > 0:
            p_front_fitness.append([ind.get_relative_fitness() for ind in p_front])
        else:
            p_front_fitness.append([0])
    # p_front_fitness_max = [max(p_front_fitness[j]) for j in range(len(p_front_fitness))]
    # p_front_fitness_min = [min(p_front_fitness[j]) for j in range(len(p_front_fitness))]
    p_front_fitness_avg = [sum(p_front_fitness[j]) / len(p_front_fitness[j]) for j in range(len(p_front_fitness))]

    # process fitness data
    fitness = [[ind.get_relative_fitness() for ind in pop.get_individuals()] for pop in generations]
    fitness_max = [max(fitness[j]) for j in range(len(fitness))]
    fitness_min = [min(fitness[j]) for j in range(len(fitness))]
    fitness_avg = [sum(fitness[j]) / len(fitness[j]) for j in range(len(fitness))]
    fit = [(fitness_max, fitness_min, fitness_avg)]

    # process diversity data
    diversity_avg = [[individual_class.get_group_diversity(pop.get_individuals())] for pop in generations]

    # create all results on one figure and save them as png
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    axes[0].plot(p_front_fitness_avg, label='AVG', color='black')
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Trend')
    axes[0].set_xlabel('Generation')
    axes[1].plot(fit[0][0], label='MAX', color='black')
    axes[1].plot(fit[0][1], label='MIN', color='black')
    axes[1].plot(fit[0][2], label='AVG', color='black')
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Fitness')
    axes[1].set_xlabel('Generation')
    axes[2].plot(diversity_avg, label='AVG', color='black')
    axes[2].set_ylabel('Diversity')
    axes[2].set_xlabel('Generation')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, name, 'quality.png'), dpi=600)
    plt.close(fig)

    for i in [0, round(len(generations) / 2), len(generations) - 1]:
        generations[i].plot(name=f"generation-{i}", directory=os.path.join(directory, name), fitness=True)

    print(f"Done! Stored results in {os.path.join(directory, name)}")
