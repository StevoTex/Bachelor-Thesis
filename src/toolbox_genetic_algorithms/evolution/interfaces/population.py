import math
import os
import random as rdm
import time
from copy import copy

from matplotlib import pyplot as plt
from tqdm import tqdm

from ..operators import SEL_ROULETTE, SEL_BEST, EVAL_PARETO, MUT_GAUSS, GEN_HALTON, REC_CROSS_POINT
from ..metrics import METRIC_AVG
from ..strategies import STRAT_B
from .individual import Individual


class Population:
    """
    A population of individuals.
    :var _IndividualClass: The class of the individuals.
    :type _IndividualClass: Class
    :var _individuals: The individuals of the population.
    :type _individuals: list
    """

    def __init__(self, individual_class=Individual, size=1, gen_funct=GEN_HALTON, seed=None, **kwargs):
        """
        Creates a population of individuals of the given class.
        :param individual_class: The class of the individuals (subclass of Individual).
        :type individual_class: Class
        :param size: Number of individuals in the population.
        :type size: int
        :param kwargs: Keyword arguments that are passed to the individual class.
        """

        # set the individual class
        self._IndividualClass = individual_class

        if 'individuals' in kwargs:
            # set the individuals
            self._individuals = kwargs['individuals']
        else:
            self._individuals = self._IndividualClass.create(size, gen_funct, seed)
            for elem in self._individuals:
                print(elem)

    def add(self, individual):
        """
        Adds an individual to the population.
        :param individual: The individual to add.
        :type individual: Individual
        """
        self._individuals.append(individual)

    def get_individuals(self, order_by='id', reverse=True):
        """
        Returns the individuals of the population ordered by the given attribute. The default is to order by id.
        :param order_by: The attribute to order by. Expected values are 'id', 'fitness', 'diversity'.
        :type order_by: str
        :param reverse: Flag whether the individuals should be ordered descending.
        :type reverse: bool
        :return: The individuals of the population.
        :rtype: list
        """
        return self.order(by=order_by, reverse=reverse)

    def select_parents(self, n=1, pressure=1, sel_funct=SEL_ROULETTE, eval_funct=EVAL_PARETO):
        """
        Selects n parents from the population.
        :param n: Number of parents to select.
        :type n: int
        :param pressure: The pressure to apply to the selection function. Expected values are between 0 and 1.
        :type pressure: float
        :param sel_funct: The selection function to use.
        :type sel_funct: function
        :param eval_funct: The evaluation function to use for each individual.
        :type eval_funct: function
        :return: The selected parents.
        :rtype: list[Individual]
        """
        return sel_funct(self, n=n, pressure=pressure, eval_funct=eval_funct)

    def order(self, by='fitness', reverse=True):
        """
        Returns the individuals of the population ordered by the given attribute. The default is to order by fitness.
        This method is very similar tu the get_individuals method. The difference is in the default values of the
        parameters.
        :param by: The attribute to order by. Expected values are 'id', 'fitness', 'diversity'.
        :type by: str
        :param reverse: Flag whether the individuals should be ordered descending.
        :type reverse: bool
        :return: The best n individuals of the population.
        :rtype: List[Individual]
        """
        return self._IndividualClass.order(group=self._individuals, by=by, reverse=reverse)

    def get_goals(self):
        """
        :return: The optimization goals of the individuals.
        :rtype: tuple(int or float)
        """
        return self._IndividualClass.get_goals()

    def get_fitness(self, relative=True, metric=METRIC_AVG):
        """
        Calculates the fitness of the population based on the fitness of the individuals using the given metric.
        :param relative: Flag whether the relative fitness should be used.
        :type relative: bool
        :param metric: The metric to use for calculating the fitness.
        :type metric: function
        :return: The calculated fitness of the population.
        :rtype: float
        """

        # get the fitness of each considered individual
        if relative:
            fitness_per_ind = [ind.get_relative_fitness() for ind in self._individuals]
        else:
            fitness_per_ind = [ind.get_fitness() for ind in self._individuals]

        # calculate the fitness of the population using the given metric and return it
        return metric(*fitness_per_ind)

    def get_pareto_front(self):
        """
        :return: The pareto front of the population.
        :rtype: list[Individual]
        """
        values = EVAL_PARETO(self._individuals)
        paretos = [self._individuals[i] for i in range(len(self)) if values[i] == len(self)-1]
        return paretos

    def get_diversity(self, metric=METRIC_AVG):
        """
        Calculates the diversity of the population using the given metric.
        :param metric: The metric to use for calculating the diversity.
        :type metric: function
        :return: The calculated diversity of the population.
        :rtype: float
        """
        return self._IndividualClass.get_group_diversity(group=self._individuals, metric=metric)

    def plot(self, name='population', directory=None, fitness=False):
        """
        Creates a pairwise scatter plot of the genotypes of the individuals in the population. The fitness of each
        individual is represented by the color of the dot if the colored flag is set to True. The results are saved in
        the given directory with the given name as png file.
        :param name: The name of the file.
        :type name: str
        :param directory: The directory where the file should be saved.
        :type directory: str or None
        :param fitness: Flag whether the fitness should be plotted.
        :type fitness: bool
        :return:
        """

        # set the default directory to the directory of the script where the method is called
        if directory is None:
            directory = os.path.dirname(os.path.realpath(__file__))

        # create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # remove the file extension if it is given
        if name.endswith('.png'):
            name = name[:-4]

        # add an index to the name if the file already exists
        i = 1
        name_tmp = name
        while os.path.exists(os.path.join(directory, f'{name_tmp}.png')):
            i += 1
            name_tmp = f'{name} ({i})'
        name = name_tmp

        # get the individuals and extract the relevant information for plotting
        if fitness:
            individuals = self.get_individuals(order_by='fitness', reverse=False)
            fitness_values = [ind.get_relative_fitness() for ind in individuals]
        else:
            individuals = self.get_individuals()
        genotypes = [ind.get_genotype(transform=True) for ind in individuals]
        blueprint = self._IndividualClass.get_blueprint()

        # create the results
        fig, axes = plt.subplots(nrows=len(genotypes[0]), ncols=len(genotypes[0]), figsize=(10, 10))
        for i in range(len(genotypes[0])):
            for j in range(len(genotypes[0])):
                ax = axes[i, j]
                if j < i:
                    fig.delaxes(ax)
                    continue
                x_min = blueprint['genotype'][i].get_min()
                x_max = blueprint['genotype'][i].get_max()
                x_values = [g[i] for g in genotypes]
                if i == j:
                    ax.hist(x_values, bins=16, color='black')
                else:
                    y_min = blueprint['genotype'][j].get_min()
                    y_max = blueprint['genotype'][j].get_max()
                    y_values = [g[j] for g in genotypes]
                    if fitness:
                        ax.scatter(x_values, y_values, c=fitness_values)
                    else:
                        ax.scatter(x_values, y_values, edgecolor='black', facecolor='none')
                    ax.set_ylabel(blueprint['genotype_labels'][j])
                    ax.set_ylim(y_min, y_max)
                ax.set_xlabel(blueprint['genotype_labels'][i])
                ax.set_xlim(x_min, x_max)

        # render and save the results
        fig.tight_layout()
        fig.savefig(os.path.join(directory, f'{name}.png'), dpi=600)
        plt.close(fig)

    def next_generation(self, aep=.5, recombination_rate=.5, parent_group_size=2, **kwargs):
        """
        Creates the next generation of the population.
        :param aep: The assumed exploration progress. The higher the value, the more exploration is assumed and the less
            mutation will be applied. The value must be between 0 and 1.
        :type aep: float
        :param recombination_rate: The probability of recombination.
        :type recombination_rate: float
        :param parent_group_size: The number of parents in each group.
        :type parent_group_size: int
        :param kwargs: Keyword arguments (population_size, sel_funct, recombination_funct, mutation_funct,
            duplicate_free, elite, alien).
        :return: Population of the next generation.
        :rtype: Population
        """

        # validate aep
        if not (0 <= aep <= 1):
            raise ValueError("Assumed exploration progress (aep) must be between 0 and 1.")

        # validate recombination rate
        if not (0 <= recombination_rate <= 1):
            raise ValueError("Recombination rate must be between 0 and 1.")

        # get the keyword arguments
        population_size = kwargs.get('population_size', len(self))  # number of individuals in the population
        sel_funct = kwargs.get('sel_funct', SEL_ROULETTE)  # selection function
        eval_funct = kwargs.get('eval_funct', EVAL_PARETO)  # evaluation function
        recombination_funct = kwargs.get('recombination_funct', REC_CROSS_POINT)  # recombination method
        mutation_funct = kwargs.get('mutation_funct', MUT_GAUSS)  # mutation method
        duplicate_free = kwargs.get('duplicate_free', True)  # flag whether the population should be duplicate free
        elite = kwargs.get('elite', parent_group_size)  # how many of the best individuals should be taken over
        alien = kwargs.get('alien', 0)  # how many individuals should be alienated (generated randomly)

        # make sure that the selection size is a multiple of the parent group size
        selection_size = math.ceil((population_size - elite - alien) / parent_group_size) * parent_group_size

        # select the parents
        # TODO: selection pressure should be adjusted based on the aep, and the diversity of the population in future
        all_parents = self.select_parents(n=selection_size, sel_funct=sel_funct, eval_funct=eval_funct, pressure=1)

        # determine the relative quality of each parent
        quality_per_parent = eval_funct(all_parents)
        q_min = min(quality_per_parent)
        q_max = max(quality_per_parent)
        if q_min == q_max:
            quality_per_parent = [.5 for _ in quality_per_parent]
        else:
            quality_per_parent = [(q - q_min) / (q_max - q_min) for q in quality_per_parent]
        quality_dict = {all_parents[i].get_id(): quality_per_parent[i] for i in range(len(all_parents))}

        # split the parents into groups
        parent_groups = [all_parents[i:i + parent_group_size] for i in range(0, len(all_parents), parent_group_size)]

        # initialize the new population with the best individuals (elitism)
        new_population = [
            copy(ind) for ind in SEL_BEST(self._IndividualClass.get_all(), n=elite, eval_funct=eval_funct)
        ]

        # add alien individuals
        new_population.extend(self._IndividualClass.create(alien))

        # create the children
        for parents in parent_groups:
            # recombination
            if rdm.random() < recombination_rate:
                children = [recombination_funct(*parents) for _ in range(len(parents))]
                avg_rel_quality = METRIC_AVG(*[quality_dict[ind.get_id()] for ind in parents])
                expected_rel_quality = [avg_rel_quality for _ in parents]
            else:
                children = [copy(ind) for ind in parents]
                expected_rel_quality = [quality_dict[ind.get_id()] for ind in parents]

            # mutation
            for i in range(len(children)):
                # trying to search close to good solutions
                mutation_heat = .5 * (1 - .5 * aep - .5 * expected_rel_quality[i])
                children[i].mutate(method=mutation_funct, heat=mutation_heat)

            # add the children to the new population
            new_population.extend(children)

        # remove individuals if the population is too large
        if len(new_population) > population_size:
            new_population = new_population[:population_size]

        # remove duplicates by mutating them
        if duplicate_free:
            while len(new_population) != len(set(new_population)):
                for i in range(len(new_population)):
                    if new_population.count(new_population[i]) > 1:
                        new_population[i].mutate(method=mutation_funct, heat=1)

        # return the next generation
        return Population(self._IndividualClass, individuals=new_population)

    def evolve(self, n=1, strategy=STRAT_B):
        """
        Evolves the population for n generations.
        :param n: Number of generations to evolve.
        :type n: int
        :param strategy: The evolution strategy to use.
        :type strategy: function
        :return: The Generations resulting from the evolution.
        :rtype: List[Population]
        """

        # copy the population
        generations = [copy(self)]

        # evolve the population
        for i in tqdm(range(n), desc=f'Evolving {self._IndividualClass.__name__}s using {strategy.__name__} as '
                                     f'evolution strategy.'):
            generations.append(strategy(generations[-1], i, n))

        # sleep a second to make sure that the tqdm bar is displayed correctly
        time.sleep(1)

        # return the generations
        return generations

    def __str__(self):
        """
        :return: A short string representation of the population.
        :rtype: str
        """
        return f'{self.__class__.__name__} of {len(self)} {self._IndividualClass.__name__}s'

    def __getitem__(self, item):
        """
        :param item: Index of the individual.
        :type item: int
        :return: The individual at the given index.
        :rtype: Individual
        """
        return self.get_individuals()[item]

    def __iter__(self):
        """
        :return: An iterator over the individuals of the population.
        :rtype: Iterator[Individual]
        """
        return iter(self.get_individuals())

    def __len__(self):
        """
        :return: The number of individuals in the population.
        :rtype: int
        """
        return len(self._individuals)

    def __copy__(self):
        return Population(self._IndividualClass, individuals=[copy(ind) for ind in self._individuals])
