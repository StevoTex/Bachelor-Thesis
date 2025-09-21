import math
from abc import abstractmethod, ABC

import numpy as np

from .allele import Allele
from ..benchmark import Benchmark
from ..operators import GEN_GRID_HALTON, GEN_SOBOL_HALTON, MUT_GAUSS
from ..metrics import METRIC_AVG


class Individual(ABC):
    """
    Abstract base class for individuals. An individual represents a potential solution to an optimization problem.
    :var _Blueprint: Blueprint for genotype and optimization goals of the phenotype.
    :type _Blueprint: dict{'genotype': list[Class(Allele)], 'goals': list[float or int]}
    :var _genotype: The genotype of the individual (tuple of Alleles according to the blueprint).
    :type _genotype: tuple(Allele)
    :var _phenotype: The phenotype of the individual.
    :type _phenotype: tuple(float) or None
    :var _instances: List of all instances of the class.
    :type _instances: list[Individual]
    """

    # Blueprint for genotype and optimization goals of the phenotype.
    # The 'genotype' key should contain a list of Allele classes, defining the structure of the individual's genotype.
    # The 'genotype_labels' key should contain a list of strings, defining the labels for the alleles.
    # The 'goals' key should contain a list of numeric values representing the optimization goals for each phenotype
    # The 'phenotype_labels' key should contain a list of strings, defining the labels for the phenotype
    # attribute. Positive values indicate maximization, negative values indicate minimization, and zero indicates no
    # preference. The absolute value of each position represents the weight or priority of the corresponding goal.
    _Blueprint = {
        'genotype': [Allele, Allele, Allele],  # example: three alleles
        'genotype_labels': ['Allele 1', 'Allele 2', 'Allele 3'],  # example: labels for the alleles
        'goals': [-1, 0, 2],  # example: minimize Allele 1, ignore Allele 2, maximize Allele 3 with double weight
        'phenotype_labels': ['Phenotype 1', 'Phenotype 2', 'Phenotype 3']  # example: labels for the phenotype
    }

    # pseudo classes (if exists)
    _pseudo = {}

    # list of all instances of the class
    _instances = []

    # caches
    _class_cache = {}

    def __init__(self, *genotype):
        """
        :param genotype: The genotype of the individual (list of Alleles).
        :type genotype: Allele
        """

        # set the id of the individual
        self._id = len(self.__class__._instances)

        if genotype is None:
            genotype = self.__class__.create(1, gen_funct=GEN_SOBOL_HALTON, return_genotypes=True)[0]

        # initialize genotype
        self._genotype = genotype
        self._phenotype = None

        # add the instance to the list of all instances of the class
        self.__class__._instances.append(self)

        # clear caches (phenotype is set to None)
        self._clear_caches()

        # make sure that the genotype is valid
        # self._enforce_constraints()

    def get_id(self):
        """
        :return: The id of the individual.
        :rtype: int
        """
        return self._id

    @classmethod
    def get_all(cls):
        """
        :return: All instances of the class.
        :return: list[Individual]
        """
        return cls._instances

    @classmethod
    def get_blueprint(cls):
        """
        :return: The blueprint of the class.
        :rtype: dict
        """
        return cls._Blueprint

    def get_genotype(self, transform=False, normalized=False):
        """
        :param transform: Flag whether the genotype should be returned as a tuple of values.
        :type transform: bool
        :param normalized: Flag whether the values should be normalized.
        :type normalized: bool
        :return: The genotype of the individual.
        :rtype: tuple
        """
        if transform:
            return tuple([allele.get(normalized=normalized) for allele in self._genotype])
        return self._genotype

    def get_phenotype(self):
        """
        :return: The phenotype of the individual.
        :rtype: tuple
        """

        # calculate the phenotype if it is not cached
        if self._phenotype is None:
            self._phenotype = self._calculate_phenotype()

        # Verify that the phenotype matches the blueprint
        if len(self._phenotype) != len(self._Blueprint['goals']):
            raise ValueError("Phenotype does not match the Blueprint (goals).")

        return self._phenotype

    @classmethod
    def get_min_phenotype(cls, others=None):
        """
        Returns a tuple of the minimum phenotype values for each phenotype attribute of the given individuals. If no
        individuals are given, all instances of the class are used.
        :param others: Other individuals. If None, all instances of the class are used.
        :type others: list[Individual] or None
        :return: The minimum phenotype values for each phenotype attribute.
        :rtype: tuple(float)
        """

        # set the default value for others to all instances of the class and whether to cache the result
        if others is None:
            if "min_phenotype" in cls._class_cache:
                # return cached value
                return cls._class_cache["min_phenotype"]
            others = cls._instances
            cache = True
        else:
            cache = False

        # calculate the minimum phenotype values
        min_phenotype = [float('inf')] * len(cls._Blueprint['goals'])
        for individual in others:
            phenotype = individual.get_phenotype()
            for i in range(len(phenotype)):
                min_phenotype[i] = min(min_phenotype[i], phenotype[i])

        # convert the result to a tuple
        min_phenotype = tuple(min_phenotype)

        # cache the result
        if cache:
            cls._class_cache["min_phenotype"] = min_phenotype

        # return the result
        return min_phenotype

    @classmethod
    def get_max_phenotype(cls, others=None):
        """
        Returns a tuple of the maximum phenotype values for each phenotype attribute of the given individuals. If no
        individuals are given, all instances of the class are used.
        :param others: Other individuals. If None, all instances of the class are used.
        :type others: list[Individual] or None
        :return: The maximum phenotype values for each phenotype attribute.
        :rtype: tuple(float)
        """

        # set the default value for others to all instances of the class and whether to cache the result
        if others is None:
            if "max_phenotype" in cls._class_cache:
                # return cached value
                return cls._class_cache["max_phenotype"]
            others = cls._instances
            cache = True
        else:
            cache = False

        # calculate the maximum phenotype values
        max_phenotype = [-float('inf')] * len(cls._Blueprint['goals'])
        for individual in others:
            phenotype = individual.get_phenotype()
            for i in range(len(phenotype)):
                max_phenotype[i] = max(max_phenotype[i], phenotype[i])

        # convert the result to a tuple
        max_phenotype = tuple(max_phenotype)

        # cache the result
        if cache:
            cls._class_cache["max_phenotype"] = max_phenotype

        # return the result
        return max_phenotype

    def get_relative_phenotype(self, others=None):
        """
        Returns the relative phenotype of the individual compared to the passed individuals. If no individuals are
        given, all instances of the class are used.
        :param others: Other individuals. If None, all instances of the class are used.
        :type others: list[Individual] or None
        :return: The relative phenotype of the individual.
        :rtype: tuple(float)
        """

        phenotype = self.get_phenotype()
        min_phenotype = self.__class__.get_min_phenotype(others=others)
        max_phenotype = self.__class__.get_max_phenotype(others=others)
        relative_phenotype = []

        for i in range(len(phenotype)):
            if np.isclose(max_phenotype[i], min_phenotype[i], atol=1e-6):  # avoid division by zero
                relative_phenotype.append(.5)
            else:
                relative_phenotype.append((phenotype[i] - min_phenotype[i]) / (max_phenotype[i] - min_phenotype[i]))

        return tuple(relative_phenotype)

    @classmethod
    def get_goals(cls):
        """
        :return: The optimization goals for the phenotype.
        :rtype: tuple(float)
        """
        return tuple(cls._Blueprint['goals'])

    @classmethod
    def get_max_distance(cls, p=2, normalized=True):
        """
        Returns the maximum possible distance between two individuals of the class.
        :param p: Which p-norm to use (p=1: Manhattan distance, p=2: Euclidean distance, p=inf: maximum distance).
        :type p: int
        :param normalized: Flag whether the axes should be normalized before calculating the distance.
        :type normalized: bool
        :return: The maximum possible distance between two individuals of the class.
        :rtype: float or int
        """

        if normalized:
            if p == float('inf'):
                result = 1
            else :
                result = len(cls._Blueprint['genotype']) ** (1 / p)
        else:
            axes = [allele.get_max() - allele.get_min() for allele in cls._Blueprint['genotype']]
            if p == float('inf'):
                result = max(axes)
            else:
                result = sum([axis ** p for axis in axes]) ** (1 / p)

        return result

    def get_diversity(self, others=None, p=2, normalized=True, metric=METRIC_AVG):
        """
        Returns the diversity of the individual compared to the passed individuals. If no individuals are given, all
        instances of the class are used.
        :param others: Other individuals. If None, all instances of the class are used.
        :type others: list[Individual] or None
        :param p: Which p-norm to use calculating the distances (p=1: Manhattan distance, p=2: Euclidean distance,
            p=inf: maximum distance).
        :type p: int or float
        :param normalized: Flag whether the calculation distance should be normalized.
        :type normalized: bool
        :param metric: The metric to use for calculating the diversity.
        :type metric: function
        :return: The calculated diversity.
        :rtype: float or int
        """

        # set the default value for others to all instances of the class
        if others is None:
            others = self.__class__._instances

        other_ids = sorted([other._id for other in others])

        # create a cache key
        cache_key = ('diversity', self._id, tuple(other_ids), p, normalized, metric.__name__)

        # check if the result is already in the cache
        if cache_key in self.__class__._class_cache:
            return self.__class__._class_cache[cache_key]

        # Calculation of the maximum possible distance between two individuals
        max_dist = self.__class__.get_max_distance(p=p)

        # calculate the distances between the individuals
        values = []
        for other in others:
            if other._id != self._id:
                values.append(self.__class__.distance(self, other, p=p, normalized=normalized) / max_dist)

        # calculate the average distance
        result = metric(*values)

        # cache the result and return it
        self.__class__._class_cache[cache_key] = result
        return result

    @classmethod
    def get_group_diversity(cls, group=None, p=2, normalized=True, metric=METRIC_AVG):
        """
        Returns the diversity of the group of individuals. If no individuals are given, all instances of the class are
        used.
        :param group: The group of individuals. If None, all instances of the class are used.
        :type group: list[Individual] or None
        :param p: Which p-norm to use calculating the distances (p=1: Manhattan distance, p=2: Euclidean distance,
            p=inf: maximum distance).
        :type p: int or float
        :param normalized: Flag whether the single distances should be normalized.
        :type normalized: bool
        :param metric: The metric to use for calculating the diversity.
        :type metric: function
        :return: The calculated diversity.
        :rtype: float or int
        """

        # set the default value for group to all instances of the class
        if group is None:
            group = cls._instances

        # create a cache key
        individual_ids = sorted([individual._id for individual in group])
        cache_key = ('group_diversity', tuple(individual_ids), p, normalized, metric.__name__)

        # check if the result is already in the cache
        if cache_key in cls._class_cache:
            return cls._class_cache[cache_key]

        # Calculation of the maximum possible distance between two individuals
        max_dist = cls.get_max_distance(p=p)

        # calculate the distances between the individuals
        values = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                values.append(cls.distance(group[i], group[j], p=p, normalized=normalized) / max_dist)

        # calculate the average distance
        result = metric(*values)

        # cache the result and return it
        cls._class_cache[cache_key] = result
        return result

    def get_fitness(self, others=None):
        """
        Returns the fitness of the individual based on the relative phenotype and the optimization goals defined in the
        blueprint.
        :param others: Other individuals.
        :type others: list[Individual] or None
        :return: The fitness of the individual.
        :rtype: float
        """

        # initialize variables
        rel_phenotype = self.get_relative_phenotype(others=others)
        goals = self.__class__._Blueprint['goals']
        fitness = 0

        # calculate the fitness
        for i in range(len(rel_phenotype)):
            if goals[i] >= 0:
                fitness += goals[i] * rel_phenotype[i]
            else:  # goals[i] < 0:
                fitness += abs(goals[i]) * (1 - rel_phenotype[i])

        # normalize and return the fitness
        goals_sum = sum([abs(goal) for goal in goals])
        if goals_sum == 0:
            return .5
        fitness /= goals_sum
        return fitness

    @classmethod
    def get_min_fitness(cls, others=None):
        """
        Returns the minimum fitness of the given individuals. If no individuals are given, all instances of the class
        are used.
        :param others: Other individuals. If None, all instances of the class are used.
        :type others: list[Individual] or None
        :return: The minimum fitness of the given individuals.
        :rtype: float
        """

        if others is None:
            if "min_fitness" in cls._class_cache:
                # return cached value
                return cls._class_cache["min_fitness"]
            others = cls._instances
            cache = True
        else:
            cache = False

        # calculate the minimum fitness
        min_fitness = float('inf')
        for individual in others:
            min_fitness = min(min_fitness, individual.get_fitness())

        # cache the result
        if cache:
            cls._class_cache["min_fitness"] = min_fitness

        # return the result
        return min_fitness

    @classmethod
    def get_max_fitness(cls, others=None):
        """
        Returns the maximum fitness of the given individuals. If no individuals are given, all instances of the class
        are used.
        :param others: Other individuals. If None, all instances of the class are used.
        :type others: list[Individual] or None
        :return: The maximum fitness of the given individuals.
        :rtype: float
        """

        if others is None:
            if "max_fitness" in cls._class_cache:
                # return cached value
                return cls._class_cache["max_fitness"]
            others = cls._instances
            cache = True
        else:
            cache = False

        # calculate the maximum fitness
        max_fitness = -float('inf')
        for individual in others:
            max_fitness = max(max_fitness, individual.get_fitness())

        # cache the result
        if cache:
            cls._class_cache["max_fitness"] = max_fitness

        # return the result
        return max_fitness

    def get_relative_fitness(self, others=None):
        """
        Returns the relative fitness of the individual compared to the passed individuals. If no individuals are given,
        all instances of the class are used.
        :param others: Other individuals. If None, all instances of the class are used.
        :type others: list[Individual] or None
        :return: The relative fitness of the individual.
        :rtype: float
        """
        fitness = self.get_fitness()
        min_fitness = self.__class__.get_min_fitness(others=others)
        max_fitness = self.__class__.get_max_fitness(others=others)
        relative_fitness = (fitness - min_fitness) / (max_fitness - min_fitness)
        return relative_fitness

    @abstractmethod
    def _calculate_phenotype(self):
        """
        This abstract method should calculate the phenotype of the individual based on its genotype. The
        implementation of this method is specific to the structure of the genotype and the type of optimization
        problem. It should be overwritten in each derived class. The phenotype is the manifested form of the genotype
        and represents the characteristics of the individual that are evaluated in the evolutionary algorithm.
        :return: The calculated phenotype.
        :rtype: tuple(float)
        """
        # Example: set the phenotype to a tuple of zeros
        return tuple([.0] * len(self._Blueprint['goals']))

    def _enforce_constraints(self):
        """
        Should implement mechanisms to enforce any constraints.
        :return: void
        :rtype: None
        """
        pass  # can be implemented in the subclass if required

    @classmethod
    def create(cls, n=1, gen_funct=GEN_GRID_HALTON, seed=None, return_genotypes=False):
        """
        Creates n individuals using the given generation method. The method is used to distribute the individuals in an
        evolutionary algorithm.
        :param n: Number of individuals to create.
        :type n: int
        :param gen_funct: The method to distribute the individuals in the search space.
        :type gen_funct: function
        :param seed: The seed of the generation method. A seed can be used to make the generation deterministic.
        :param return_genotypes: Flag whether the genotypes should be returned instead of the individuals.
        :type return_genotypes: bool
        :return: A list of individuals or genotypes (if return_genotypes is True).
        :rtype: list
        """

        if n == 0:
            return []

        points = gen_funct(n, len(cls._Blueprint['genotype']), seed)
        result = []
        for i in range(len(points)):
            genotype = []
            for j in range(len(cls._Blueprint['genotype'])):
                genotype.append(cls._Blueprint['genotype'][j](points[i][j], True))
            if return_genotypes:
                result.append(tuple(genotype))
            else:
                result.append(cls(*genotype))
        return result

    @classmethod
    def pseudo_class(cls, complexity=.5, seed=None):
        """
        Generates a pseudo individual which is a subclass of the individual class. The pseudo individual can be used
        to test algorithms without using a cost intensive simulation.
        :param complexity: The complexity of the pseudo simulation (0: simple, 1: complex).
        :type complexity: float or int
        :param seed: The seed of the pseudo simulation. A seed can be used to make the simulation deterministic.
        :type seed: int
        :return: The Pseudo Individual.
        :rtype: Class(Individual)
        """

        cache_key = (complexity, seed)
        if cls._pseudo.get(cache_key) is not None:
            return cls._pseudo[cache_key]

        # create the pseudo simulation
        simulation = Benchmark(m=len(cls._Blueprint['genotype']), n=len(cls._Blueprint['goals']), complexity=complexity,
                               seed=seed)

        # create the pseudo individual
        class PseudoIndividual(cls):
            def _calculate_phenotype(self):
                return simulation(*self.get_genotype(transform=True, normalized=True))

            @classmethod
            def get_sim(cls):
                return simulation

        cls._pseudo[cache_key] = PseudoIndividual
        return PseudoIndividual

    def _clear_caches(self):
        """
        Clears the caches of the Class and the instance.
        :rtype: void
        """
        self._phenotype = None
        self.__class__._class_cache = {}

    def mutate(self, method=MUT_GAUSS, heat=1):
        """
        Performs a mutation of the individual, changing the genetic characteristics according to the specified
        mutation method and intensity. The method adjusts the alleles of the individual to create new variations in
        the population, which is an important component in exploring the solution space in evolutionary algorithms.
        :param method: The mutation method to use. A Gaussian mutation is used by default.
        :type method: function
        :param heat: The heat of the mutation.
        :type heat: float or int
        :rtype: void
        """

        # mutate the individual
        for allele in self._genotype:
            allele.mutate(method=method, heat=heat)

        # clear the cache
        self._clear_caches()
        self._enforce_constraints()

    @classmethod
    def distance(cls, a, b, p=2, normalized=True):
        """
        Returns the distance between two individuals based on their genotype using a p-norm.
        :param p: Which p-norm to use (p=1: Manhattan distance, p=2: Euclidean distance, p=inf: maximum distance).
        :type p: int or float
        :param a: The first individual.
        :type a: Individual
        :param b: The second individual.
        :type b: Individual
        :param normalized: Whether values of the alleles should be normalized before calculating the distance.
        :type normalized: bool
        :return: The distance between the two individuals.
        :rtype: float
        """

        # make sure that a is the individual with the lower id
        if a._id == b._id:
            return 0  # the distance between an individual and itself is always 0
        elif a._id > b._id:
            a, b = b, a

        # create a cache key
        cache_key = ('distance', a._id, b._id, p, normalized)

        # check if the result is already in the cache
        if cache_key in cls._class_cache:
            return cls._class_cache[cache_key]

        # calculate the distance
        genotype_a = a.get_genotype(transform=True, normalized=normalized)
        genotype_b = b.get_genotype(transform=True, normalized=normalized)

        # calculate the distance
        if p == float('inf'):
            dist = max([abs(genotype_a[i] - genotype_b[i]) for i in range(len(genotype_a))])
        else:
            dist = sum([abs(genotype_a[i] - genotype_b[i]) ** p for i in range(len(genotype_a))])
            dist = dist ** (1 / p)

        # cache the result and return it
        cls._class_cache[cache_key] = dist
        return dist

    def __str__(self):
        """
        :return: A string representation of the individual.
        :rtype: str
        """
        return f'{self.__class__.__name__}({str(self._genotype)})'

    def __eq__(self, other):
        """
        :param other: The other individual.
        :type other: Individual
        :return: True if the individuals are equal, False otherwise.
        :rtype: bool
        """
        return self.__class__ == other.__class__ and self._genotype == other._genotype

    def __copy__(self):
        """
        :return: A copy of the individual.
        :rtype: Individual
        """
        copy = self.__class__(*[allele.__copy__() for allele in self._genotype])  # create a copy of the individual
        copy._phenotype = self._phenotype  # copy the phenotype for better performance
        return copy  # return the copy

    def __hash__(self):
        """
        :return: The hash value of the individual.
        :rtype: int
        """
        return hash(self._genotype)

    @classmethod
    def order(cls, group=None, by='fitness', reverse=True):
        """
        Returns a list of all instances wich are ordered by the given attribute. The default is to order by fitness. If
        no group is given, all instances of the class are used.
        :param group: The group of individuals to order. If None, all instances of the class are used.
        :type group: list[Individual] or None
        :param by: The attribute to order by ('id', 'fitness' or 'diversity').
        :type by: str
        :param reverse: Flag whether the individuals should be ordered descending.
        :type reverse: bool
        :return: All instances of the class order by their fitness.
        :rtype: list[Individual]
        """

        # set the default value for group to all instances of the class
        if group is None:
            group = cls._instances

        switcher = {
            'id': lambda individual: individual.get_id(),
            'fitness': lambda individual: individual.get_fitness(),
            'diversity': lambda individual: individual.get_diversity(),
        }

        # return the result
        return sorted(group, key=switcher[by], reverse=reverse)

    @classmethod
    def reset(cls):
        """
        Resets the class. All instances are removed from future calculations and the caches are cleared.
        :rtype: void
        """
        cls._instances = []
        cls._class_cache = {}
