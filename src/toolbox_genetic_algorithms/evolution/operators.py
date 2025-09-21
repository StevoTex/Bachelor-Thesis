import math
import random as rdm
from copy import copy

import numpy as np
from scipy.stats import qmc


def GEN_UNIFORM(n, m=1, seed=None):
    """
    Generates n random points in an m-dimensional space using a uniform distribution.
    :param n: Number of points.
    :type n: int
    :param m: Number of dimensions.
    :type m: int
    :param seed: The seed to use for the random number generator.
    :return: The points
    :rtype: list[list[float]]
    """

    rdm_original_state = rdm.getstate()
    if seed is not None:
        rdm.seed(seed)

    points = [[rdm.random() for _ in range(m)] for _ in range(n)]

    if seed is not None:
        rdm.setstate(rdm_original_state)

    return points


def GEN_HALTON(n, m=1, seed=None):
    """
    Generates n random points in [0,1]^m using a Halton sequence.
    :param n: Number of points.
    :type n: int
    :param m: Number of dimensions.
    :type m: int
    :param seed: The seed to use for the random number generator.
    :return: The points
    :rtype: list[list[float]]
    """
    halton = qmc.Halton(d=m, seed=seed)
    points = halton.random(n)
    return points.tolist()


def GEN_SOBOL(n, m=1, seed=None):
    """
    Generates a maximum of n points in [0,1]^m using the Sobol sequence. n should be a power of 2.
    :param n: Number of points.
    :type n: int
    :param m: Number of dimensions.
    :type m: int
    :param seed: The seed to use for the random number generator.
    :return: The points
    :rtype: list[list[float]]
    """

    # Return an empty list if n is 0
    if n == 0:
        return []

    # Calculate the number of points that can be generated with the Sobol sequence
    n_base2 = 2 ** math.floor(np.log2(n))

    # Generate and return the points
    sobol = qmc.Sobol(d=m, seed=seed)
    points = sobol.random_base2(m=int(np.log2(n_base2)))
    return points.tolist()


def GEN_SOBOL_HALTON(n, m=1, seed=None):
    """
    Generates n points in [0,1]^m using the Sobol sequence. The first points are generated using the Sobol sequence, the
    remaining points are generated using a Halton sequence.
    :param n: Number of points.
    :type n: int
    :param m: Number of dimensions.
    :type m: int
    :param seed: The seed to use for the random number generator.
    :return: The points
    :rtype: list[list[float]]
    """
    points_sobol = GEN_SOBOL(n, m, seed)
    points_halton = GEN_HALTON(n - len(points_sobol), m, seed)
    return points_sobol + points_halton


def GEN_GRID(n, m=1, seed=None):
    """
    Generates a maximum of n points in [0,1]^m using a grid that tries to maximize the number of points.
    :param n: Number of points.
    :type n: int
    :param m: Number of dimensions.
    :type m: int
    :param seed: The seed has no effect because the grid is deterministic.
    :return: The points
    :rtype: list[list[float]]
    """

    # calculate the number of points per dimension (log)
    points_per_dim = math.floor(n ** (1 / m))

    # create a grid and return the points
    grid = np.meshgrid(*[np.linspace(0, 1, points_per_dim) for _ in range(m)])
    return np.array(grid).reshape(m, -1).T.tolist()


def GEN_GRID_HALTON(n, m=1, seed=None):
    """
    Generates n points in [0,1]^m. The first points are generated using a grid, the remaining points are generated
    using the Halton sequence.
    :param n: Number of points.
    :type n: int
    :param m: Number of dimensions.
    :type m: int
    :param seed: The seed to use for the random number generator.
    :return: The points
    :rtype: list[list[float]]
    """
    points_grid = GEN_GRID(n, m, seed)
    points_sobol = GEN_HALTON(n - len(points_grid), m, seed)
    return points_grid + points_sobol


def EVAL_FITNESS(individuals):
    """
    Evaluates individuals based on their fitness.
    :param individuals: The individuals.
    :type individuals: list[Individual]
    :return: The fitness values for each individual.
    :rtype: list[float or int]
    """
    return [ind.get_fitness() for ind in individuals]


def EVAL_REL_FITNESS(individuals):
    """
    Evaluates individuals based on their relative fitness.
    :param individuals: The individuals.
    :type individuals: list[Individual]
    :return: The fitness values for each individual.
    :rtype: list[float or int]
    """
    return [ind.get_relative_fitness() for ind in individuals]


def EVAL_PARETO(individuals, **kwargs):
    """
    Evaluates individuals based on Pareto dominance. The value of an individual will be the number of individuals that
    it is not dominated by.
    :param individuals: The individuals.
    :type individuals: list[Individual]
    :return: The Pareto dominance ranks for each individual (attention: the interpretation of the values is inverted so
        that a higher value means a better individual).
    :rtype: list[int]
    """

    # get the goals
    goals = individuals[0].get_goals()

    # calculate the pareto ranks
    pareto_ranks = [0] * len(individuals)
    for i, ind_a in enumerate(individuals):
        for j, ind_b in enumerate(individuals):
            if i == j:
                continue
            phenotype_a = ind_a.get_phenotype()
            phenotype_b = ind_b.get_phenotype()
            b_gt_a = [(g > 0 and p_b > p_a) or (g < 0 and p_b < p_a) for g, p_a, p_b in
                      zip(goals, phenotype_a, phenotype_b)]
            b_gte_a = [(g > 0 and p_b >= p_a) or (g < 0 and p_b <= p_a) for g, p_a, p_b in
                       zip(goals, phenotype_a, phenotype_b)]
            if not (any(b_gt_a) and all(b_gte_a)):
                pareto_ranks[i] += 1

    # return the result
    return pareto_ranks


def SEL_BEST(individuals, n=1, pressure=1, eval_funct=EVAL_FITNESS):
    """
    Selects the best n individuals.
    :param individuals: The individuals.
    :type individuals: list[Individual]
    :param n: The number of individuals to select.
    :type n: int
    :param pressure: The pressure won't have any effect to this selection method.
    :type pressure: float or int
    """
    # calculate the values using the evaluation function
    values = eval_funct(individuals)

    # sort the individuals by their values in descending order
    sorted_individuals = [individuals[i] for i in np.argsort(values)[::-1]]

    # duplicate the first individual if necessary to ensure that the list is long enough
    if n > len(sorted_individuals):
        sorted_individuals = sorted_individuals + [sorted_individuals[0]] * (n - len(sorted_individuals))

    # return the best n individuals
    return sorted_individuals[:n]


def SEL_ROULETTE(individuals, n=1, pressure=1, eval_funct=EVAL_FITNESS):
    """
    Selects n individuals using roulette wheel selection.
    :param individuals: The individuals.
    :type individuals: list[Individual]
    :param n: The number of individuals to select.
    :type n: int
    :param pressure: The pressure to apply to the selection. Expected to be in range [0, 1]. A low pressure will result
        in a more uniform selection, a high pressure will result into the untouched roulette wheel selection.
    :type pressure: float or int
    :param eval_funct: The evaluation function to use. The evaluation function should take a list of individuals
        as input and return a list of floats or ints as output. The output values should be proportional to the
        probability of the corresponding individual being selected. If no evaluation function is given, the fitness
        values of the individuals will be used.
    :type eval_funct: function
    :return: The selected individuals.
    :rtype: list[Individual]
    """

    # raise an error if pressure is not in range [0, 1]
    if not 0 <= pressure <= 1:
        raise ValueError("pressure must be in range [0, 1].")

    # calculate the values using the evaluation function
    values = eval_funct(individuals)

    # set the probabilities by normalizing the values
    v_sum = sum(values)
    uniform_prob = 1 / len(individuals)
    if v_sum == 0:
        probabilities = [uniform_prob] * len(individuals)
    else:
        probabilities = [v / v_sum for v in values]

    # apply pressure
    probabilities = [pressure * p + (1 - pressure) * uniform_prob for p in probabilities]

    def _roulette_wheel_selection():
        # select an individual using roulette wheel selection
        r = rdm.random()
        cumulative_fitness = 0
        for ind, prob in zip(individuals, probabilities):
            cumulative_fitness += prob
            if r <= cumulative_fitness:
                return ind

        # return the last individual if no individual was selected (should never happen)
        return individuals[-1]

    # select the individuals using roulette wheel selection
    selected_individuals = []
    for _ in range(n):
        selected_individuals.append(_roulette_wheel_selection())

    # return the selected individuals
    return selected_individuals

def REC_CROSS_SBX(*parents, eta):
    """
    Recombines two parents using Simulated Binary Crossover (SBX),
    the standard crossover operator for NSGA-II with real-valued problems.

    :param parents: A tuple that must contain exactly two parents.
    :type parents: tuple(Individual)
    :param eta: The distribution index for crossover. A high value (e.g., 15-20)
                creates offspring similar to the parents. A low value (e.g., 2-5)
                allows for more distant offspring.
    :type eta: float
    :return: A single offspring as an Individual object.
             Note: SBX produces two offspring; one is chosen randomly to fit
             the function style of returning a single individual.
    :rtype: Individual
    """
    # --- 1. Validate inputs ---
    # SBX is defined for exactly two parents.
    if len(parents) != 2:
        raise ValueError("Simulated Binary Crossover (SBX) requires exactly two parents.")

    # Extract parents and their genotypes.
    parent1, parent2 = parents
    geno1 = parent1.get_genotype()
    geno2 = parent2.get_genotype()

    # Genotypes must be of equal length.
    if len(geno1) != len(geno2):
        raise ValueError("All genotypes must be of equal length.")

    # SBX performs arithmetic operations, so genes must be numbers.
    if not all(isinstance(g, (int, float)) for g in geno1 + geno2):
        raise TypeError("SBX only works with numeric (int or float) genotypes.")

    # --- 2. Apply SBX logic for each gene ---
    offspring1_geno = []
    offspring2_geno = []

    # Iterate over each pair of genes from the parents.
    for g1, g2 in zip(geno1, geno2):
        # Generate a random number u in [0, 1].
        u = np.random.random()

        # Calculate the spread factor 'beta'.
        if u <= 0.5:
            beta = (2 * u)**(1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u)))**(1 / (eta + 1))

        # Calculate the new genes for the two offspring.
        c1 = 0.5 * ((1 + beta) * g1 + (1 - beta) * g2)
        c2 = 0.5 * ((1 - beta) * g1 + (1 + beta) * g2)

        # Append the new genes to the offspring's genotypes.
        offspring1_geno.append(c1)
        offspring2_geno.append(c2)

    # --- 3. Create and return the final offspring object ---
    # Get the class of the parents to create an offspring of the same type.
    OffspringClass = parents[0].__class__

    # Create the two new Individual objects.
    offspring1 = OffspringClass(*offspring1_geno)
    offspring2 = OffspringClass(*offspring2_geno)

    # To match the style of returning a single offspring, randomly choose one.
    return np.random.choice([offspring1, offspring2])



def REC_CROSS_UNIFORM(*parents):
    """
    Recombines the given parents using a uniform crossover.
    :param parents: The parents.
    :type parents: tuple(Individual)
    :return: The offspring as an Individual.
    :rtype: Individual
    """

    # randomize the order of the parents
    parents = np.random.permutation(parents)

    # get the genotypes of the parents
    genotypes = [parent.get_genotype() for parent in parents]

    # check if genotypes are of equal length
    gen_length = len(genotypes[0])
    if not all([len(genotype) == gen_length for genotype in genotypes]):
        raise ValueError("All genotypes must be of equal length.")

    # recombine the genotypes
    offspring_genotype = []
    for j in range(gen_length):
        parent = np.random.choice(parents)  # Choose a random parent
        offspring_genotype.append(copy(parent.get_genotype()[j]))  # Append the allele to the offspring genotype

    # return the offspring
    return parents[0].__class__(*offspring_genotype)


def REC_CROSS_POINT(*parents, **kwargs):
    """
    Recombines the given parents using an n point crossover. The default number of crossover points will be calculated
    so that, if possible, each parent will be used once. The number of crossover points can also be set manually using
    the keyword argument 'n'.
    :param parents: The parents.
    :type parents: tuple(Individual)
    :return: The offspring.
    :return: The offspring as tuple of Individuals.
    :rtype: Individual
    """

    # randomize the order of the parents
    parents = np.random.permutation(parents)

    # get the genotypes of the parents
    genotypes = [parent.get_genotype() for parent in parents]

    # get the length of the genotypes and check if they are of equal length
    gen_length = len(genotypes[0])
    if not all([len(genotype) == gen_length for genotype in genotypes]):
        raise ValueError("All genotypes must be of equal length.")

    # determine the number of crossover points
    n = kwargs.get('n', min(gen_length, len(parents)) - 1)

    # generate the crossover points
    crossover_points = list(np.random.choice(gen_length, n, replace=False))

    # recombine the genotypes
    offspring_genotype = []
    i = 0
    for j in range(gen_length):
        # switch to the next parent if the crossover point is reached
        if j in crossover_points:
            i = (i + 1) % len(parents)

        # append the allele to the offspring genotype
        offspring_genotype.append(copy(genotypes[i][j]))

    # return the offspring
    return parents[0].__class__(*offspring_genotype)


def REC_CROSS_ARITHMETIC(*parents):
    """
    Recombines the given parents using an arithmetic crossover.
    :param parents: The parents.
    :type parents: tuple(Individual)
    :return: The offspring as an Individual.
    :rtype: Individual
    """

    # Get the genotypes of the parents
    genotypes = [parent.get_genotype() for parent in parents]

    # Check if genotypes are of equal length
    gen_length = len(genotypes[0])
    if not all([len(genotype) == gen_length for genotype in genotypes]):
        raise ValueError("All genotypes must be of equal length.")

    # Recombine the genotypes using arithmetic crossover
    offspring_genotype = []
    weight = 1 / len(parents)
    for i in range(gen_length):
        avg = 0
        for j in range(len(parents)):
            avg += weight * genotypes[j][i].get()
        calculated_allele = genotypes[0][i].__class__(avg)  # create a new allele with the calculated value
        offspring_genotype.append(calculated_allele)  # append the allele to the offspring genotype

    # Return the offspring
    return parents[0].__class__(*offspring_genotype)


def MUT_FLIP(allele, heat=.5):
    """
    Flips the given allele.
    :param allele: The allele.
    :type allele: Allele
    :param heat: The likelihood of flipping the allele.
    :type heat: float
    """

    # raise an error if heat is not in range [0, 1]
    if not 0 <= heat <= 1:
        raise ValueError("heat must be in range [0, 1].")

    if rdm.random() < heat:
        allele.set((-allele).get())


def MUT_UNIFORM(allele, heat=.5):
    """
    Mutates the given allele using a uniform distribution.
    :param allele: The allele.
    :type allele: Allele
    :param heat: The normalized range of the uniform distribution around the old value.
    :type heat: float or int
    """

    # raise an error if heat is not in range [0, 1]
    if not 0 <= heat <= 1:
        raise ValueError("heat must be in range [0, 1].")

    # calculate the new value
    old = allele.get(normalized=True)
    min_value = max(old - heat / 2, 0)
    max_value = min(old + heat / 2, 1)
    new = rdm.uniform(min_value, max_value)

    # set the new value
    allele.set(new, normalized=True)


def MUT_GAUSS(allele, heat=.5):
    """
    Mutates the given allele using a Gaussian distribution.
    :param allele: The allele.
    :type allele: Allele
    :param heat: The normalized mutation range (approx. 99.7% of the values will be in the range
        [old - heat/2, old + heat/2]).
    :type heat: float or int
    """

    # raise an error if heat is not in range [0, 1]
    if not 0 <= heat <= 1:
        raise ValueError("heat must be in range [0, 1].")

    # calculate the new value
    old = allele.get(normalized=True)
    min_value = old - heat / 2
    max_value = min_value + heat
    new = -1
    while not 0 <= new <= 1:
        new = rdm.gauss(old, (max_value - min_value) / 6)

    # set the new value
    allele.set(new, normalized=True)


def MUT_POLYNOMIAL(individual, pmut, eta, low, high):
    """
    Performs a Polynomial Mutation (PM) on an individual.
    This is the standard mutation operator for NSGA-II with real-valued problems.

    :param individual: The individual to be mutated.
    :type individual: Individual
    :param pmut: The probability of mutating a single gene.
    :type pmut: float
    :param eta: The distribution index for mutation. A high value favors small
                changes, a low value allows for larger changes.
    :type eta: float
    :param low: The lower bound for the gene values. Can be a single number or a
                list/array of the same length as the genotype.
    :type low: float or list or np.ndarray
    :param high: The upper bound for the gene values. Can be a single number or a
                 list/array of the same length as the genotype.
    :type high: float or list or np.ndarray
    :return: A new, mutated individual.
    :rtype: Individual
    """
    # Use deepcopy to ensure the original individual is not modified.
    mutated_genotype = deepcopy(individual.get_genotype())
    gen_length = len(mutated_genotype)

    # Convert bounds to numpy arrays for vectorized operations.
    if not isinstance(low, np.ndarray):
        low = np.full(gen_length, low)
    if not isinstance(high, np.ndarray):
        high = np.full(gen_length, high)

    # Apply mutation to each gene based on the probability pmut.
    for i in range(gen_length):
        if np.random.random() <= pmut:
            # Current gene value and its bounds.
            y = mutated_genotype[i]
            yl = low[i]
            yu = high[i]

            # Calculate the perturbation 'delta'.
            u = np.random.random()

            if u <= 0.5:
                # Calculate delta for the lower part of the distribution.
                delta_l = (y - yl) / (yu - yl) if (yu - yl) > 1e-9 else 0.0
                val = 2 * u + (1 - 2 * u) * ((1 - delta_l) ** (eta + 1))
                delta_q = val ** (1 / (eta + 1)) - 1
            else:
                # Calculate delta for the upper part of the distribution.
                delta_u = (yu - y) / (yu - yl) if (yu - yl) > 1e-9 else 0.0
                val = 2 * (1 - u) + 2 * (u - 0.5) * ((1 - delta_u) ** (eta + 1))
                delta_q = 1 - val ** (1 / (eta + 1))

            # Apply the mutation and clamp the value to the bounds.
            y = y + delta_q * (yu - yl)
            y = np.clip(y, yl, yu)
            mutated_genotype[i] = y

    # Return a new individual with the mutated genotype.
    return individual.__class__(*mutated_genotype)
