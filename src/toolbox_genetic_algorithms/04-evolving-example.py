"""
In this example, we will have a look at the next generation function. We will create an initial population of cars and
demonstrate the next_generation method and the evolve method using a custom strategy.
"""
import evolution as evo
from car import Car


if __name__ == '__main__':
    # create the initial population (first generation)
    initial_population = evo.Population(Car, 10, seed=42)

    # create the second generation
    second_generation = initial_population.next_generation(
        aep=.2,
        eval_funct=evo.EVAL_FITNESS,
        recombination_funct=evo.REC_CROSS_ARITHMETIC,
        elite=2,
        # ... other parameters can be passed here
    )

    # define a custom strategy
    def STRAT_CUSTOM(population, i, n):
        aep = (i + 1) / (n + 1)  # assumed exploration progress (simulated annealing)
        return population.next_generation(
            aep=aep,
            eval_funct=evo.EVAL_FITNESS,  # use the fitness function for evaluation
            recombination_funct=evo.REC_CROSS_ARITHMETIC,  # use the arithmetic crossover recombination
            elite=2,  # keep 2 elite individuals
        )


    # evolve the initial population for 10 generations using the custom strategy
    generations = initial_population.evolve(10, STRAT_CUSTOM)
