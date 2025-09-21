"""
This module contains the strategies for the evolution of the population. You can create your own strategies by defining
a function that takes the population, the current generation and the total number of generations as arguments and
returns a new population. STRAT_A is the default strategy that is used if no strategy is specified.
"""
from .operators import *


def STRAT_A(population, i, n):
    aep = (i + 1) / (n + 1)  # assumed exploration progress
    return population.next_generation(
        aep=aep,
        eval_funct=EVAL_FITNESS,
        recombination_funct=REC_CROSS_ARITHMETIC,
    )


def STRAT_B(population, i, n):
    aep = (i + 1) / (n + 1)  # assumed exploration progress
    return population.next_generation(
        aep=aep,
        eval_funct=EVAL_PARETO,
        recombination_funct=REC_CROSS_POINT,
    )

# def STRAT_C(population, i, n)
#     aep = (i + 1) / (n + 1)
#     return population.next_generation(
#         eval_funct=EVAL_PARETO,
#         eval_funct =
#     )
