"""
In this script, we compare the diversity of populations generated using different generation functions to get a better
understanding of how they work. We use a sample individual with two alleles for an easy visualization of the diversity
in 2D.
"""
import evolution as evo


# Create a sample allele class with values between 0 and 10
class SampleAllele(evo.Allele):
    _min = 0
    _max = 10


# Create a sample individual class with two alleles
class SampleIndividual(evo.Individual):
    _Blueprint = {
        'genotype': [SampleAllele, SampleAllele],
        'genotype_labels': ['Allele 1', 'Allele 2'],
        'goals': [1, 1],
        'phenotype_labels': ['Phenotype 1', 'Phenotype 2'],
    }

    # We need to implement the _calculate_phenotype method, but it doesn't need to do anything useful in this case
    def _calculate_phenotype(self):
        allele1, allele2 = self.get_genotype()
        return allele1.get(), allele2.get()


if __name__ == '__main__':
    # Set the seed and population size
    seed = 42
    pop_size = 64

    # Create a population using the uniform generation function, plot it and print its diversity
    uniform_pop = evo.Population(SampleIndividual, pop_size, gen_funct=evo.GEN_UNIFORM, seed=seed)
    uniform_pop.plot(name="uniform", directory="results/02-comparing-generation-functions-2d")
    print(f"GEN_UNIFORM diversity: {uniform_pop.get_diversity()}")

    # Create a population using the halton generation function, plot it and print its diversity
    halton_pop = evo.Population(SampleIndividual, pop_size, gen_funct=evo.GEN_HALTON, seed=seed)
    halton_pop.plot(name="halton", directory="results/02-comparing-generation-functions-2d")
    print(f"GEN_HALTON diversity: {halton_pop.get_diversity()}")

    # Create a population using the sobol halton generation function, plot it and print its diversity
    sobol_halton_pop = evo.Population(SampleIndividual, pop_size, gen_funct=evo.GEN_SOBOL, seed=seed)
    sobol_halton_pop.plot(name="sobol", directory="results/02-comparing-generation-functions-2d")
    print(f"GEN_SOBOL diversity: {sobol_halton_pop.get_diversity()}")

    # Create a population using the grid halton generation function, plot it and print its diversity
    grid_halton_pop = evo.Population(SampleIndividual, pop_size, gen_funct=evo.GEN_GRID, seed=seed)
    grid_halton_pop.plot(name="grid", directory="results/02-comparing-generation-functions-2d")
    print(f"GEN_GRID diversity: {grid_halton_pop.get_diversity()}")
