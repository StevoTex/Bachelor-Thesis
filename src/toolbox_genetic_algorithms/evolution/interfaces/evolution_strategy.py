from abc import ABC, abstractmethod
from typing import Type

class EvolutionStrategy(ABC):
    """Abstract base class for an evolutionary strategy.

    This class provides a template for evolutionary algorithms. It defines the
    main `evolve` loop and requires subclasses to implement the specific
    algorithmic steps.

    Attributes:
        population (Population): The current population of individuals.
        population_size (int): The target size of the population.
    """

    def __init__(self, population: Type['Population'], **kwargs: dict):
        """Initializes the EvolutionStrategy.

        Args:
            population (Population): The initial population of individuals.
            **kwargs: A dictionary of keyword arguments for configuration.
                      Expected keys include 'population_size'.
        """
        self.population: Type['Population'] = population
        self.population_size: int = kwargs.get('population_size', len(population))
        print(f"Initializing '{self.__class__.__name__}' with a population of {len(self.population)}.")

    @abstractmethod
    def _selection(self) -> Type['Population']:
        """Selects a group of parent individuals from the current population.

        This method must be implemented by a subclass.

        Returns:
            A population object containing the selected parents.
        """
        pass

    @abstractmethod
    def _recombination(self, parents: Type['Population']) -> Type['Population']:
        """Creates a new population of offspring from the selected parents.

        This method must be implemented by a subclass.

        Args:
            parents: The population of parents selected for breeding.

        Returns:
            A new population of offspring individuals.
        """
        pass

    @abstractmethod
    def _mutation(self, offspring: Type['Population']) -> Type['Population']:
        """Applies mutation to the offspring population.

        This method must be implemented by a subclass.

        Args:
            offspring: The population of offspring to be mutated.

        Returns:
            The mutated population of offspring.
        """
        pass

    @abstractmethod
    def _create_new_generation(self, offspring: Type['Population']) -> Type['Population']:
        """Forms the next generation from the current population and offspring.

        This method defines the survival strategy (e.g., elitism) and is
        responsible for maintaining the population size. It must be
        implemented by a subclass.

        Args:
            offspring: The newly created and mutated offspring.

        Returns:
            The population for the next generation.
        """
        pass

    def evolve(self, generations: int) -> 'Population':
        """Executes the main evolutionary loop for a given number of generations.

        This method orchestrates the evolutionary process by calling the
        abstract methods in the correct sequence for each generation.

        Args:
            generations: The number of generations to run the evolution.

        Returns:
            The final population after the evolution has completed.
        """
        print(f"Starting evolution for {generations} generations...")
        for gen in range(generations):
            # Step 1: Select parents for breeding
            parents = self._selection()

            # Step 2: Create new offspring through recombination (crossover)
            offspring = self._recombination(parents)

            # Step 3: Apply mutation to the new offspring
            offspring = self._mutation(offspring)

            # Step 4: Create the next generation's population based on a survival strategy
            self.population = self._create_new_generation(offspring)

            print(f"Generation {gen + 1}/{generations} complete.")

        print("Evolution finished.")
        return self.population