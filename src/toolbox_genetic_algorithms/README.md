# Evolutionary Optimization of Vehicles

This project deals with the implementation and comparison of different genetic algorithms for vehicle development. The
aim is to gain a deep understanding of the functionality and effectiveness of different approaches to genetic algorithms
and to compare them in different scenarios.

## Usage
In this section, I will concentrate on how to use the Genetic Algorithms package ([evolution/](evolution/__init__.py))
to create and optimize new individuals. This package is the core of the project and is meant to be used as a basis for
new projects. It contains many different genetic operators that can be used to create and optimize many different
types of individuals, which should have in common that they can be represented by a real-coded genotype.

### 1. Import the Package
```python
import evolution as evo
```

### 2. Creating Allele Classes
Alleles are the building blocks of individuals. They are used to represent the different parameters of the individuals.
The [Allele](evolution/interfaces/allele.py) class is an abstract class that is extended by specific Alleles, such as
'FinalDrive' which represents the final drive ratio of a vehicle.

**Example:**
In this example, we create three Alleles, which are meant to represent the final drive ratio, the roll radius, and the 
gear ratios of a vehicle. The Alleles are created by extending the `Allele` class and defining the `_min` and `_max`
attributes. These attributes are used to define the range of the Alleles values.
```python
class FinalDrive(evo.Allele):
    _min = 1
    _max = 6

class RollRadius(evo.Allele):
    _min = 0.2
    _max = 0.5

class Gear(evo.Allele):
    _min = 0.5
    _max = 2
```

### 3. Creating Individual Classes
The [Individual](evolution/interfaces/individual.py) class is an abstract class, which is meant to be extended by
specific individuals.

Individuals are the key component of genetic algorithms. Each individual represents a possible solution to the problem
at hand. The first thing to do when optimizing a new problem is to create a new individual class, which also includes the
goals of the optimization problem.

Each Individual class has to extend the `Individual` class and has to define these attributes and methods:
1. **_Blueprint**: This attribute has to be a dictionary, which defines the structure of the individual. It has to contain 
the following keys:
    - `'genotype'`: A list of Allele classes, which define the structure of the genotype of the individual.
    - `'genotype_labels'`: A list of strings that define the labels of the Alleles in the genotype.
    - `'goals'`: A list of floats or ints that define the weighted goals of the optimization. Negative values indicate
    that the goal has to be minimized, while positive values indicate that the goal has to be maximized. The higher the
    absolute value of the goal, the more important it is.
    - `'phenotype_labels'`: A list of strings that define the labels of the objectives in the phenotype.
2. **_calculate_phenotype(self)**: This method has to return the phenotype of the individual. The phenotype is a tuple
of numerical values that represent the objectives of the individual. The number of objectives has to be equal to the
number of goals defined in the blueprint.
3. (Optional) **_enforce_constraints(self)**: This method can be used to enforce constraints on the genotype of the
individual. It is called after the genotype is created and before the phenotype is calculated.

**Example:**
In this example, we create a new individual class, which represents a vehicle. For the genotype, we are using the
Alleles we created in the previous step.

```python
class Car(evo.Individual):
    _Blueprint = {
        'genotype': [FinalDrive, RollRadius, Gear, Gear, Gear],
        'genotype_labels': ['Final Drive', 'Roll Radius', 'Gear 3', 'Gear 4', 'Gear 5'],
        'goals': [-4, -2, -1, -1],
        'phenotype_labels': ['Consumption', 'Elasticity 3', 'Elasticity 4', 'Elasticity 5'],
    }

    def _calculate_phenotype(self):
        # extract the alleles from the genotype
        final_drive_ratio, roll_radius, gear3, gear4, gear5 = self.get_genotype()

        # run the simulation
        consumption, ela_3, ela_4, ela_5 = run_simulation(final_drive_ratio, roll_radius, gear3, gear4, gear5)

        # return the phenotype
        return consumption, ela_3, ela_4, ela_5

    def _enforce_constraints(self):
        # putting the gears in the correct order
        final_drive_ratio, roll_radius, gear3, gear4, gear5 = self.get_genotype()
        gear3, gear4, gear5 = sorted([gear3, gear4, gear5], reverse=True)
        self._genotype = final_drive_ratio, roll_radius, gear3, gear4, gear5
```

In this code the goals are defined as follows:
- The first priority goal is to minimize the consumption of the vehicle.
- The second priority goal is to maximize the elasticity of the third gear.
- The third priority goal is to maximize the elasticity of the fourth and fifth gear.

In the `_calculate_phenotype` method, a simulation is used to calculate the objectives of the individual. You can use
any method you want to calculate the objectives, as long as it returns a tuple of numerical values that has the same
length as the number of goals defined in the blueprint.

The `_enforce_constraints` method is used to enforce constraints on the genotype of the individual. In this case, we are
using it to make sure that the gears are in the correct order.

### 4. Creating Instances of Individuals and Populations
Now that we have created a new individual class, we can create an instance of it - a specific vehicle in this case.

**Example 1:** Creating a random vehicle.
```python
car1 = Car()
car2 = Car()
```

**Example 2:** Creating 100 vehicles using a population.
```python
population = evo.Population(Car, 100)
```
Using a population is the preferred way of creating multiple individuals because it uses smart mechanisms to create a
diverse population of individuals.

### 5. Evolving Populations
Now that we have created a population of individuals, you can start evolving them. This is done by calling the `evolve`
method of the population.

**Example 1:** Evolving the population for 1 generation.
```python
generation1 = population.next_generation()
```

**Example 2:** Evolving the population for 10 generations.
```python
generations = population.evolve(n=10)
```
This function will return a list of populations, where each population represents a generation. The first population in
the list is the initial population, while the last population is the final population.

### 6. Getting the Results
After evolving the population, you can get access to the single individuals and their objectives as follows:
```python
# get the last generation
last_generation = generations[-1]

# extract the first individual of the last generation
best_car = last_generation.get_individuals(order_by='fitness', reverse=True)[0]

# access the genotype of the individual as numerical values
final_drive_ratio, roll_radius, gear3, gear4, gear5 = best_car.get_genotype(transform=True)

# access the objectives of the individual as numerical values
consumption, ela_3, ela_4, ela_5 = best_car.get_phenotype()

# access the fitness of the individual in relation to all other individuals
fitness = best_car.get_relative_fitness()
```

The `transform` flag of the `get_genotype` method can be used to get the values of the Alleles instead of the Alleles
themselves.

We are using the `get_relative_fitness` method to access the normalized fitness of the individual. This value is trying 
to evaluate the fitness of the individual in relation to all other individuals in the population. It is calculated based
on the optimization goals defined in the blueprint of the individual.

You can also use the `plot` method of the population to get a scatter plot of the genotype of the individuals.
```python
population.plot(name="test")
```

### 7. Using specific Genetic Operators and Creating new ones
The [operators.py](evolution/operators.py) file contains many different genetic operators, which can be used to create
and optimize many different types of individuals. These operators can be classified into five categories, which I will
describe in the following sections, before I will show you how to use and combine them to create new genetic mechanisms.

#### Category 1: Generation Operators
These operators are supposed to create n points in [0, 1]^m. They are used to create the initial population of a genetic
algorithm. To make distributions reproducible, each generation operator has a `seed` parameter, which can be used to
set the seed of the random number generator.

As an example, the following function is part of the [operators.py](evolution/operators.py) file. It generates n random
points in an m-dimensional space using a uniform distribution.
```python
import random as rdm

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
```

If you want to change the way the initial population is created, you can simply pass a different generation operator to
the `Population` class as follows:
```python
# create a population of 100 individuals using the GEN_UNIFORM operator
population = evo.Population(Car, 100, generation_operator=GEN_UNIFORM)
```

The `GEN_GRID_HALTON` operator is used by default because it turned out to be the most effective operator in my tests.
This operator generates points in a grid and fills the remaining points with Halton sequences.

#### Category 2: Evaluation Operators
These operators are supposed to evaluate the individuals in a population. This should result in a list of numerical
values that represent the quality of the individuals.

A very basic evaluation operator is fitness based evaluation. It simply returns the fitness of each individual.
```python
def EVAL_FITNESS(individuals):
    """
    Evaluates individuals based on their fitness.
    :param individuals: The individuals.
    :type individuals: list[Individual]
    :return: The fitness values for each individual.
    :rtype: list[float or int]
    """
    return [ind.get_fitness() for ind in individuals]
```

You may be right to criticize that the whole `get_fitness` method should not be part of the individual and that it
should better be part of the evaluation operator in order to be consistent. But my implementation has the advantage of
making some things faster because of the use of a caching system in the [Individual](evolution/interfaces/individual.py)
class.

Evaluation operators are used by selection operators to evaluate individuals and make decisions based on the results. To
tell the population to use a specific evaluation operator, you can parse it to the `next_generation` method using the
`eval_funct` parameter. This parameter is optional and defaults to the `EVAL_PARETO` function.
```python
# create a new generation using the EVAL_FITNESS operator
next_generation = population.next_generation(eval_funct=EVAL_FITNESS)
```

#### Category 3: Selection Operators
These operators are used to select individuals from a population. They are meant to use a
given evaluation operator to evaluate the individuals and then make decisions based on the results.

The `SEL_BEST` operator may be the most basic selection operator. It simply returns the best n individuals in the
population. But attention: It may be more effective to select very good individuals multiple times. To make this clear,
think about what it means to select the best n of n individuals.
```python
import numpy as np

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
```

The `pressure` parameter can be used to add the ability to adjust the selection pressure. To see how this works, I
recommend taking a look at the `SEL_ROULETTE` function in the [operators.py](evolution/operators.py) file. This function
is also used by default.

#### Category 4: Recombination Operators
These operators are used to create new individuals from more than one existing individuals.

The `REC_CROSS_UNIFORM` a recombination function which decides for each allele of the offspring randomly from which
parent it should inherit the allele.

```python
import numpy as np
from copy import copy

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
```

The default is to use the `REC_CROSS_ARITHMETIC` operator, which is a simple arithmetic crossover.

#### Category 5: Mutation Operators
These operators are used to modify existing individuals by defining how to mutate a single allele.

As an example, we can take a look at the `MUT_GAUSS` operator, which is used to mutate the alleles of an individual
with a Gaussian distribution.

```python
import random as rdm

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
```

Mutation operators functions should feature a `heat` parameter, which defines how much of the allele should be mutated.
The `heat` parameter is normalized, which means that it is in the range [0, 1]. This functionality is used by the
`next_generation` method of the [Population](evolution/interfaces/population.py) class to adjust the mutation rate based
on various factors.

The `MUT_GAUSS` operator is used by default.

#### Using Operators to Customize Genetic Algorithms
You can add and use custom operators as long as they follow the same interface as the existing operators. The simplest
way to customize the genetic algorithms is to create a new evolution function, which is a combination of the existing
operators and can be parsed to the `evolve` method of the [Population](evolution/interfaces/population.py) class.

Here is an example of a custom evolution function:
```python
# defining a custom evolution function
def STRAT_CUSTOM(population, i, n):
    
    # trying to estimate the exploration progress
    aep = (i + 1) / (n + 1)  # aep = assumed exploration progress
    
    return population.next_generation(
        aep=aep,
        eval_funct=evo.EVAL_PARETO,
        recombination_funct=evo.REC_CROSS_ARITHMETIC,
        elite=5,
    )

# calling the evolve method of the population
generations = population.evolve(n=10, evolution_funct=STRAT_CUSTOM)
```

Possible parameters for the `next_generation` method are:
- `aep`: The assumed exploration progress. This parameter is used to adjust the mutation rate and is expected to be in
the range [0, 1]. The higher the value, the smaller will each mutation be.
- `recombination_rate`: Defines the probability of recombination of individuals after the selection. This parameter is
expected to be in the range [0, 1]. The higher the value, the more recombination will be applied.
- `parent_group_size`: Defines the number of parents used for recombination. This parameter is expected to be an integer
greater or equal to 1. In most cases, it is recommended to use a value of 2.
- `population_size`: Defines the size of the population after the selection. This parameter is expected to be an integer
greater or equal to 1. The default is to keep the population size constant.
- `sel_funct`: The selection operator.
- `eval_funct`: The evaluation operator.
- `recombination_funct`: The recombination operator.
- `mutation_funct`: The mutation operator.
- `duplicate_free`: Defines if the population should be duplicate-free after selection, recombination and mutation. This
parameter is expected to be a boolean value. The default is `True`.
- `elite`: Defines the number of elite individuals. This parameter is expected to be an integer greater or equal to 0.
Elite means that the best individuals in the population will be copied to the next generation without any changes. The
value is set to `parent_group_size` by default.
- `alien`: Defines the number of alien individuals. This parameter is expected to be an integer greater or equal to 0.
Alien means to add some randomly generated individuals to the next generation. Default is 0.

More examples of evolution functions can be found in the [evolving.py](evolution/strategies.py) file.

#### Building a completely new Genetic Algorithm
For making more significant changes to the genetic mechanisms, you can simply extend the Population class and overwrite
the `next_generation` and `evolve` methods.

## Project Structure

The project is divided into several main components, described as follows:

```
project_directory/
|
|-- evolution/
|   |-- __init__.py
|   |-- benchmark.py
|   |-- metrics.py
|   |-- operators.py
|   |-- strategies.py
|   |
|   |-- interfaces/
|       |-- __init__.py
|       |-- allele.py
|       |-- population.py
|       |-- individual.py
|
|-- results/
|   |-- ... (the results of the experiments)
|
|-- utilities/
|   |-- __init__.py
|   |-- consumption_model.py
|   |-- ConsumptionCar.exe
|   |-- helpers.py
|
|-- ... (the codes for the bachelor thesis)
|
|-- car.py
|-- README.md
|-- requirements.txt
```

### Root Directory

- `README.md`: This file, providing an overview of the project and its structure.
- `requirements.txt`: Lists all external Python libraries required for the project.

### Genetic Algorithms Package ([evolution/](evolution/__init__.py))
This is the real core of the project, containing the implementation of the different genetic algorithms. There are many
variations of genetic mechanisms, such as different selection, recombination and mutation operators.

#### Interfaces ([evolution/interfaces/](evolution/interfaces/))
- `allele.py`: Defines the abstract `Allele` class, serving as a basis for specific real coded alleles.
- `population.py`: Contains the `Population` class, which manages a population of individuals.
- `individual.py`: Defines the abstract `Individual` class, serving as a basis for specific individuals.

#### Operators ([evolution/operators.py](evolution/operators.py))
This file contains the implementations of the different genetic operators used in the genetic algorithms. These include
Selection, Recombination and Mutation operators, as well as distribution functions and metrics.

#### Strategies ([evolution/strategies.py](evolution/strategies.py))
This file contains the implementations of different evolution functions that can be used to evolve populations.

#### Benchmark ([evolution/benchmark.py](evolution/benchmark.py))
This file contains a benchmark class, which can be used to compare different genetic algorithms.

#### Metrics ([evolution/metrics.py](evolution/metrics.py))
This file contains the implementations of different metrics like the average of n values. These metrics, are used as
utility functions in the genetic algorithms. This is kind of a mess right now and should be ignored.

### Car Class ([car.py](car.py))
This file contains the implementation of the `Car` class, which is used to represent a vehicle in the project. It
extends the `Individual` class from the `evolution/` package.

### Utilities Package ([utilities/](utilities/__init__.py))
This package contains all the utility functions used to run Tests and Experiments in the specific use case of this
project, which is the optimization of vehicles. It also contains the executable of the consumption model.
- `consumption_model.py`: Contains the `run_simulation` function, which is used to calculate the objectives of our
    vehicles.
- `ConsumptionCar.exe`: The executable of the consumption model, used by `run_simulation`.
- `experiments.py`: Contains the functions to run and evaluate the experiments (this is kind of a mess right now).

### Results Directory ([results/](results/))
This directory contains the results of the scrips for the bachelor thesis.

## License

### License for the Utilities Package
The Utilities package of this project contains the executable of the consumption model, for which all rights are 
reserved. The use of the executable is not permitted without the express permission of the copyright holder. For 
inquiries regarding the use or for questions about licensing terms, please contact me at 
[marius@kleinformat.net](mailto:marius@kleinformat.net).

### License for the Rest of the Project
The remainder of this project is released under the MIT License. Below is the full text of the license:

**MIT License**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.