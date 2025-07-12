import pandas as pd
import numpy as np

import random
from typing import Optional

from src.bias.stbc import STBC

# Steady state genetic algorithm
class SSGA:
    def __init__(
        self,
        df: pd.DataFrame,
        pop_size: int,
        num_generations: int,
        mutation_rate: float,
        crossover_rate: float,
        tournament_size: int,
        random_seed: Optional[int],
    ):
        self.df = df.copy()
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        if random_seed is not None:
            np.random.seed(random_seed)

    def _initialize_population(self):
        population = []
        
        for _ in range(self.pop_size):
            individual = np.random.random()  # Generate a random float between 0 and 1
            population.append(individual)
            
        return population

    def _fitness(self, df: pd.DataFrame, individual: int):
        stbc = STBC(df, individual)
        stbc.calibrate()
        return 1 / stbc.evaluate_accuracy()
    
    def _selection(self, population: list[int], fitness_scores: list[float]):
        # randomly get the indexes of population
        tournament = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]

        winner = tournament[tournament_fitness.index(max(tournament_fitness))]
        return population[winner]
    
    def _crossover(self, parent1: float, parent2: float):
        # One-point crossover implementation for floats
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # For floats, we can do a weighted average (linear interpolation)
        # Choose a random weight between 0 and 1
        alpha = random.random()
        
        # Create two children using complementary weights
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        
        # Ensure the values stay in the [0,1] range
        child1 = max(0.0, min(1.0, child1))
        child2 = max(0.0, min(1.0, child2))
        
        return child1, child2
        
    def _mutate(self, individual: int):
        mutated = individual

        if random.random() < self.mutation_rate:
            # Randomly flip a bit in the integer representation
            bit_to_flip = random.randint(0, 31)
            mutated ^= (1 << bit_to_flip)  # Flip the bit at the position

        return mutated
    
    def __call__(self):
        population = self._initialize_population()
        best_individual = None
        best_fitness = float('-inf')

        fitness_scores = [self._fitness(self.df, individual) for individual in population]

        for _ in range(self.num_generations):
            parent1 = self._selection(population, fitness_scores)
            parent2 = self._selection(population, fitness_scores)

            child1, child2 = self._crossover(parent1, parent2)

            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            child1_fitness = self._fitness(self.df, child1)
            child2_fitness = self._fitness(self.df, child2)

            if child1_fitness > min(fitness_scores):
                index_to_replace = fitness_scores.index(min(fitness_scores))
                population[index_to_replace] = child1
                fitness_scores[index_to_replace] = child1_fitness
            if child2_fitness > min(fitness_scores):
                index_to_replace = fitness_scores.index(min(fitness_scores))
                population[index_to_replace] = child2
                fitness_scores[index_to_replace] = child2_fitness

            if max(fitness_scores) > best_fitness:
                best_fitness = max(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]

        return best_individual, best_fitness