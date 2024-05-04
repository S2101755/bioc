import random
import matplotlib.pyplot as plt
from statistics import mean
import csv

class Rule:
    def __init__(self, condition, output):
        self.condition = condition
        self.output = output

class Individual:
    def __init__(self, gene=None):
        self.gene = gene if gene else self.randomiseGene()
        self.fitness = 0

    def randomiseGene(self):
        return [random.randint(0, 1) for _ in range(N)]

    def fitnessFunction(self, rulebase):
        count = sum(any(all(gene[i] == r.condition[i] for i in range(condLength)) and gene[-1] == r.output[0] for r in rulebase) for gene in self.sliceGene())
        self.fitness = count

    def sliceGene(self):
        return [self.gene[i:i + (condLength + outLength)] for i in range(0, N, condLength + outLength)]

def loadDataset(dataloc):
    rds = []
    with open(dataloc, "r") as f:
        for line in f:
            line = line.strip().split(",")
            if len(line) == 2:
                condition = [int(bit) for bit in line[0]]
                output = [int(line[1])]  # Store output as a list
                rule = Rule(condition, output)  # Create a Rule object
                rds.append(rule)  # Append the Rule object to the list
            else:
                print("Skipping invalid line:", line)
    return rds

def createInitialPopulation():
    return [Individual() for _ in range(P)]

def selection(pop):
    return [max(random.sample(pop, k=tournament_size), key=lambda x: x.fitness) for _ in range(P)]

def crossover(parent1, parent2):
    if random.random() < crossoverRate:
        point = random.randint(0, N)
        child1_gene = parent1.gene[:point] + parent2.gene[point:]
        child2_gene = parent2.gene[:point] + parent1.gene[point:]
        return Individual(child1_gene), Individual(child2_gene)
    else:
        return parent1, parent2

def elitism(pop, best):
    worst = min(pop, key=lambda x: x.fitness)
    if best.fitness > worst.fitness:
        pop[pop.index(worst)] = best
    return pop

def geneticAlgorithm():
    population = createInitialPopulation()
    rulebase = loadDataset(dataloc)
    generation = 0
    best_individual = None
    mean_fitnesses = []

    while generation < nGen:
        for individual in population:
            individual.fitnessFunction(rulebase)
        
        mean_fitness = mean(individual.fitness for individual in population)
        mean_fitnesses.append(mean_fitness)
        best_individual = max(population, key=lambda x: x.fitness)

        if best_individual.fitness >= maxFitness:
            break

        selected_parents = selection(population)
        next_generation = []
        for i in range(0, len(selected_parents), 2):
            parent1, parent2 = selected_parents[i], selected_parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([child1, child2])

        population = elitism(next_generation, best_individual)
        generation += 1

    return mean_fitnesses, best_individual

if __name__ == "__main__":
    dataloc = "dataset1.csv"
    condLength = 5
    outLength = 1
    N = 6 * (condLength + outLength)
    P = 300
    nGen = 100
    crossoverRate = 0.95
    tournament_size = 3
    maxFitness = 30

    mean_fitnesses, best_individual = geneticAlgorithm()

    plt.plot(mean_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Mean Fitness')
    plt.title('Mean Fitness over Generations')
    plt.show()

    print("Best Individual Fitness:", best_individual.fitness)
    print("Best Individual Gene:", best_individual.gene)
