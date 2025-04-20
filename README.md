# fuzzy_traffic
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import random

queue_length = ctrl.Antecedent(np.arange(0, 101, 1), 'queue_length')
waiting_time = ctrl.Antecedent(np.arange(0, 301, 1), 'waiting_time')
green_time = ctrl.Consequent(np.arange(10, 91, 1), 'green_time')

queue_length['short'] = fuzz.trimf(queue_length.universe, [0, 0, 40])
queue_length['medium'] = fuzz.trimf(queue_length.universe, [20, 50, 80])
queue_length['long'] = fuzz.trimf(queue_length.universe, [60, 100, 100])

waiting_time['low'] = fuzz.trimf(waiting_time.universe, [0, 0, 100])
waiting_time['medium'] = fuzz.trimf(waiting_time.universe, [50, 150, 250])
waiting_time['high'] = fuzz.trimf(waiting_time.universe, [200, 300, 300])

green_time['short'] = fuzz.trimf(green_time.universe, [10, 20, 30])
green_time['medium'] = fuzz.trimf(green_time.universe, [30, 50, 70])
green_time['long'] = fuzz.trimf(green_time.universe, [60, 80, 90])

rule1 = ctrl.Rule(queue_length['long'] & waiting_time['high'], green_time['long'])
rule2 = ctrl.Rule(queue_length['medium'] & waiting_time['medium'], green_time['medium'])
rule3 = ctrl.Rule(queue_length['short'] & waiting_time['low'], green_time['short'])

green_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
green_simulator = ctrl.ControlSystemSimulation(green_ctrl)

np.random.seed(0)
X = np.random.rand(100, 4)
y = np.random.rand(100) * 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(X, weights):
    w1, b1, w2, b2 = weights
    hidden = sigmoid(np.dot(X, w1) + b1)
    output = np.dot(hidden, w2) + b2
    return output

def create_weights():
    w1 = np.random.randn(4, 6)
    b1 = np.random.randn(6)
    w2 = np.random.randn(6, 1)
    b2 = np.random.randn(1)
    return [w1, b1, w2, b2]

def flatten(weights):
    return np.concatenate([w.flatten() for w in weights])

def unflatten(flat_vector):
    w1 = flat_vector[:24].reshape(4, 6)
    b1 = flat_vector[24:30]
    w2 = flat_vector[30:36].reshape(6, 1)
    b2 = flat_vector[36:37]
    return [w1, b1, w2, b2]

def fitness(chromosome):
    weights = unflatten(chromosome)
    preds = feedforward(X, weights).flatten()
    return -np.mean((preds - y)**2)

def crossover(p1, p2):
    alpha = 0.5
    return alpha * p1 + (1 - alpha) * p2

def mutate(chrom, mutation_rate=0.1):
    for i in range(len(chrom)):
        if np.random.rand() < mutation_rate:
            chrom[i] += np.random.normal(0, 0.1)
    return chrom

population_size = 30
generations = 50
pop = [flatten(create_weights()) for _ in range(population_size)]

for gen in range(generations):
    scored = [(chrom, fitness(chrom)) for chrom in pop]
    scored.sort(key=lambda x: x[1], reverse=True)
    pop = [x[0] for x in scored[:10]]

    while len(pop) < population_size:
        p1, p2 = random.sample(pop[:5], 2)
        child = crossover(p1, p2)
        child = mutate(child)
        pop.append(child)

    print(f"Generation {gen}, Best Fitness: {-scored[0][1]:.4f}")

best_weights = unflatten(pop[0])
final_preds = feedforward(X, best_weights).flatten()

plt.plot(y, label='Actual')
plt.plot(final_preds, label='Predicted')
plt.legend()
plt.title('Traffic Popularity Prediction')
plt.xlabel('Samples')
plt.ylabel('Popularity Score')
plt.grid(True)
plt.show()
