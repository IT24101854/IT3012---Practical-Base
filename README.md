# Genetic Algorithm for 0/1 Knapsack

## Requirements

Python 3.x is required.

Install the required libraries using:

```bash
pip install deap matplotlib

##Dataset
The program uses the dataset:
kp_wc_100.txt
The dataset contains 100 items with a knapsack capacity of 525.

## How to Run
Run the program using:
python knapsack_ga.py

The program implements a Genetic Algorithm using:

- Binary chromosome representation
- Tournament, roulette-wheel and rank selection
- One-point, two-point and uniform crossover
- Bit-flip mutation
- Elitism
- Best and average fitness tracking
- Convergence analysis

The program runs the baseline configuration and the required operator comparisons.

## Baseline Parameters

- Population size: 150
- Generations: 200
- Crossover probability: 0.9
- Mutation probability: 0.02
- Elitism: 2
- Tournament selection: k = 3
- Two-point crossover
- Bit-flip mutation
Experiments

Selection
- Tournament k = 2
- Tournament k = 3
- Tournament k = 4
- Roulette
- Rank

Crossover
- One-point
- Two-point
- Uniform

Mutation
- pm = 0.01
- pm = 0.05
- pm = 0.10

## Output
The program displays:
- Best fitness
- Total weight
- Solution validity
- Convergence information