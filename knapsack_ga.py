import random
import matplotlib.pyplot as plt
from deap import base, creator, tools


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "kp_wc_100.txt"

POPULATION_SIZE = 150
GENERATIONS = 200
CROSSOVER_PROBABILITY = 0.9
BASELINE_MUTATION_PROBABILITY = 0.02
ELITISM = 2

# Median fitness obtained from the baseline run used in the report
BASELINE_MEDIAN = 607

# None = stochastic runs
# Set to an integer if deterministic runs are required
SEED = None


# ============================================================
# LOAD DATASET
# ============================================================

def load_knapsack(filename):

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])

    values = []
    weights = []

    for line in lines[1:n + 1]:
        parts = line.split()

        values.append(int(parts[1]))
        weights.append(int(parts[2]))

    capacity = int(lines[n + 1])

    return values, weights, capacity


values, weights, capacity = load_knapsack(DATASET_FILE)


# ============================================================
# DEAP TYPES
# ============================================================

if not hasattr(creator, "FitnessMaxKnapsack"):
    creator.create(
        "FitnessMaxKnapsack",
        base.Fitness,
        weights=(1.0,)
    )

if not hasattr(creator, "IndividualKnapsack"):
    creator.create(
        "IndividualKnapsack",
        list,
        fitness=creator.FitnessMaxKnapsack
    )


# ============================================================
# FITNESS FUNCTION
# ============================================================

def evaluate(individual):

    total_value = 0
    total_weight = 0

    for i, gene in enumerate(individual):

        if gene == 1:
            total_value += values[i]
            total_weight += weights[i]

    # Valid solution
    if total_weight <= capacity:
        return (total_value,)

    # Penalty for exceeding capacity
    penalty = total_weight - capacity

    return (-penalty,)


# ============================================================
# ROULETTE-WHEEL SELECTION
# ============================================================

def sel_roulette_shifted(individuals, k):

    # Roulette selection requires non-negative weights.
    # Shift negative fitness values when necessary.

    min_fitness = min(
        ind.fitness.values[0]
        for ind in individuals
    )

    if min_fitness <= 0:
        offset = -min_fitness + 1e-6
    else:
        offset = 0

    selection_weights = [
        ind.fitness.values[0] + offset
        for ind in individuals
    ]

    total_weight = sum(selection_weights)

    if total_weight <= 0:
        return [
            random.choice(individuals)
            for _ in range(k)
        ]

    selected = []

    for _ in range(k):

        r = random.random() * total_weight
        cumulative = 0

        for individual, weight in zip(
                individuals,
                selection_weights
        ):

            cumulative += weight

            if cumulative >= r:
                selected.append(individual)
                break

    return selected


# ============================================================
# RANK SELECTION
# ============================================================

def sel_rank(individuals, k):

    ranked = sorted(
        individuals,
        key=lambda ind: ind.fitness.values[0],
        reverse=True
    )

    ranks = list(
        range(len(ranked), 0, -1)
    )

    total_rank = sum(ranks)

    selected = []

    for _ in range(k):

        r = random.random() * total_rank
        cumulative = 0

        for individual, rank in zip(
                ranked,
                ranks
        ):

            cumulative += rank

            if cumulative >= r:
                selected.append(individual)
                break

    return selected


# ============================================================
# GET SELECTION OPERATOR
# ============================================================

def get_selection_operator(method):

    if method == "tournament_k2":
        return tools.selTournament, {"tournsize": 2}

    if method == "tournament_k3":
        return tools.selTournament, {"tournsize": 3}

    if method == "tournament_k4":
        return tools.selTournament, {"tournsize": 4}

    if method == "roulette":
        return sel_roulette_shifted, {}

    if method == "rank":
        return sel_rank, {}

    raise ValueError("Unknown selection method")


# ============================================================
# APPLY CROSSOVER
# ============================================================

def apply_crossover(child1, child2, method):

    if method == "one_point":

        tools.cxOnePoint(
            child1,
            child2
        )

    elif method == "two_point":

        tools.cxTwoPoint(
            child1,
            child2
        )

    elif method == "uniform":

        tools.cxUniform(
            child1,
            child2,
            indpb=0.5
        )

    else:
        raise ValueError("Unknown crossover method")


# ============================================================
# RUN ONE GA EXPERIMENT
# ============================================================

def run_ga(
        selection_method,
        crossover_method,
        mutation_probability,
        experiment_name,
        show_graph=False
):

    if SEED is not None:
        random.seed(SEED)

    toolbox = base.Toolbox()

    # --------------------------------------------------------
    # Binary gene
    # --------------------------------------------------------

    toolbox.register(
        "attr_bool",
        random.randint,
        0,
        1
    )

    # --------------------------------------------------------
    # Individual
    # --------------------------------------------------------

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.IndividualKnapsack,
        toolbox.attr_bool,
        n=len(values)
    )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    # --------------------------------------------------------
    # Fitness
    # --------------------------------------------------------

    toolbox.register(
        "evaluate",
        evaluate
    )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selection_function, selection_arguments = (
        get_selection_operator(selection_method)
    )

    toolbox.register(
        "select",
        selection_function,
        **selection_arguments
    )

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = toolbox.population(
        n=POPULATION_SIZE
    )

    # Evaluate initial population

    for individual in population:

        individual.fitness.values = (
            toolbox.evaluate(individual)
        )

    # History

    best_fitness_history = []
    average_fitness_history = []

    # ========================================================
    # EVOLUTION
    # ========================================================

    for generation in range(GENERATIONS):

        # ----------------------------------------------------
        # Selection
        # ----------------------------------------------------

        selected = toolbox.select(
            population,
            len(population)
        )

        offspring = list(
            map(toolbox.clone, selected)
        )

        # ----------------------------------------------------
        # Crossover
        # ----------------------------------------------------

        for child1, child2 in zip(
                offspring[::2],
                offspring[1::2]
        ):

            if random.random() < CROSSOVER_PROBABILITY:

                apply_crossover(
                    child1,
                    child2,
                    crossover_method
                )

                del child1.fitness.values
                del child2.fitness.values

        # ----------------------------------------------------
        # Mutation
        # ----------------------------------------------------

        for individual in offspring:

            tools.mutFlipBit(
                individual,
                indpb=mutation_probability
            )

            del individual.fitness.values

        # ----------------------------------------------------
        # Evaluate changed individuals
        # ----------------------------------------------------

        invalid_individuals = [
            individual
            for individual in offspring
            if not individual.fitness.valid
        ]

        for individual in invalid_individuals:

            individual.fitness.values = (
                toolbox.evaluate(individual)
            )

        # ----------------------------------------------------
        # Elitism
        # ----------------------------------------------------

        elites = tools.selBest(
            population,
            ELITISM
        )

        for i, elite in enumerate(elites):

            offspring[-(i + 1)] = (
                toolbox.clone(elite)
            )

        # New population

        population = offspring

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        best = tools.selBest(
            population,
            1
        )[0]

        best_fitness = (
            best.fitness.values[0]
        )

        average_fitness = sum(
            individual.fitness.values[0]
            for individual in population
        ) / len(population)

        best_fitness_history.append(
            best_fitness
        )

        average_fitness_history.append(
            average_fitness
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    best_individual = tools.selBest(
        population,
        1
    )[0]

    final_value = 0
    final_weight = 0

    for gene, value, weight in zip(
            best_individual,
            values,
            weights
    ):

        if gene == 1:
            final_value += value
            final_weight += weight

    final_fitness = (
        best_individual.fitness.values[0]
    )

    valid = final_weight <= capacity

    # --------------------------------------------------------
    # Convergence
    # --------------------------------------------------------

    first_generation = None

    for generation, fitness in enumerate(
            best_fitness_history,
            start=1
    ):

        if fitness > BASELINE_MEDIAN:

            first_generation = generation
            break

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    if show_graph:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            range(1, GENERATIONS + 1),
            best_fitness_history,
            label="Best Fitness"
        )

        plt.plot(
            range(1, GENERATIONS + 1),
            average_fitness_history,
            label="Average Fitness"
        )

        plt.xlabel("Generation")
        plt.ylabel("Fitness")

        plt.title(
            f"GA Convergence - {experiment_name}"
        )

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return {
        "name": experiment_name,
        "fitness": final_fitness,
        "weight": final_weight,
        "valid": valid,
        "first_generation": first_generation,
        "best_history": best_fitness_history,
        "average_history": average_fitness_history
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print()
    print("=" * 55)
    print(result["name"])
    print("=" * 55)

    print(
        "Best fitness:",
        result["fitness"]
    )

    print(
        "Total weight:",
        result["weight"]
    )

    print(
        "Capacity:",
        capacity
    )

    print(
        "Valid solution:",
        "Yes" if result["valid"] else "No"
    )

    if result["first_generation"] is not None:

        print(
            "First generation surpassing baseline median:",
            result["first_generation"]
        )

    else:

        print(
            "Baseline median was not surpassed."
        )


# ============================================================
# RUN ALL REQUIRED EXPERIMENTS
# ============================================================

def run_all_experiments():

    print("=" * 55)
    print("GENETIC ALGORITHM - 0/1 KNAPSACK")
    print("=" * 55)

    print(
        "Number of items:",
        len(values)
    )

    print(
        "Knapsack capacity:",
        capacity
    )

    print(
        "Population size:",
        POPULATION_SIZE
    )

    print(
        "Generations:",
        GENERATIONS
    )

    print(
        "Crossover probability:",
        CROSSOVER_PROBABILITY
    )

    print(
        "Baseline mutation probability:",
        BASELINE_MUTATION_PROBABILITY
    )

    print(
        "Elitism:",
        ELITISM
    )

    print(
        "Baseline median:",
        BASELINE_MEDIAN
    )

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = run_ga(
        selection_method="tournament_k3",
        crossover_method="two_point",
        mutation_probability=0.02,
        experiment_name="Baseline",
        show_graph=True
    )

    print_result(baseline)

    # ========================================================
    # SELECTION EXPERIMENTS
    # ========================================================

    selection_experiments = [

        ("tournament_k2", "Tournament k=2"),

        ("tournament_k3", "Tournament k=3"),

        ("tournament_k4", "Tournament k=4"),

        ("roulette", "Roulette Selection"),

        ("rank", "Rank Selection")

    ]

    print("\n\nSELECTION EXPERIMENTS")

    for method, name in selection_experiments:

        result = run_ga(
            selection_method=method,
            crossover_method="two_point",
            mutation_probability=0.02,
            experiment_name=name,
            show_graph=False
        )

        print_result(result)

    # ========================================================
    # CROSSOVER EXPERIMENTS
    # ========================================================

    crossover_experiments = [

        ("one_point", "One-point Crossover"),

        ("two_point", "Two-point Crossover"),

        ("uniform", "Uniform Crossover")

    ]

    print("\n\nCROSSOVER EXPERIMENTS")

    for method, name in crossover_experiments:

        result = run_ga(
            selection_method="tournament_k3",
            crossover_method=method,
            mutation_probability=0.02,
            experiment_name=name,
            show_graph=False
        )

        print_result(result)

    # ========================================================
    # MUTATION EXPERIMENTS
    # ========================================================

    mutation_experiments = [

        (0.01, "Mutation pm=0.01"),

        (0.05, "Mutation pm=0.05"),

        (0.10, "Mutation pm=0.10")

    ]

    print("\n\nMUTATION EXPERIMENTS")

    for mutation_probability, name in mutation_experiments:

        result = run_ga(
            selection_method="tournament_k3",
            crossover_method="two_point",
            mutation_probability=mutation_probability,
            experiment_name=name,
            show_graph=False
        )

        print_result(result)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_all_experiments()