import random
import numpy as np
import matplotlib.pyplot as plt

# Constants
PARAMETER = 10
LINEAR_REGRESSION_ERROR_RANGE = 3
ERROR_RANGE = 1.0001
MUTATION_RATE = 0.1
NUM_POINTS = 45
NUM_ELITES = 2

# Function 1 (given)
def generate_random_points(num_points, slope, intercept):
    noise = np.random.normal(0, LINEAR_REGRESSION_ERROR_RANGE, num_points)
    x_coords = np.random.uniform(-PARAMETER, PARAMETER, num_points)
    y_coords = slope * x_coords + intercept + noise
    return np.column_stack((x_coords, y_coords))

# Function 2
def generate_random_lines(num_lines):
    slopes = np.random.uniform(-PARAMETER, PARAMETER, num_lines)
    intercepts = np.random.uniform(-PARAMETER, PARAMETER, num_lines)
    return np.column_stack((slopes, intercepts))

# Function 3
def calculate_errors(lines, points):
    x = points[:, 0]
    y = points[:, 1]

    slopes = lines[:, 0][:, np.newaxis]
    intercepts = lines[:, 1][:, np.newaxis]

    y_pred = slopes * x + intercepts
    errors = np.sum((y_pred - y) ** 2, axis=1)
    return errors

# Function 4
def mutate(line, mutation_rate=MUTATION_RATE):
    if random.random() < mutation_rate:
        line[0] += np.random.normal(0, 0.1 * abs(line[0]) + 0.01)
    if random.random() < mutation_rate:
        line[1] += np.random.normal(0, 0.1 * abs(line[1]) + 0.01)
    return line

# Function 5
def create_next_generation(lines, points, num_elites=NUM_ELITES):
    errors = calculate_errors(lines, points)
    sorted_idx = np.argsort(errors)

    elites = lines[sorted_idx[:num_elites]]
    new_generation = elites.tolist()

    while len(new_generation) < len(lines):
        p1, p2 = random.sample(list(lines), 2)
        child = [p1[0], p2[1]]
        child = mutate(child)
        new_generation.append(child)

    return np.array(new_generation)

# Function 6 (given)
def plot_progress(points, actual_line, predicted_line, generation):
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Data Points", color="blue")
    
    x_fit = np.linspace(min(x), max(x), 100)
    y_actual = actual_line[0] * x_fit + actual_line[1]
    y_predicted = predicted_line[0] * x_fit + predicted_line[1]

    plt.plot(x_fit, y_actual, label="Actual Line", color="green", linewidth=2)
    plt.plot(x_fit, y_predicted, label=f"Predicted Line (Gen {generation})", color="red", linestyle="dashed")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Genetic Algorithm Progress")
    plt.legend()
    plt.show()

# Function 7
def genetic_algorithm():
    num_points = NUM_POINTS
    true_slope = random.uniform(-PARAMETER, PARAMETER)
    true_intercept = random.uniform(-PARAMETER, PARAMETER)
    test_points = generate_random_points(num_points, true_slope, true_intercept)
    actual_line = [true_slope, true_intercept]

    print("Actual Line:", actual_line, flush=True)

    population_size = 50
    lines = generate_random_lines(population_size)

    generation = 0
    max_generations = 100

    while generation < max_generations:
        errors = calculate_errors(lines, test_points)
        best_idx = np.argmin(errors)
        best_line = lines[best_idx]
        best_error = errors[best_idx]

        print(f"Generation {generation} | Best error: {best_error}", flush=True)

        if generation % 50 == 0:
            plot_progress(test_points, actual_line, best_line, generation)

        if best_error <= ERROR_RANGE:
            print("Converged!", flush=True)
            break

        lines = create_next_generation(lines, test_points)
        generation += 1

# Run
genetic_algorithm()
