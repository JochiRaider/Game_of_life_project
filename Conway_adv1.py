import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


ON = 255
OFF = 0
vals = [ON, OFF]
MUTATION_PROB = 0.1
n = 100


def random_grid(n: int) -> np.array:
    return np.random.choice(vals, n*n, p=[.2, .8]).reshape(n, n)


def pat_glider(i: int, j: int, grid: np.array):
    glider = np.array([[OFF, OFF, ON], [ON, OFF, ON], [OFF, ON, ON]])
    grid[i:i+3, j:j+3] = glider


def update_grid(frame_num, img, grid: np.array, n: int, germ_grid: np.array):
    local_grid = grid.copy()
    local_germ_grid = germ_grid.copy()
    for i in range(n):
        for j in range(n):
            total = int(((grid[i, (j-1) % n] + grid[i, (j+1) % n]) + grid[(i-1) % n, j] +
                         grid[(i+1) % n, j] + grid[(i+1) % n, (j+1) % n] + grid[(i-1) % n, (j-1) % n] +
                         grid[(i+1) % n, (j-1) % n] + grid[(i-1) % n, (j+1) % n])/ON)
            if grid[i, j] == ON and (total < 2 or total > 3):
                local_grid[i, j] = OFF
            elif total == 3:
                local_grid[i, j] = ON
                local_germ_grid[i, j] = germ_rule(grid, germ_grid, i, j)  # apply germ rule
            elif total == 2:  # add inheritance rule
                local_germ_grid[i, j] = germ_rule(grid, germ_grid, i, j)
    img.set_data(local_grid)
    grid[:] = local_grid[:]
    germ_grid[:] = local_germ_grid[:]
    return img


def health_measure(grid: np.array, germ_grid: np.array, i: int, j: int) -> int:
    """
    Calculates the health of a germ cell at (i, j) based on the number of consecutive frames in which it has been alive.
    """
    if germ_grid[i, j] == ON:
        if grid[i, j] == ON:  # germ cell is alive
            return germ_grid[i, j] + 1
        else:  # germ cell is dead
            return 0
    else:  # not a germ cell
        return 0

def germ_rule(grid: np.array, germ_grid: np.array, i: int, j: int) -> int:
    """
    Returns the new state of the germ cell at (i, j) based on the current state of the grid and germ grid.
    """
    alpha = 0.4
    beta = 0.6
    if germ_grid[i, j] == ON:  # if the germ cell is already ON, no need to change it
        return ON
    elif germ_grid[i, j] == OFF:  # if the germ cell is OFF, apply stochastic mutation or recombination
        if random.random() < MUTATION_PROB:  # apply mutation with probability MUTATION_PROB
            return ON
        else:  # otherwise, apply recombination with selection based on fitness
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
            valid_neighbors = [(x % n, y % n) for (x, y) in neighbors if germ_grid[x % n, y % n] == ON]
            if valid_neighbors:
                fitness_values = []
                for x, y in valid_neighbors:
                    total_germ_cells = np.sum(germ_grid[(x-1)%n:(x+2)%n, (y-1)%n:(y+2)%n]) - germ_grid[x, y]
                    health = health_measure(grid, germ_grid, x, y)
                    fitness = germ_grid[x, y] * (1 + alpha * total_germ_cells) * (1 + beta * health)
                    fitness_values.append(fitness)
                fitness_sum = sum(fitness_values)
                if fitness_sum > 0:
                    fitness_probs = [abs(fitness / fitness_sum) for fitness in fitness_values]
                    fitness_probs /= sum(fitness_probs)  # normalize probabilities
                    idx = np.random.choice(len(valid_neighbors), p=fitness_probs)
                    return germ_grid[valid_neighbors[idx]]
                else:
                    return OFF
            else:
                return OFF


def main():
    update_interval = 50
    grid = random_grid(n)
    germ_grid = np.zeros(n*n).reshape(n, n) # initialize germ grid to all OFF
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    generation = 0
    def update(frame_num, img, grid, n, germ_grid):
        nonlocal generation
        generation += 1
        update_grid(frame_num, img, grid, n, germ_grid)
        print("Generation:", generation)
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, n, germ_grid), 
                                   frames=10, interval=update_interval, save_count=50)
    plt.show()

if __name__ == '__main__':
    main()