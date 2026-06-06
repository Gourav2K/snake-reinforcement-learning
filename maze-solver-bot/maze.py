import random

def create_maze(size, num_traps, seed=None):
    # Set the seed for the random number generator
    if seed is not None:
        random.seed(seed)
    
    # Initialize the maze with empty cells
    maze = [['E' for _ in range(size)] for _ in range(size)]
    
    # Set the start and finish points
    maze[0][0] = 'S'
    maze[size-1][size-1] = 'F'
    
    # Place traps randomly in the maze
    traps = 0
    while traps < num_traps:
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        if maze[x][y] == 'E':
            maze[x][y] = 'T'
            traps += 1

    return maze

def print_maze(maze):
    for row in maze:
        print(" ".join(row))
