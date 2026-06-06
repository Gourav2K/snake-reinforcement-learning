import random

def make_random_move():
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, left, down, up
    return random.choice(possible_moves)

def game_over(coordinates, grid_size):
    x = coordinates[0]
    y = coordinates[1]
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return True
    else:
        return False
    
def analyse_game_state(coordinates, grid_size):
    if game_over(coordinates, grid_size):
        return -1
    else:
        return 0

def reward_function(coordinates, maze_grid):
    x = coordinates[0]
    y = coordinates[1]
    print("Cell contains " + str(maze_grid[x][y]))
    if maze_grid[x][y] == '1':
        return 0
    elif maze_grid[x][y] == '2':
        return -10
    elif maze_grid[x][y] == '3':
        return 10
    elif maze_grid[x][y] == '0':
        return -100


    
    