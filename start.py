import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Hey")

directions_agent_can_move = 4
environment_x_length = 10
environment_y_length = 10

q_table_width = directions_agent_can_move

q_table = np.zeros((environment_x_length*environment_y_length,directions_agent_can_move))

direction_map = {
    "up": (0,-1),
    "down":(0,1),
    "left":(-1,0),
    "right":(1,0)
    }


'''
Plan for which functions to create to allow the agent to move across the environment.

1. A function that converts (x, y) to Q-table row index

2. A function that takes current position and direction, returns new position

3. A function that checks if a move is valid (not hitting a wall)

4. A function that updates the Q-table after learning'''

def coordinates_to_q_table_index(coordinates: tuple[int, int]) -> int:
    '''A function that converts (x, y) to Q-table row index. If the x and y coordinates are outside the environment's dimensions, the function returns -1'''
    
    global q_table_width
    global environment_x_length
    global environment_y_length
    
    x_coord = coordinates[0]
    y_coord = coordinates[1]
    
    if (x_coord >= environment_x_length) or (y_coord >= environment_y_length) or (x_coord < 0) or (y_coord < 0 ):
        return -1
    
    row_index = x_coord + y_coord * q_table_width
    
    
    return row_index


'''
Here I learn how tuples work

print([5,5] + [0,1])
# results in [5,5,0,1]
'''


def coordinates_after_moving(coordinates: tuple[int, int], direction: str) -> tuple[int, int]:
    """A function that takes current position and direction, returns new position.
    
    If the inputted coordinates are outside the environmet, the function returns [-1.-1].
    
    If the movement takes the agent outside the environment, the function returns [-5,-5].  """
    
    global q_table_width
    global environment_x_length
    global environment_y_length
    global direction_map
    
    x_coord = coordinates[0]
    y_coord = coordinates[1]
    
    # checking if the inputted coordinates are inside the environment
    if (x_coord >= environment_x_length) or (y_coord >= environment_y_length) or (x_coord < 0) or (y_coord < 0 ):
        return (-1,-1)
    
    # creating new coordinates
    
    new_x_coord = x_coord + direction_map[direction][0]
    new_y_coord = y_coord + direction_map[direction][1]

    
    # checking if the resulting coordinates are inside the environment
    if (new_x_coord >= environment_x_length) or (new_y_coord >= environment_y_length) or (new_x_coord < 0) or (new_y_coord < 0 ):
        return (-5,-5)
    
    # at this point, since everything has been checked and the movement is valid, we create the new tuple (different place in memory) representing the new coordinates after the movement.
    
    new_coords = (new_x_coord, new_y_coord)

    return new_coords