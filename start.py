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

print(q_table)


direction_map = {
    "up": (0,-1),
    "down":(0,1),
    "left":(-1,0),
    "right":(1,0)
    }


'''A function that converts (x, y) to Q-table row index

A function that takes current position and direction, returns new position

A function that checks if a move is valid (not hitting a wall)

A function that updates the Q-table after learning'''

def coordinates_to_q_table_index(x_coord: int, y_coord: int) -> int:
    '''A function that converts (x, y) to Q-table row index. If the x and y coordinates are outside the environment's dimensions, the function returns -1'''
    
    global q_table_width
    global environment_x_length
    global environment_y_length
    
    if (x_coord > environment_x_length) or (y_coord > environment_y_length) or (x_coord < 0) or (y_coord < 0 ):
        return -1
    
    row_index = x_coord + y_coord * q_table_width
    
    
    return row_index