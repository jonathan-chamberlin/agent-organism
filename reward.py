import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import math
from logic import *
from setup_environment import *

cell_reward = {
    "wall": -10,
    "goal": 50,
    "empty": -1,
    "start": 0 
}

wall_reward = cell_reward["wall"]
goal_reward = cell_reward["goal"]
start_reward = cell_reward["start"]
empty_reward = cell_reward["empty"]

'''LEFT OFF, what do do next. Here is a list of functions that I need to build, the comments below are a detailed plan about what has to be built.

# You have three key functions to build:
# get_reward(old_pos, new_pos, is_valid, environment) - Returns reward number
# choose_action(current_pos, q_table, epsilon) - Returns direction string
# update_q_table(old_pos, action, new_pos, q_table, alpha, gamma) - Modifies q_table


# q table starts blank
# I use a get_reward function that takes in the starting coordinates and a direction, and it uses coordinates_after_moving to determine if the move is valid, and it finds the type of cell the agent is trying to move to, and it looks at the cell_reward dictionary, and therefore it outputs the reward the agent.

# Then I need a function choose_action that takes in the agent's current coordinates and the whole Q table, and just reads teh Q table and finds which value is the highest, and it has a 90% chance of picking the move with the highest q value (this is the explotation rate, or 1-epsilon, where epsilon is the exploration rate), and 10% likely to chose another move at random. This function will use epsilon as an inputs. If I want, I call the draw_agent function to render the agent at those new coordinates. 

# I need another function update_q_table which takes in a starting coordiate, direction moved from that starting coordinate, and the q table, and the final coordinates (even though this can be calcuaed from starting coords and direction moved, I'll already have the next coords from using the coordinates_after_moving function, so this will save redundant computation.). It calls get_reward to determine the reward for making that move, and uses the Q(state,action) ML formula {{  Q(state, action) = Q(state, action) + α × [reward + γ × max(Q(next_state, all_possible_actions_from_that_state)) - Q(state, action)]  }}, then the function updates the row of the Q table with that reward value.

# Then I have to make a game loop that makes it so multiple simulations happen consecutively, and that subsequent simulations use the prior Q table.

# The simulation run ends if the agent hits the goal, or reaches the max number of steps. The ideal path is 30 moves, so my step_cap will be 50.

# To save knowledge between runs, I'd store the Q table in a text file using np.save(file,array) and np.load(file). This allows me to use data from previous times I ran the code, instead of the agent having to relearn everything from scratch.'''

def get_reward(old_pos: tuple[int,[int]], direction: str, environment: tuple[tuple[int,int]], walls: list[tuple[int,int]]) -> tuple[int , tuple[int,int], bool]:
    """Takes in the starting coordinates and a direction, and it uses coordinates_after_moving to determine if the move is valid, and it finds the type of cell the agent is trying to move to, and it looks at the cell_reward dictionary, and therefore it outputs the reward the agent.
    
    Outputs reward, new_pos coords, and
    movement_valid 
    """
    
    global cell_reward
    
    new_adjacent_coords = adjacent_coords(old_pos,direction)
    
    movement_valid = coordinates_after_moving(old_pos,direction,walls)[1]
    
    object_at_adjacent_coords = object_at_coords(new_adjacent_coords, environment)
    
    reward = cell_reward[object_at_adjacent_coords]
    
    return (reward, new_adjacent_coords, movement_valid) 

# LEFT OFF. Create function choose_action explained above. So do Ctrl+F choose_action to find the commented out part above, read it, then build it.