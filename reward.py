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

'''what do do next. Here is a list of functions that I need to build, the comments below are a detailed plan about what has to be built.

# You have three key functions to build:
# get_reward(old_pos, new_pos, is_valid, environment) - Returns reward number
# choose_action(current_pos, q_table, epsilon) - Returns action string
# update_q_table(old_pos, action, new_pos, q_table, alpha, gamma) - Modifies q_table


# DONE. q table starts blank
# DONE. I use a get_reward function that takes in the starting coordinates and a action, and it uses coordinates_after_moving to determine if the move is valid, and it finds the type of cell the agent is trying to move to, and it looks at the cell_reward dictionary, and therefore it outputs the reward the agent.

# DONE. Then I need a function choose_action that takes in the agent's current coordinates and the whole Q table, and just reads teh Q table and finds which value is the highest, and it has a 90% chance of picking the move with the highest q value (this is the explotation rate, or 1-epsilon, where epsilon is the exploration rate), and 10% likely to chose another move at random. This function will use epsilon as an inputs.
# DONE. First I tested the implementation of choose_action while only considering a 0 explore rate. Now I have to add some randomness.

# DONE.I need another function update_q_table which takes in a starting coordiate, action moved from that starting coordinate, and the q table, and the final coordinates (even though this can be calcuaed from starting coords and action moved, I'll already have the next coords from using the coordinates_after_moving function, so this will save redundant computation.). It calls get_reward to determine the reward for making that move, and uses the Q(state,action) ML formula {{  Q(state, action) = Q(state, action) + α × [reward + γ × max(Q(next_state, all_possible_actions_from_that_state)) - Q(state, action)]  }}, then the function updates the row of the Q table with that reward value.

DONE. Modify game_loop_manual, add an input 'rendering': str as an input. It could speed up my simulation. Value of 'pygame' could make it so every frame is rendered on the pygame window. Value of 'print' means that the full environment is printed in the terminal every frame. And 'none' means that the function does all the calculations without printing or rendering anything.

DONE. Create game_loop_learning that makes it so the agent moves through the enviornment using choose_action and update_q_table. Each move is stored in a list of moves. After all those calculations are done, if rendering = 'pygame', the function calls game_loop_manual using that list of moves to render the agent's moves using draw_agent. After all the moves, print the q table.

Now test game_loop_learning by using the function with all it's inputs

# Modify function game_loop that makes it so the agent moves through the enviornment using choose_action and update_q_table, and that the agent's moves are rendered using draw_agent. After all the moves, print the q table.

# Modify function game_loop that makes it so multiple simulations happen consecutively, and that subsequent simulations use the updated Q table.

# The simulation run ends if the agent hits the goal, or reaches the max number of steps. The ideal path is 30 moves, so my step_cap will be 50.

# To save knowledge between runs, I'd store the Q table in a text file using np.save(file,array) and np.load(file). This allows me to use data from previous times I ran the code, instead of the agent having to relearn everything from scratch.

# Make a function get_q_values_at_coordinates(coords: tuple[int,int], q_table) -> list[float]. It outputs gets the q value row for the inputted coordinates. It uses coords_to_q_table_index

# Make a function display_q_values_around_agent. It takes in a bunch of stuff, and uses get_q_values_at_coordinates, then every frame displays the q values around the agent representing the q value for down, right, up, left, remain.

'''

def get_reward(old_pos: tuple[int,[int]], action: tuple[int,int], possible_actions: list[tuple[int,int]], environment: tuple[tuple[int,int]], walls: list[tuple[int,int]]) -> tuple[int , tuple[int,int], bool]:
    """Takes in the starting coordinates and a action, and it uses coordinates_after_moving to determine if the move is valid, and it finds the type of cell the agent is trying to move to, and it looks at the cell_reward dictionary, and therefore it outputs the reward the agent.
    
    Outputs reward, new_pos coords, and
    movement_valid 
    """
    
    global cell_reward
    
    new_adjacent_coords = adjacent_coords(old_pos,action)
    
    movement_valid = coordinates_after_moving(old_pos,action,possible_actions,walls)[1]
    
    object_at_adjacent_coords = object_at_coords(new_adjacent_coords, environment)
    
    reward = cell_reward[object_at_adjacent_coords]
    
    return (reward, new_adjacent_coords, movement_valid) 


def choose_action(current_pos: tuple(int,int), q_table: tuple[tuple[int,int]], possible_actions: list[tuple[int,int]], environment_x_length: int, environment_y_length: int, epsilon: float) -> tuple[tuple[int,int]]: 
    """takes in the agent's current coordinates and the whole Q table, and just reads thh Q table and finds which value is the highest, and it has a 1-epsilon chance of picking the move with the highest q value (this is the explotation rate, or 1-epsilon, where epsilon is the exploration rate), and an epsilon likely to chose another move at random.

    Returns a tuple where the first string is the optimal action, and the second is an action which was randomly chosen using epsilon the exploration rate.
    
    If every value in the q table row is 0, then the agent picks a direction randomly.
    """
    
    q_table_width = len(q_table[0])
    
    q_table_index = coordinates_to_q_table_index(current_pos,environment_x_length, environment_y_length, q_table_width)
    
    row = q_table[q_table_index]
    
    action_indices = list(range(0,len(possible_actions)))
    
    # If every value in the q table row is 0, then the agent picks a direction randomly.
    if np.all(row == 0):
        randomly_chosen_action_index = np.random.choice(action_indices)
        randomly_chosen_action = possible_actions[randomly_chosen_action_index]
        return (randomly_chosen_action,randomly_chosen_action)
    
    # index of the maximum q value in the row for the inputted coords current_pos. if multiple actions have the same q value, the function will always return the index of the first instance with that maximum value.
    optimal_action_index = np.argmax(row)
    
    optimal_action = possible_actions[optimal_action_index]
    
    # decides if the function will return the optimal_action or another
    pick_optimal = (np.random.random() > epsilon)
    
    if pick_optimal == True:
        return (optimal_action, optimal_action)
    
    
    # removing the optimal action from action_indicies so the function can output another one.
    action_indices.pop(optimal_action_index)
    
    random_action_index_not_optimal = np.random.choice(action_indices)
    
    random_action_not_optimal = possible_actions[random_action_index_not_optimal]
    
    return (optimal_action,random_action_not_optimal)

def update_q_table(old_pos: tuple[int,int], action: tuple[int,int], new_pos: tuple[int,int], possible_actions: list[tuple[int,int]],environment: tuple[tuple[int,int]], environment_x_length: int, environment_y_length: int, walls: list[tuple[int,int]], q_table: tuple[tuple[int,int]], alpha:float, gamma: float) -> list[float,int,int, bool]:
    """
    Updates the Q table.
    Output is a list of the following: new_q_table, new q value, old_pos_q_table_index,new_pos_q_table_index, movement_valid
    
    Takes in the coordiates for new_pos even though this can be calcuaed from starting coords and action moved, I'll already have the next coords from using the coordinates_after_moving function, so this will save redundant computation.), and the q table.
    
    It calls get_reward to determine the reward for making that move, and uses the Q(state,action) ML formula {{  Q(state, action) = Q(state, action) + α × [reward + γ × max(Q(next_state, all_possible_actions_from_that_state)) - Q(state, action)]  }}, then the function updates the row of the Q table with that reward value.
    
    α (alpha) - The learning rate, typically 0.1
    This controls how much we update based on new information. Think of it as a volume knob:

    α = 1.0: Completely replace old estimate with new information
    α = 0.1: Gently blend new information into existing estimate (10% new, 90% old)
    α = 0.0: Don't learn at all (keep old value)
    
    """
    
    reward_calc = get_reward(old_pos, action,possible_actions,environment,walls)
    reward = reward_calc[0]
    movement_valid = reward_calc[2]
    
    q_table_width = len(q_table[0])
    
    old_pos_q_table_index = coordinates_to_q_table_index(old_pos,environment_x_length,environment_y_length,q_table_width)
    new_pos_q_table_index = coordinates_to_q_table_index(new_pos,environment_x_length,environment_y_length,q_table_width)
    
    print(f"Action being looked up: {action}")
    print(f"Possible actions list: {possible_actions}")
    print(f"Trying to find index...")
    action_index = possible_actions.index(action)
    print(f"Found index: {action_index}")
    
    old_q_value = q_table[old_pos_q_table_index][action_index]
    
    learning_adjustment = alpha * (reward + (gamma * max(q_table[new_pos_q_table_index]) - old_q_value ))
    
    new_q_value = old_q_value + learning_adjustment
    
    q_table[old_pos_q_table_index][action_index] = new_q_value
    
    return [new_q_value, old_pos_q_table_index,new_pos_q_table_index, movement_valid]


# Create game_loop_learning that makes it so the agent moves through the enviornment using choose_action and update_q_table. Each move is stored in a list of moves. After all those calculations are done, if rendering = 'pygame', the function calls game_loop_manual using that list of moves to render the agent's moves using draw_agent. After all the moves, print the q table.

def game_loop_learning(actions_list: list, action_limit: int, possible_actions: list[tuple[int,int]],environment: tuple[tuple[int,int]], environment_x_length: int, environment_y_length: int, start: tuple[int,int], goals: list[tuple[int,int]], walls: list[tuple[int,int]], object_coloring: dict, color_for_background: tuple[int],q_table: tuple[tuple[int,int]], epsilon: float, alpha:float, gamma: float, rendering: str):
    """Makes it so the agent moves through the enviornment using choose_action and update_q_table. Each action is stored in a list of actions. After all those calculations are done, if rendering = 'pygame', the function calls game_loop_manual using that list of moves to render the agent's actions using draw_agent. After all the actions, print the q table. Then, depending on rendering, it renders all actions"""
    
    chosen_actions_list = []
    current_pos = start
    action_counter = 0
    
    for action in range(0,action_limit):
        
        print(f"--------------/nThis is action {action_counter}")
        
        chosen_action = choose_action(current_pos,q_table,possible_actions,environment_x_length,environment_y_length,epsilon)[1]
    
        chosen_actions_list.append(chosen_action)
        print("Chosen action: {chosen_action}")

        next_pos_calc = coordinates_after_moving(current_pos, chosen_action, possible_actions,walls)
    
        new_pos = next_pos_calc[0]
        movement_valid = next_pos_calc[1]
        print(f"New position: {new_pos}")
    
        reward = get_reward(current_pos,chosen_action,possible_actions,environment,walls)[0]
        print(f"Reward: {reward}")
        
        update_q_table(current_pos, chosen_action, new_pos,possible_actions,environment,environment_x_length,environment_y_length,walls,q_table,alpha,gamma)
        print("Just updated Q table")
        
        current_pos = new_pos
        
        action_counter = action_counter + 1
        
    print(q_table)
    
    game_loop_manual(environment,start,walls,object_coloring, color_for_background, chosen_actions_list, possible_actions,rendering)

    return None

# LEFT OFF Now test game_loop_learning by using the function with all it's inputs
