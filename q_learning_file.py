from inputs_file import *
from coords_and_movement_file import *
import random
import numpy as np
import math

q_table_width = actions_agent_can_move
q_table = np.zeros((environment_row_count*environment_column_count,actions_agent_can_move), dtype=float)

wall_reward = cell_reward["wall"]
goal_reward = cell_reward["goal"]
start_reward = cell_reward["start"]
empty_reward = cell_reward["empty"]

def coordinates_to_q_table_index(coordinates: tuple[int, int], environment_column_count: int, environment_row_count: int, q_table_width: int) -> int:
    '''A function that converts (x, y) to Q-table row index. If the x and y coordinates are outside the environment's dimensions, the function returns -100'''
    
    row_index = coordinates[0]
    column_index = coordinates[1]
    
    if (row_index >= environment_row_count) or (column_index >= environment_column_count) or (row_index < 0) or (column_index < 0 ):
        return -100
    
    q_table_row_index = column_index + row_index * environment_column_count
    
    
    return q_table_row_index

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

def choose_action(current_pos: tuple(int,int), q_table: tuple[tuple[int,int]], possible_actions: list[tuple[int,int]], environment_column_count: int, environment_row_count: int, epsilon: float) -> tuple[tuple[int,int]]: 
    """takes in the agent's current coordinates and the whole Q table, and just reads thh Q table and finds which value is the highest, and it has a 1-epsilon chance of picking the move with the highest q value (this is the explotation rate, or 1-epsilon, where epsilon is the exploration rate), and an epsilon likely to chose another move at random.

    Returns a tuple where the first string is the optimal action, and the second is an action which was randomly chosen using epsilon the exploration rate.
    
    If every value in the q table row is 0, then the agent picks a direction randomly.
    """
    
    q_table_width = len(q_table[0])
    
    q_table_index = coordinates_to_q_table_index(current_pos,environment_column_count, environment_row_count, q_table_width)
    
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

def update_q_table(old_pos: tuple[int,int], action: tuple[int,int], new_pos: tuple[int,int], possible_actions: list[tuple[int,int]],environment: tuple[tuple[int,int]], environment_column_count: int, environment_row_count: int, walls: list[tuple[int,int]], q_table: tuple[tuple[int,int]], alpha:float, gamma: float) -> list[float,int,int, bool]:
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
    
    old_pos_q_table_index = coordinates_to_q_table_index(old_pos,environment_column_count,environment_row_count,q_table_width)
    new_pos_q_table_index = coordinates_to_q_table_index(new_pos,environment_column_count,environment_row_count,q_table_width)
    
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


