import pytest
import numpy as np
import pygame as pg
import math
pg.init()

# The order of the actions here determine which column of the q table mean what. For example, the action below at index 1 represents the q_table column with index 1.
possible_actions = [(1,0), (0,1),(-1,0),(0,-1),(0,0), (1,1),(1,-1),(-1,1),(-1,-1)]
actions_to_execute = [(1,0)] #only when you want to have the agent execute a list of predetermined actions, you would plug this into game_loop_manual


framerate = 60


# setting custom environemnts
if False: 
    runs = 5000
    action_limit = 225
    environment_row_count = 30
    environment_column_count = 30
    cell_y_length = 25
    cell_x_length = 25
    walls_input = [(2,2),(2,3),(3,2)]
    goals = [(28,28),(27,28),(26,28)]
    start_list = [(1,1)]
if True:
    runs = 400
    action_limit = 250
    environment_row_count = 15
    environment_column_count = 15
    cell_y_length = 60
    cell_x_length = 60
    
    goals = [(10,8)]
    start_list = [(1,1)]
    
    walls_input = [
    # Vertical wall segments
    (2,3), (3,3), (4,3), (5,3),
    (2,6), (3,6), (4,6), (5,6), (6,6), (7,6),
    (2,9), (3,9), (4,9), (5,9), (6,9),
    (8,3), (9,3), (10,3), (11,3), (12,3),
    (8,9), (9,9), (10,9), (11,9),(1,6),
    
    # Horizontal wall segments  
    (6,1), (6,2), (6,3), (6,4),
    (3,7), (3,8), (3,9), (3,10), (3,11),
    (9,4), (9,5), (9,6), (9,7),
    (12,6), (12,7), (12,8), (12,9), (12,10),
    
    # Internal obstacles
    (5,11), (6,11), (7,11), (8,11),
    (10,1), (10,2), (11,1), (11,2),
    (7,13), (8,13), (9,13),
    (13,3), (13,4),(9,8)] 

run_indexes_to_render = [0,50,100,150,200,250,300,350, 399]

# custom wall layouts. Set them equal to walls_input to activate them





# Test environment
# environment_row_count = 6
# environment_column_count = 3
# goals = [(2,1)]
# start_list = [(1,1)]
# walls_input = []

epsilon = 0.01
alpha = 0.1
gamma = 0.99 #Discount factor. Reward future rewards x% as much as rewards now

Font = pg.font.Font(None, int(cell_x_length*0.5))

pixel_rendering_offset_x_from_top_left = 50
pixel_rendering_offset_y_from_top_left = 50
coords_of_run_action_message = (0,0)

window_x_length = 1300
window_y_length = 900

agent_color = (0,0,255)
background_color = (0,0,0)

cell_name_to_value_map = {
    "wall": 3,
    "goal": 4,
    "empty": 0,
    "start": 1
}

color_map = {
    "wall": (30,30,30),
    "goal": (255,215,0),
    "empty": (100,100,100),
    "start": (0,255,0)
    }

cell_reward = {
    "wall": -2,
    "goal": 5,
    "empty": -1,
    "start": -1
}

cell_value_to_name_map = {value: key for key, value in cell_name_to_value_map.items()}