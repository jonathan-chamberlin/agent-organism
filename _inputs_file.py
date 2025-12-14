import pytest
import numpy as np
import pygame as pg
import math
pg.init()

# The order of the actions here determine which column of the q table mean what. For example, the action below at index 1 represents the q_table column with index 1.
possible_actions = [(1,0), (0,1),(-1,0),(0,-1),(0,0), (1,1),(1,-1),(-1,1),(-1,-1)]
actions_to_execute = [(1,0)] #only when you want to have the agent execute a list of predetermined actions, you would plug this into game_loop_manual

runs = 1000
action_limit = 200
framerate = 60


run_indexes_to_render = [*range(runs-3,runs-1)]



# setting custom environemnts
if True: 
    environment_row_count = 30
    environment_column_count = 30
    cell_y_length = 25
    cell_x_length = 25
    walls_input = [(2,2),(2,3),(3,2)]
    goals = [(28,28),(27,28),(26,28)]
    start_list = [(1,1)]


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