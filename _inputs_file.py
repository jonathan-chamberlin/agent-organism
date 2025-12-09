import pytest
import numpy as np
import pygame as pg
import math
pg.init()


# The order of the actions here determine which column of the q table mean what. For example, the action below at index 1 represents the q_table column with index 1.
possible_actions = [(1,0), (0,1),(-1,0),(0,-1),(0,0)]
actions_to_execute = [(1,0)] #only when you want to have the agent execute a list of predetermined actions, you would plug this into game_loop_manual

runs = 6
action_limit = 8
framerate = 6

environment_row_count = 6
environment_column_count = 6

goals = [(4,4)]
start_list = [(1,1)]
walls_input = []

cell_y_length = 50
cell_x_length = 50

# Test environment
# environment_row_count = 6
# environment_column_count = 3
# goals = [(2,1)]
# start_list = [(1,1)]
# walls_input = []

epsilon = 0.3
alpha = 0.1
gamma = 0.1

Font = pg.font.Font(None, int(cell_x_length*0.5))

pixel_rendering_offset_x_from_top_left = 50
pixel_rendering_offset_y_from_top_left = 50

window_x_length = 1100
window_y_length = 900

agent_color = (0,0,255)
background_color = (255,255,255)

cell_name_to_value_map = {
    "wall": -2,
    "goal": 5,
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
    "wall": -10,
    "goal": 50,
    "empty": -1,
    "start": -2
}

cell_value_to_name_map = {value: key for key, value in cell_name_to_value_map.items()}