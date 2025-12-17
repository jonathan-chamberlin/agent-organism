import pytest
import numpy as np
import pygame as pg
import math
pg.init()

# The order of the actions here determine which column of the q table mean what. For example, the action below at index 1 represents the q_table column with index 1.
possible_actions = [(1,0), (0,1),(-1,0),(0,-1),(0,0), (1,1),(1,-1),(-1,1),(-1,-1)]
actions_to_execute = [(1,0)] #only when you want to have the agent execute a list of predetermined actions, you would plug this into game_loop_manual



simulation_option = 2
recording = True

# setting custom environemnts
if simulation_option == 2:
    framerate = 50
    runs = 1000
    run_indexes_to_render = [0,250,500,999]
    
    action_limit = 400
    environment_row_count = 25
    environment_column_count = 25
    cell_y_length = 30
    cell_x_length = 30
    
    goals = [(10,8)]
    start_list = [(1,1)]
    
    walls_input = [
    # Dense vertical walls creating narrow passages
    (1,3), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (9,3),
    (3,5), (4,5), (5,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5),
    (1,7), (2,7), (3,7), (4,7), (5,7), (6,7),
    (8,7), (9,7), (10,7), (12,7),
    (3,9), (4,9), (5,9), (6,9), (7,9),
    (9,9), (10,9), (11,9), (12,9), (13,9), (14,9), (15,9),
    (1,11), (2,11), (3,11), (4,11), (5,11), (6,11), (7,11),
    (9,11), (10,11), (11,11), (12,11), (13,11),
    (2,13), (3,13), (4,13), (5,13), (6,13), (7,13), (8,13),
    (10,13), (11,13), (12,13), (13,13), (14,13),
    (1,15), (2,15), (3,15), (4,15), (5,15),
    (7,15), (8,15), (9,15), (10,15), (11,15), (12,15),
    (14,15), (15,15), (17,15), (16,13), (18,15),
    (3,17), (4,17), (5,17), (6,17), (7,17), (8,17),
    (10,17), (11,17), (12,17), (13,17), (14,17), (15,17),
    (1,19), (2,19), (3,19), (4,19), (5,19), (6,19),
    (8,19), (9,19), (10,19), (11,19), (12,19), (13,19), (14,19),
    (16,19), (17,19), (18,19), (20,19),
    (3,21), (4,21), (5,21), (6,21), (7,21), (8,21), (9,21),
    (11,21), (12,21), (13,21), (14,21), (15,21), (16,21),
    (18,21), (19,21), (20,21), (21,21), (22,21),
    
    # Horizontal walls creating complexity
    (11,1), (11,2), (11,3), (11,4),
    (7,2), (7,3), (7,5),
    (13,4), (13,5), (13,6), (13,7), (13,8),
    (15,1), (15,2), (15,3), (15,4), (15,5), (15,6),
    (9,6), (9,7), (9,8), (9,9), (9,10),
    (15,8), (15,9), (15,10), (15,11),
    (17,6), (17,7), (17,8), (17,9), (17,10), (17,11),
    (19,2), (19,3), (19,4), (19,5), (19,6),
    (11,12), (11,13), (11,15),
    (13,10), (13,11), (13,12), (13,13),
    (15,13), (15,15), (15,16),
    (17,13), (17,14), (17,15), (17,16), (17,17),
    (19,8), (19,9), (19,10), (19,11), (19,12),
    (21,4), (21,5), (21,6), (21,7), (21,8), (21,9),
    (21,11), (21,12), (21,13), (21,14),
    (19,15), (19,16), (19,17),
    (21,17), (21,18), (21,19), (21,20),
    (23,6), (23,7), (23,8), (23,9), (23,10),
    (23,13), (23,14), (23,15), (23,16),
    (23,19), (23,20), (23,21), (23,22),
    
    # Interior obstacles and blocks
    (5,10), (6,10), (5,11), (6,11),
    (7,18), (8,18), (7,19), (8,19),
    (13,17), (14,17), (13,18), (14,18),
    (18,2), (18,3), (19,3),
    (16,20), (17,20), (18,20),
    (22,2), (22,3), (23,2),
    (20,23), (21,23), (22,23)]
if simulation_option == 0: 
    framerate = 100
    runs = 5000
    action_limit = 225
    environment_row_count = 30
    environment_column_count = 30
    cell_y_length = 25
    cell_x_length = 25
    walls_input = [(2,2),(2,3),(3,2)]
    goals = [(28,28),(27,28),(26,28)]
    start_list = [(1,1)]
if simulation_option == 1:
    framerate = 100

    runs = 400
    run_indexes_to_render = [0,50,100,150,200,250,300,350, 399]
    
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
if simulation_option == 3: 
    run_indexes_to_render = [0,1]
    framerate = 20
    runs = 2
    action_limit = 10
    environment_row_count = 6
    environment_column_count = 6
    cell_y_length = 60
    cell_x_length = 60
    walls_input = []
    goals = [(2,2)]
    start_list = [(1,1)]



# Test environment
# environment_row_count = 6
# environment_column_count = 3
# goals = [(2,1)]
# start_list = [(1,1)]
# walls_input = []

epsilon = 0.01
alpha = 0.1
gamma = 0.99 #Discount factor. Reward future rewards x% as much as rewards now

Font = pg.font.Font(None, int(cell_x_length*0.9))
Font_q_values = pg.font.Font(None, int(cell_x_length*0.5))

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