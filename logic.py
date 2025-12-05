import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import math

# The order of the actions here determine which column of the q table mean what. For example, the action below at index 1 represents the q_table column with index 1.
possible_actions = [(1,0), (0,1),(-1,0),(0,-1),(0,0)]
action_limit = 3

actions_agent_can_move = len(possible_actions)
environment_y_length = 8
environment_x_length = 10

cell_y_length = 50
cell_x_length = 50
# opportunity to improve: the rendering is incorrect when cell_x_length != cell_y_length 

pixel_rendering_offset_x_from_top_left = 50
pixel_rendering_offset_y_from_top_left = 50
# opportunity to improve: the location of the agent is offset when pixel_rendering_offset_x_from_top_left != pixel_rendering_offset_y_from_top_left 

window_x_length = 800
window_y_length = 700
window_dimensions = (window_x_length,window_y_length)

background_color = (255,255,255)
wall_color = (20,20,20)
empty_cell_color = (100,100,100)

framerate = 10
delay_in_ms_for_framerate = int((1 / framerate) * 1000)

cell_name_to_value_map = {
    "wall": -2,
    "goal": 5,
    "empty": 0,
    "start": 1
}

cell_value_to_name_map = cell_value_to_name_map = {value: key for key, value in cell_name_to_value_map.items()}


cell_color_map = {
    "wall": (30,30,30),
    "goal": (255,215,0),
    "empty": (100,100,100),
    "start": (0,255,0)
    }


wall_value = cell_name_to_value_map["wall"]
goal_value = cell_name_to_value_map["goal"]
start_value = cell_name_to_value_map["start"]
empty_value = cell_name_to_value_map["empty"]

empty_maze = np.full((environment_y_length,environment_x_length), empty_value,dtype=int)


goals = [(8,8)]
start_list = [(1,1)]
start = start_list[0]

q_table_width = actions_agent_can_move
q_table = np.zeros((environment_y_length*environment_x_length,actions_agent_can_move), dtype=float)
epsilon = 0.3
alpha = 0.1
gamma = 0.9


'''
Plan for which functions to create to allow the agent to move across the environment.

1. A function that converts (x, y) to Q-table row index

2. A function that takes current position and action, returns new position

3. A function that checks if a move is valid (not hitting a wall)

4. A function that updates the Q-table after learning'''


def coordinates_to_q_table_index(coordinates: tuple[int, int], environment_x_length: int, environment_y_length: int, q_table_width: int) -> int:
    '''A function that converts (x, y) to Q-table row index. If the x and y coordinates are outside the environment's dimensions, the function returns -100'''
    
    y_coord = coordinates[0]
    x_coord = coordinates[1]
    
    if (y_coord >= environment_y_length) or (x_coord >= environment_x_length) or (y_coord < 0) or (x_coord < 0 ):
        return -100
    
    q_table_row_index = x_coord + y_coord * environment_x_length
    
    
    return q_table_row_index

'''
Here I learn how tuples work

print([5,5] + [0,1])
# results in [5,5,0,1]
'''

def add_walls(maze_grid, wall_cells: list[tuple]) -> ndarray:
    """Takes in a list of cells in the form a a tuple. The cells are in the form (x_coord, y_coord). The output is a modification of the maze array where every cell in list of wall cells sets the value of the maze to wall_value.
    
    Because of aliasing, this will not modify whatever maze_grid is inputted.
    """
    global cell_name_to_value_map
    
    # This makes sure that the orignal maze_grid isn't changed in memory.
    copied_maze_grid = maze_grid.copy()
    
    for cell in wall_cells:
        #get the x and y coords of the wall
        y_coord_of_wall = cell[0]
        x_coord_of_wall = cell[1]
        
        # now go into the maze_grid and find the place in it that is in the same place as the wall. Then set that equal to the wall_value
        copied_maze_grid[x_coord_of_wall][y_coord_of_wall] = cell_name_to_value_map["wall"]
    
    
    return copied_maze_grid

def add_goals(maze_grid, goal_cells: list[tuple]) -> ndarray:
    """Takes in a list of cells in the form a a tuple. The cells are in the form (x_coord, y_coord). The output is a modification of the maze array where every cell in list of wall cells sets the value of the maze to goal_value.
    
    Because of aliasing, this will not modify whatever maze_grid is inputted.
    """
    
    global cell_name_to_value_map
    
    # This makes sure that the orignal maze_grid isn't changed in memory.
    copied_maze_grid_with_goals = maze_grid.copy()
    
    for cell in goal_cells:
        #get the x and y coords of the wall
        y_coord_of_wall = cell[0]
        x_coord_of_wall = cell[1]
        
        # now go into the maze_grid and find the place in it that is in the same place as the wall. Then set that equal to the wall_value
        copied_maze_grid_with_goals[x_coord_of_wall][y_coord_of_wall] = cell_name_to_value_map["goal"]
    
    
    return copied_maze_grid_with_goals

def add_custom_object(maze_grid, cells_to_put_object_in: list[tuple], chosen_value) -> ndarray:
    """Takes in a list of cells in the form a a tuple. The cells are in the form (x_coord, y_coord). The output is a modification of the maze array where every cell in list of wall cells sets the value of the maze to chosen_value.
    
    Because of aliasing, this will not modify whatever maze_grid is inputted.
    """
    
    # This makes sure that the orignal maze_grid isn't changed in memory.
    copied_maze_grid_with_custom_objects = maze_grid.copy()
    
    for cell in cells_to_put_object_in:
        #get the x and y coords of the wall
        y_coord = cell[0]
        x_coord = cell[1]
        
        # now go into the maze_grid and find the place in it that is in the same place as the wall. Then set that equal to the wall_value
        copied_maze_grid_with_custom_objects[x_coord][y_coord] = chosen_value
    
    
    return copied_maze_grid_with_custom_objects

    '''By seeing that this evaluates to true, it's clear that add_custom_object is a generalized application of add_walls and add_gaols
    print(np.array_equal((add_walls(add_goals(maze,goals),walls)),add_custom_object(add_custom_object(maze, goals, goal_value),walls,wall_value)))

    print(add_walls(add_goals(maze,goals),walls))

    print(add_custom_object(add_custom_object(maze, goals, goal_value),walls,wall_value))
    '''

def coords_to_center_of_cell_in_pixels(coords: tuple[int, int]) -> tuple[int,int]:
    """This is used for drawing objects at the center of a cell. Takes in coordinates of the cell, and return the location in pixels of the cell's center"""
    
    global pixel_rendering_offset_y_from_top_left
    global pixel_rendering_offset_x_from_top_left
    global cell_x_length
    global cell_y_length
    
    center_in_pixels = (pixel_rendering_offset_y_from_top_left + coords[1]*cell_y_length+cell_y_length/2, pixel_rendering_offset_x_from_top_left + coords[0]* cell_x_length+cell_x_length/2)
    
    return center_in_pixels

# This function I have the scaffold but I haven't finished it yet. I need to see how the agent can be moved on the pygame display.
def move_agent(starting_coords: tuple[int,int], action_to_move: tuple[int,int], possible_actions: list[tuple[int,int]]) -> bool:
    """Taking in the agent's starting coordinates, and a action, this moves the agent, as long as it makes a valid move, meaning that the agent doesn't exit the environment or hit a wall. Returns a boolean of if the move was valid or not. This function draws the agent too, but doesn't render it."""
    
    from setup_environment import window
    global walls
    from setup_environment import agent_color
    from setup_environment import agent_width
    from setup_environment import agent_height


    coords_calc = coordinates_after_moving(starting_coords, action_to_move, possible_actions,walls)
    
    final_coords = coords_calc[0]
    movement_valid = coords_calc[1]
    
    if movement_valid == True:
        output_in_coords = final_coords
    else:
        output_in_coords = starting_coords
    
    # now i have to convert the output_in_coords to pixels
    coords_in_pixels = coords_to_center_of_cell_in_pixels((output_in_coords))
        
    pg.draw.circle(window, agent_color, coords_in_pixels, agent_width, agent_height) 
    
    return movement_valid

def draw_agent(coords: tuple[int,int]) -> bool:
    """Taking in the agent's starting coordinates, this draws the agent at those coordinates. Returns a boolean of if the move was valid or not, meaning if the coordinates are outside the environment or on top of a wall. This function draws the agent too, but doesn't render it."""
    
    from setup_environment import window
    global walls
    from setup_environment import agent_color
    from setup_environment import agent_width
    from setup_environment import agent_height

    coords_in_pixels = coords_to_center_of_cell_in_pixels((coords))
        
    pg.draw.circle(window, agent_color, coords_in_pixels, agent_width, agent_height) 
    
    return None


def add_walls_on_border(grid: tuple[tuple[int,int]], environment_x_length: int, environment_y_length: int, wall_value: int) -> list[tuple[tuple[int,int]], list[tuple[int,int]]]:
    
    border_cells_top_edge =  [(x, 0) for x in range(environment_x_length)]
    # print(f"Border cells on top: {border_cells_top_edge}")
    
    border_cells_left_edge =  [(0, x) for x in range(environment_y_length)]
    # print(f"Border cells on left: {border_cells_left_edge}")

    
    border_cells_bottom_edge =  [(environment_y_length-1, x) for x in range(environment_x_length)]
    # print(f"Border cells on bottom: {border_cells_bottom_edge}")

    border_cells_right_edge =  [(x, environment_x_length-1) for x in range(environment_y_length)]
    # print(f"Border cells on right: {border_cells_right_edge}")

    
    border_cells = (border_cells_left_edge+ border_cells_right_edge+ border_cells_bottom_edge+ border_cells_top_edge)
    
    # print(f"All border cells: {border_cells}")

    grid = add_custom_object(grid, border_cells, wall_value)
    
    return (grid,border_cells)

# This is the creation of the environment the agent will move through
full_environment = add_custom_object(empty_maze, goals, goal_value)
full_environment = add_custom_object(full_environment,start_list,start_value)
add_walls_to_border_calc = add_walls_on_border(full_environment,environment_x_length,environment_y_length,wall_value)
full_environment = add_walls_to_border_calc[0]
border_cells = add_walls_to_border_calc[1]
walls = [*border_cells, (4,4)]
# print(f"Walls: {walls}")
full_environment = add_custom_object(full_environment,walls,wall_value)
print(full_environment)

def coordinates_after_moving(coordinates: tuple[int, int], action: tuple[int,int], possible_actions: list[tuple[int,int]],walls: list[tuple[int,int]]) -> tuple[tuple[int, int],bool]:
    """A function that takes current position and action, returns a tuple. The first item of the tuple is the new position as a tuple of (x,y), and the second item in that tuple is whether or not it's a valid move, meaning if the agent hits a wall or exists the environment. 
    
    In short, the function returns the new coordinates, but if either the start or next coords are invalid (outside the environment, or if the next coords cause the agent to hit a wall), the the output is the original coordinates and false.
    
    If both the starting and next coorindates are valid, then the output is the next coordinates and true.
    
    If the inputted coordinates are outside the environment's borders, the function returns the original coordinates and false.
    
    If the movement takes the agent outside the environment, the function returns the original coordinates and false.
    
    If the movement takes the agent to hit a wall, the function returns the original coordinates and false."""
    
    global environment_x_length
    global environment_y_length
    
    y_coord = coordinates[0]
    x_coord = coordinates[1]
    output_valid = False
    
    if (y_coord > environment_y_length) or (x_coord > environment_x_length) or (y_coord < 0) or (x_coord < 0 ):
        return (coordinates, output_valid)
    
    if not(action in possible_actions):
        return (coordinates,output_valid)
    
    new_y_coord = y_coord + action[0]
    new_x_coord = x_coord + action[1]
    
    if (new_y_coord >= environment_y_length) or (new_x_coord >= environment_x_length) or (new_y_coord < 0) or (new_x_coord < 0 ):
        return (coordinates, output_valid)
    
    new_coords = (new_y_coord, new_x_coord) 
    
    if new_coords in walls:
        return (coordinates, output_valid)

    output_valid = True
    return(new_coords, output_valid)

# So first make a function object_at_coords which takes in coordinates and the grid and tells you what object is at those coordinates.

def object_at_coords(coords: tuple[int,int], grid: tuple[tuple[int,int]]) -> str:
    """takes in coordinates and the grid and returns the name of the object at those coordinates."""
    
    global cell_value_to_name_map
    
    y_coord = coords[0]
    x_coord = coords[1]
    
    cell_value = grid[x_coord][y_coord]
    
    cell_name = cell_value_to_name_map[cell_value]
    
    return cell_name


def adjacent_coords(start_coords: tuple[int, int], action: tuple[int,int], ) -> tuple[int, int]:
    """Takes in coordinates and a action, and outputs the next coordinates.
    
    This is different from coordinates_after_move because this function outputs the next coordinates even if there is a wall or boundary.  
"""
    
    y_coord = start_coords[0]
    x_coord = start_coords[1]
        
    new_y_coord = y_coord + action[0]
    new_x_coord = x_coord + action[1]
        
    new_coords = (new_y_coord, new_x_coord) 

    return new_coords


