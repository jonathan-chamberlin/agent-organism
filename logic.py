import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import math

directions_agent_can_move = 4
environment_x_length = 10
environment_y_length = 10

cell_x_length = 50
cell_y_length = 50
pixel_rendering_offset_x_from_top_left = 20
pixel_rendering_offset_y_from_top_left = 20

window_dimensions = (800,700)

background_color = (255,255,255)
wall_color = (20,20,20)
empty_cell_color = (100,100,100)

cell_value_to_name_map = {
    -1: "wall",
    2: "goal",
    0: "empty",
    1: "start"
}

cell_name_to_value_map = {
    "wall": -1,
    "goal": 2,
    "empty": 0,
    "start": 1
    
}

cell_color_map = {
    "wall": (30,30,30),
    "goal": (255,215,0),
    "empty": (100,100,100),
    "start": (0,255,0)
    }


empty_maze = np.zeros((environment_x_length,environment_y_length), dtype=int)

walls = [(5,0), (2,1), (5,1), (7,1), (8,1), (9,1), (0,2), (1,2), (2,2), (5,2), (7,2), (4,3), (5,3), (7,3), (1,4), (2,4), (3,4), (4,4), (7,4), (9,4), (1,5), (6,5), (7,5), (9,5), (1,6), (3,6), (4,6), (5,6), (6,6), (1,7), (8,7), (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (8,8), (8,9)]
goals = [(9,0), (9,9)]
start = [(0,0)]

wall_value = cell_name_to_value_map["wall"]
goal_value = cell_name_to_value_map["goal"]
start_value = cell_name_to_value_map["start"]


q_table_width = directions_agent_can_move

q_table = np.zeros((environment_x_length*environment_y_length,directions_agent_can_move), dtype=float)

direction_map = {
    "up": (-1,0),
    "down":(1,0),
    "left":(0,1),
    "right":(0,-1),
    "remain": (0,0)
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

def add_walls(maze_grid, wall_cells: list[tuple]) -> ndarray:
    """Takes in a list of cells in the form a a tuple. The cells are in the form (x_coord, y_coord). The output is a modification of the maze array where every cell in list of wall cells sets the value of the maze to wall_value.
    
    Because of aliasing, this will not modify whatever maze_grid is inputted.
    """
    global cell_name_to_value_map
    
    # This makes sure that the orignal maze_grid isn't changed in memory.
    copied_maze_grid = maze_grid.copy()
    
    for cell in wall_cells:
        #get the x and y coords of the wall
        x_coord_of_wall = cell[0]
        y_coord_of_wall = cell[1]
        
        # now go into the maze_grid and find the place in it that is in the same place as the wall. Then set that equal to the wall_value
        copied_maze_grid[y_coord_of_wall][x_coord_of_wall] = cell_name_to_value_map["wall"]
    
    
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
        x_coord_of_wall = cell[0]
        y_coord_of_wall = cell[1]
        
        # now go into the maze_grid and find the place in it that is in the same place as the wall. Then set that equal to the wall_value
        copied_maze_grid_with_goals[y_coord_of_wall][x_coord_of_wall] = cell_name_to_value_map["goal"]
    
    
    return copied_maze_grid_with_goals

def add_custom_object(maze_grid, cells_to_put_object_in: list[tuple], chosen_value) -> ndarray:
    """Takes in a list of cells in the form a a tuple. The cells are in the form (x_coord, y_coord). The output is a modification of the maze array where every cell in list of wall cells sets the value of the maze to chosen_value.
    
    Because of aliasing, this will not modify whatever maze_grid is inputted.
    """
    
    # This makes sure that the orignal maze_grid isn't changed in memory.
    copied_maze_grid_with_custom_objects = maze_grid.copy()
    
    for cell in cells_to_put_object_in:
        #get the x and y coords of the wall
        x_coord = cell[0]
        y_coord = cell[1]
        
        # now go into the maze_grid and find the place in it that is in the same place as the wall. Then set that equal to the wall_value
        copied_maze_grid_with_custom_objects[y_coord][x_coord] = chosen_value
    
    
    return copied_maze_grid_with_custom_objects

    '''By seeing that this evaluates to true, it's clear that add_custom_object is a generalized application of add_walls and add_gaols
    print(np.array_equal((add_walls(add_goals(maze,goals),walls)),add_custom_object(add_custom_object(maze, goals, goal_value),walls,wall_value)))

    print(add_walls(add_goals(maze,goals),walls))

    print(add_custom_object(add_custom_object(maze, goals, goal_value),walls,wall_value))
    '''

def coordinates_after_moving(coordinates: tuple[int, int], direction: str, walls: list[tuple[int,int]]) -> tuple[tuple[int, int],bool]:
    """A function that takes current position and direction, returns a tuple. The first item of the tuple is the new position as a tuple of (x,y), and the second item in that tuple is whether or not it's a valid move, meaning if the agent hits a wall or exists the environment. 
    
    If the inputted coordinates are outside the environment's borders, the function returns (-1.-1).
    
    If the movement takes the agent outside the environment, the function returns (-5,-5).
    
    If the movement takes the agent to hit a wall, the function returns (-2,-2)."""
    
    global q_table_width
    global environment_x_length
    global environment_y_length
    global direction_map
    
    
    
    x_coord = coordinates[0]
    y_coord = coordinates[1]
    output_valid = True
    
    # checking if the inputted coordinates are inside the environment
    if (x_coord >= environment_x_length) or (y_coord >= environment_y_length) or (x_coord < 0) or (y_coord < 0 ):
        output_valid = False
    
    # creating new coordinates
    
    new_x_coord = x_coord + direction_map[direction][0]
    new_y_coord = y_coord + direction_map[direction][1]
    
    # checking if the resulting coordinates are inside the environment
    if (new_x_coord >= environment_x_length) or (new_y_coord >= environment_y_length) or (new_x_coord < 0) or (new_y_coord < 0 ):
        output_valid = False
    
    # we create the new tuple (different place in memory) representing the new coordinates after the movement.
    
    new_coords = (new_x_coord, new_y_coord)
    
    # check if the new coords hit a wall
    if new_coords in walls:
        output_valid = False
    return (new_coords, output_valid)

def coords_to_center_of_cell_in_pixels(coords: tuple[int, int]) -> tuple[int,int]:
    """This is used for drawing objects at the center of a cell. Takes in coordinates of the cell, and return the location in pixels of the cell's center"""
    
    global pixel_rendering_offset_y_from_top_left
    global pixel_rendering_offset_x_from_top_left
    global cell_x_length
    global cell_y_length
    
    center_in_pixels = (pixel_rendering_offset_y_from_top_left + coords[1]*cell_y_length+cell_y_length/2, pixel_rendering_offset_x_from_top_left + coords[0]* cell_x_length+cell_x_length/2)
    
    return center_in_pixels

# This function I have the scaffold but I haven't finished it yet. I need to see how the agent can be moved on the pygame display.
def move_agent(starting_coords: tuple[int,int], direction_to_move: str) -> bool:
    """Taking in the agent's starting coordinates, and a direction, this moves the agent, as long as it makes a valid move, meaning that the agent doesn't exit the environment or hit a wall. Returns a boolean of if the move was valid or not. This function draws the agent too, but doesn't render it."""
    
    from setup_environment import window
    global walls
    from setup_environment import agent_color
    from setup_environment import agent_width
    from setup_environment import agent_height


    coords_calc = coordinates_after_moving(starting_coords, direction_to_move, walls)
    
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

    coords_calc = coordinates_after_moving(coords, "remain", walls)
    
    coords_to_draw_at = coords_calc[0]
    movement_valid = coords_calc[1]
    
    # now i have to convert the output_in_coords to pixels
    coords_in_pixels = coords_to_center_of_cell_in_pixels((coords_to_draw_at))
        
    pg.draw.circle(window, agent_color, coords_in_pixels, agent_width, agent_height) 
    
    return movement_valid

# This is the creation of the environment the agent will move through
environemnt_with_walls_and_goals = add_custom_object(add_custom_object(empty_maze, goals, goal_value),walls,wall_value)
full_environment = add_custom_object(environemnt_with_walls_and_goals,start,start_value)
print(full_environment)

# For display, I need to print out the maze array with the addition of walls and goals.

