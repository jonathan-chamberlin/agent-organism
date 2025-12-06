from inputs_file import *
import pytest
import numpy as np
import pygame as pg
import math

pg.init()

agent_width = int(math.floor(0.5* cell_x_length))
agent_height = int(math.floor(0.5* cell_y_length))

window_dimensions = (window_x_length,window_y_length)

window = pg.display.set_mode(window_dimensions)
pg.display.set_caption("Maze Robot Simulator")

delay_in_ms_for_framerate = int((1 / framerate) * 1000)

cell_value_to_name_map = {value: key for key, value in cell_name_to_value_map.items()}

"""Drawing one thing at a time
# draws a wall
pg.draw.rect(window, wall_color, (0,0,cell_y_length, cell_x_length))

# draws a empty cell
pg.draw.rect(window, empty_cell_color, (0,0,cell_y_length, cell_x_length))
"""

def draw_objects(objects: tuple[int,int], object_name: str, object_coloring: map) -> None:
    """The objects are rendered by looking at the int value in each cell, and finding the color for that value from the object_coloring map, then rending the cell in that color."""
    
    global cell_type_map
    global pixel_rendering_offset_x_from_top_left
    global pixel_rendering_offset_y_from_top_left
    global cell_x_length
    global cell_y_length
    
    for cell in objects:
        # convert cell cordinate locations to placement on the window using pixels
        row_index = cell[0]
        column_index = cell[1]
        
        row_position_in_pixels = row_index * cell_y_length + pixel_rendering_offset_y_from_top_left
        column_position_in_pixels = column_index * cell_x_length +pixel_rendering_offset_x_from_top_left
        
        # finds color of cell from the object_coloring_map
        object_color = object_coloring[object_name]
        
        # draws the cell
        pg.draw.rect(window, object_color, (column_position_in_pixels,row_position_in_pixels,cell_y_length, cell_x_length))

def coords_to_center_of_cell_in_pixels(coords: tuple[int, int]) -> tuple[int,int]:
    """This is used for drawing objects at the center of a cell. Takes in coordinates of the cell, and return the location in pixels of the cell's center"""
    
    global pixel_rendering_offset_y_from_top_left
    global pixel_rendering_offset_x_from_top_left
    global cell_x_length
    global cell_y_length
    
    center_in_pixels = (pixel_rendering_offset_y_from_top_left + coords[1]*cell_y_length+cell_y_length/2, pixel_rendering_offset_x_from_top_left + coords[0]* cell_x_length+cell_x_length/2)
    
    return center_in_pixels

def draw_agent(coords: tuple[int,int]) -> bool:
    """Taking in the agent's starting coordinates, this draws the agent at those coordinates. Returns a boolean of if the move was valid or not, meaning if the coordinates are outside the environment or on top of a wall. This function draws the agent too, but doesn't render it."""
    
    global window
    from inputs_file import agent_color
    global agent_width
    global agent_height

    coords_in_pixels = coords_to_center_of_cell_in_pixels((coords))
        
    pg.draw.circle(window, agent_color, coords_in_pixels, agent_width, agent_height) 
    
    return None

    """
    TEST. Does <1> produce the same as <2>?
    <both>
    example_walls_to_draw = [(0,0), (3,1),(2,3),(0,9),(9,0),(9,9)]
    </both>
    <1>
    for cell in example_walls_to_draw:
    
    # convert cell cordinate locations to placement on the window using pixels
    
    x_coord = cell[0]
    y_coord = cell[1]
    
    x_coord_in_pixels = x_coord * cell_x_length + pixel_rendering_offset_x_from_top_left
    y_coord_in_pixels = y_coord * cell_y_length +pixel_rendering_offset_y_from_top_left
    
    pg.draw.rect(window, wall_color, (x_coord_in_pixels,y_coord_in_pixels,cell_x_length, cell_y_length))
    </1>
    
    <2> 
    draw_objects(example_walls_to_draw,"wall",cell_color_map)
    </2>
    
    """

def draw_one_object(cell_coordinates: tuple[int,int], cell_type: str, object_color: tuple[int,int,int]) -> None:
    """Accepts the (x,y) coordinates of one object, and it's type, and draws it."""
    global cell_type_map
    global pixel_rendering_offset_x_from_top_left
    global pixel_rendering_offset_y_from_top_left
    global cell_x_length
    global cell_y_length
    
    # convert cell cordinate locations to placement on the window using pixels
    row_index = cell_coordinates[0]
    column_index = cell_coordinates[1]
    
    row_position_in_pixels = row_index * cell_y_length + pixel_rendering_offset_y_from_top_left
    column_position_in_pixels = column_index * cell_x_length +pixel_rendering_offset_x_from_top_left
    
    # draws the cell
    pg.draw.rect(window, object_color, (column_position_in_pixels,row_position_in_pixels,cell_x_length, cell_y_length))
    
    """TESTS:
    draw_one_object((5,5), "wall", cell_color_map["wall"])
    draw_one_object((6,6), "wall", cell_color_map["wall"])
    draw_one_object((4,5), "wall", cell_color_map["wall"])
    """

def draw_grid(grid: tuple[tuple[int,int]], object_coloring: map,cell_value_to_name_map: dict)-> None:
    """Accepts a grid, not a one-dimensional tuple. For each cell in the grid, it gets the cell's value. From that value, it looks up its name using the object_values map. Using the name, the function looks up the cell's coloring, then draws the object"""
    
    row_index = 0
    
    
    for row in grid:
        cell_index = 0
        for cell in row:
            cell_value = cell
            cell_name = cell_value_to_name_map[cell_value]
            cell_color = object_coloring[cell_name]
            draw_one_object((row_index, cell_index),cell_name,cell_color)
            
            cell_index = cell_index + 1
        
        row_index = row_index + 1

def draw_background(color: tuple[int,int,int]) -> None:
    """Accepts a color and draws the background in that color."""
    
    global window
    from inputs_file import window_x_length
    from inputs_file import window_y_length
    
    pg.draw.rect(window, color, (0,0,window_x_length, window_y_length))

def draw_grid_and_background(grid: tuple[tuple[int,int]], object_coloring: map, color_for_background, cell_value_to_name_map: dict) -> None:
        """Accepts a grid, not a one-dimensional tuple. It first draws the background, then draws the whole grid using draw_grid."""
        
        draw_background(color_for_background)
        draw_grid(grid, object_coloring, cell_value_to_name_map)

