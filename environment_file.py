from _inputs_file import *
import numpy as np
import math



wall_value = cell_name_to_value_map["wall"]
goal_value = cell_name_to_value_map["goal"]
start_value = cell_name_to_value_map["start"]
empty_value = cell_name_to_value_map["empty"]

empty_maze = np.full((environment_row_count,environment_column_count), empty_value,dtype=int)

start = start_list[0]

def add_custom_object(maze_grid, cells_to_put_object_in: list[tuple], chosen_value) -> ndarray:
    """Takes in a list of cells in the form a a tuple. The cells are in the form (row index, column index). The output is a modification of the maze array where every cell in list of wall cells sets the value of the maze to chosen_value.
    
    Because of aliasing, this will not modify whatever maze_grid is inputted.
    """
    
    # This makes sure that the orignal maze_grid isn't changed in memory.
    copied_maze_grid_with_custom_objects = maze_grid.copy()
    
    for cell in cells_to_put_object_in:
        
        row_index = cell[0]
        column_index = cell[1]
        
        # now go into the maze_grid and find the place in it that is in the same place as the wall. Then set that equal to the wall_value
        copied_maze_grid_with_custom_objects[row_index][column_index] = chosen_value
    
    return copied_maze_grid_with_custom_objects

    '''By seeing that this evaluates to true, it's clear that add_custom_object is a generalized application of add_walls and add_gaols
    print(np.array_equal((add_walls(add_goals(maze,goals),walls)),add_custom_object(add_custom_object(maze, goals, goal_value),walls,wall_value)))

    print(add_walls(add_goals(maze,goals),walls))

    print(add_custom_object(add_custom_object(maze, goals, goal_value),walls,wall_value))
    '''

def add_walls_on_border(grid: tuple[tuple[int,int]], environment_column_count: int, environment_row_count: int, wall_value: int) -> list[tuple[tuple[int,int]], list[tuple[int,int]]]:
    
    border_cells_top_edge =  [(0, x) for x in range(environment_column_count)]
    # print(f"Border cells on top: {border_cells_top_edge}")
    
    border_cells_left_edge =  [(x, 0) for x in range(environment_row_count)]
    # print(f"Border cells on left: {border_cells_left_edge}")

    
    border_cells_bottom_edge =  [(environment_row_count-1, x) for x in range(environment_column_count)]
    # print(f"Border cells on bottom: {border_cells_bottom_edge}")

    border_cells_right_edge =  [(x, environment_column_count-1) for x in range(environment_row_count)]
    # print(f"Border cells on right: {border_cells_right_edge}")

    
    border_cells = (border_cells_left_edge+ border_cells_right_edge+ border_cells_bottom_edge+ border_cells_top_edge)
    
    # print(f"All border cells: {border_cells}")

    grid = add_custom_object(grid, border_cells, wall_value)
    
    return (grid,border_cells)

# This is the creation of the environment the agent will move through
full_environment = add_custom_object(empty_maze, goals, goal_value)
add_walls_to_border_calc = add_walls_on_border(full_environment,environment_column_count,environment_row_count,wall_value)
full_environment = add_walls_to_border_calc[0]
border_cells = add_walls_to_border_calc[1]
walls = [*border_cells, *walls_input]
full_environment = add_custom_object(full_environment,walls,wall_value)
full_environment = add_custom_object(full_environment,start_list,start_value)
# print(full_environment)

