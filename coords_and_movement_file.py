from _inputs_file import *
from rendering_file import cell_value_to_name_map

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

def object_at_coords(coords: tuple[int,int], grid: tuple[tuple[int,int]]) -> str:
    """takes in coordinates and the grid and returns the name of the object at those coordinates."""
    
    global cell_value_to_name_map
    
    row_index = coords[0]
    column_index = coords[1]
    
    cell_value = grid[row_index][column_index]
    
    cell_name = cell_value_to_name_map[cell_value]
    
    return cell_name

def coordinates_after_moving(coordinates: tuple[int, int], action: tuple[int,int], possible_actions: list[tuple[int,int]],walls: list[tuple[int,int]]) -> tuple[tuple[int, int],bool]:
    """A function that takes current position and action, returns a tuple. The first item of the tuple is the new position as a tuple of (x,y), and the second item in that tuple is whether or not it's a valid move, meaning if the agent hits a wall or exists the environment. 
    
    In short, the function returns the new coordinates, but if either the start or next coords are invalid (outside the environment, or if the next coords cause the agent to hit a wall), the the output is the original coordinates and false.
    
    If both the starting and next coorindates are valid, then the output is the next coordinates and true.
    
    If the inputted coordinates are outside the environment's borders, the function returns the original coordinates and false.
    
    If the movement takes the agent outside the environment, the function returns the original coordinates and false.
    
    If the movement takes the agent to hit a wall, the function returns the original coordinates and false."""
    
    global environment_column_count
    global environment_row_count
    
    row_index = coordinates[0]
    column_index = coordinates[1]
    output_valid = False
    
    if (row_index >= environment_row_count) or (column_index >= environment_column_count) or (row_index < 0) or (column_index < 0 ):
        return (coordinates, output_valid)
    
    if not(action in possible_actions):
        return (coordinates,output_valid)
    
    new_row_index = row_index + action[0]
    new_column_index = column_index + action[1]
    
    if (new_row_index >= environment_row_count) or (new_column_index >= environment_column_count) or (new_row_index < 0) or (new_column_index < 0 ):
        return (coordinates, output_valid)
    
    new_coords = (new_row_index, new_column_index) 
    
    if new_coords in walls:
        return (coordinates, output_valid)

    output_valid = True
    return(new_coords, output_valid)

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

def agent_stays_inside_environment(starting_coords: tuple[int,int], action_to_move: tuple[int,int], environment_row_count: int, environment_column_count: int) -> bool:
    """Given a starting position and an action, the function returns true if the agent's new position would be inside the environment, and false if outside the environment."""
    
    row_index = starting_coords[0]
    column_index = starting_coords[1]
    
    new_row_index = row_index + action_to_move[0]
    new_column_index = column_index + action_to_move[1]
    
    if (new_row_index >= environment_row_count) or (new_column_index >=environment_column_count) or (new_row_index < 0) or (new_column_index < 0):
        return False
    
    return True