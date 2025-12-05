from logic import *

pg.init()

agent_color = (0,0,255)
agent_width = int(math.floor(0.5* cell_x_length))
agent_height = int(math.floor(0.5* cell_y_length))


window = pg.display.set_mode(window_dimensions)
pg.display.set_caption("Maze Robot Simulator")


# pg.draw.rect(window, background_color, (0,0,window_dimensions[0], window_dimensions[1]))

"""Drawing one thing at a time
# draws a wall
pg.draw.rect(window, wall_color, (0,0,cell_y_length, cell_x_length))

# draws a empty cell
pg.draw.rect(window, empty_cell_color, (0,0,cell_y_length, cell_x_length))
"""





# the grid is just an array with 0s
# the grid is updated to include walls and goals, and custom objects
# the grid is rendered by looking at the int value in each cell, and finding the color for that value, then rending the cell in that color.

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
# draw_objects(example_walls_to_draw,"wall",cell_color_map)

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
    from logic import window_x_length
    from logic import window_y_length
    
    pg.draw.rect(window, color, (0,0,window_x_length, window_y_length))

def draw_grid_and_background(grid: tuple[tuple[int,int]], object_coloring: map, color_for_background, cell_value_to_name_map: dict) -> None:
        """Accepts a grid, not a one-dimensional tuple. It first draws the background, then draws the whole grid using draw_grid."""
        
        draw_background(color_for_background)
        draw_grid(grid, object_coloring, cell_value_to_name_map)

def game_loop_manual(environment: tuple[tuple[int,int]], start: tuple[int,int], walls: list[tuple(int,int)], object_coloring: map, color_for_background, actions_to_do: list[tuple[int,int]], possible_actions: list[tuple[int,int]],rendering: str, cell_value_to_name_map: dict) -> list[bool]:
    """Takes in a bunch of inputs, and for every move it draws the full environment (grid and background), then draws the agent, then calculates its next move and position, then checks if that next position would be valid, then draws it, and renders it. 
    
    It returns a list of booleans representing what MOVES were valid, NOT positions. So if the agent starts on a valid square, and the first move (index 0) is to an invalid square, then the output of this function will be [False, ...].
    If the agent is on a valid square, and the first move (index 0) is to a valid square, then it's second move (index 1) is to an invalid square, the output of this function will be [True, False, ...]
    
    Input 'rendering': str. Value of 'pygame' could make it so every frame is rendered on the pygame window. Value of 'print' means that the agent coords are printed every frame. And 'none' means that the function does all the calculations without printing or rendering anything.

    
    Improvement: The data for start and walls is inside the full_environment variable that is inputted into this function, so teh inputs of start and walls are redundant. I could write code to look at the environment input and find which cell has the start_value and store those coords as a tuple, and which cells have the wall_value and store that as a list of tuples, and then use those variables in the code below.
    
    Worse improvement: The full environment is created from the variables of empty_maze, goals, goal_value),walls,wall_value, start_list,start_value. I could input all those things into this game loop function and have it run, but that would be too many inputs.
    """
    
    rendering_pygame_value = "pygame"
    movement_valid_list = []

    agent_coords = start
    rendering_print_value = "print"
    
    if rendering == rendering_pygame_value:
        draw_grid_and_background(environment, object_coloring, color_for_background,cell_value_to_name_map)
        draw_agent(agent_coords)
        pg.display.flip()
        pg.time.delay(delay_in_ms_for_framerate)

    for action in actions_to_do:
        if rendering == rendering_pygame_value:
            # clear and redraw environment
            draw_grid_and_background(full_environment, cell_color_map, background_color, cell_value_to_name_map)
        
        # find the next place the agent will go
        coords_calc = coordinates_after_moving(agent_coords,action,possible_actions,walls)
        next_coords = coords_calc[0]
        
        # determines if that next movement is valid
        action_valid = coords_calc[1] 
        movement_valid_list.append(action_valid)
        
        if rendering == rendering_print_value:
            print(next_coords)
        
        if rendering == rendering_pygame_value:
            # draws agent at the next coordinates
            draw_agent(next_coords)
            # renders everything
            pg.display.flip()
        # updates the next coordinates value since the agent is now there
        agent_coords = next_coords
        if rendering == rendering_pygame_value:
            # time control for framerate
            pg.time.delay(delay_in_ms_for_framerate)
    
    # Stop light
    if rendering == rendering_pygame_value:
        pg.draw.rect(window,(255,0,0),(0.5*pixel_rendering_offset_x_from_top_left,0.5*pixel_rendering_offset_y_from_top_left,0.4*cell_x_length,0.4*cell_y_length))
        pg.display.flip()
    return movement_valid_list


# drawing agent on start square before I created draw_agent
# circle_centered_on_start_coords = start
# agent_at_start_location = coords_to_center_of_cell_in_pixels(start[0])
# pg.draw.circle(window, agent_color, agent_at_start_location, agent_width, agent_height)

# renders everything
pg.display.flip()


