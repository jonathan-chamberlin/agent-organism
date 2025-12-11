from _inputs_file import *
from environment_file import *
from rendering_file import *
from coords_and_movement_file import *
from q_learning_file import *

np.set_printoptions(precision=2, suppress=True)

def game_loop_manual(environment: tuple[tuple[int,int]], start: tuple[int,int], walls: list[tuple(int,int)], object_coloring: map, color_for_background, actions_to_do: list[tuple[int,int]], possible_actions: list[tuple[int,int]],rendering: str, cell_value_to_name_map: dict, q_table: tuple[tuple[float]], run_index: int, coords_of_run_action_message: tuple[int,int]) -> list[bool]:
    """Takes in a bunch of inputs, and for every move it draws the full environment (grid and background), then draws the agent, then calculates its next move and position, then checks if that next position would be valid, then draws it, and renders it. 
    
    It returns a list of booleans representing what MOVES were valid, NOT positions. So if the agent starts on a valid square, and the first move (index 0) is to an invalid square, then the output of this function will be [False, ...].
    If the agent is on a valid square, and the first move (index 0) is to a valid square, then it's second move (index 1) is to an invalid square, the output of this function will be [True, False, ...]
    
    Input 'rendering': str. Value of 'pygame' could make it so every frame is rendered on the pygame window. Value of 'print' means that the agent coords are printed every frame. And 'none' means that the function does all the calculations without printing or rendering anything.

    Improvement: The data for start and walls is inside the full_environment variable that is inputted into this function, so teh inputs of start and walls are redundant. I could write code to look at the environment input and find which cell has the start_value and store those coords as a tuple, and which cells have the wall_value and store that as a list of tuples, and then use those variables in the code below.
    """
    from rendering_file import display_q_values_around_agent
    from rendering_file import display_run_and_action_index
    
    rendering_pygame_value = "pygame"
    movement_valid_list = []

    agent_coords = start
    rendering_print_value = "print"
    action_index = 0

    for action in actions_to_do:
        draw_grid_and_background(full_environment, color_map, background_color, cell_value_to_name_map)
        display_run_and_action_index(run_index,action_index, pixel_rendering_offset_x_from_top_left,pixel_rendering_offset_y_from_top_left, coords_of_run_action_message, Font)
        
        draw_agent(agent_coords)
        
        q_value_list = display_q_values_around_agent(agent_coords,possible_actions,environment_row_count,environment_column_count,q_table)
        
        # print(f"Current coordinates: {agent_coords}. q_value_list: {q_value_list}")
        
        coords_calc = coordinates_after_moving(agent_coords,action,possible_actions,walls)
        
        next_coords = coords_calc[0]
        # print(f"Next agent will go to {next_coords}")
        
        action_valid = coords_calc[1] 
        movement_valid_list.append(action_valid)
        
        pg.display.flip()
        pg.time.delay(delay_in_ms_for_framerate)
        
        agent_coords = next_coords
        action_index = action_index + 1
    
    # Stop light
    if rendering == rendering_pygame_value:
        pg.draw.rect(window,(255,0,0),(0.5*pixel_rendering_offset_x_from_top_left,0.5*pixel_rendering_offset_y_from_top_left,0.4*cell_x_length,0.4*cell_y_length))
        pg.display.flip()
    return movement_valid_list

def game_loop_learning_one_run(action_limit: int, possible_actions: list[tuple[int,int]],environment: tuple[tuple[int,int]], environment_row_count: int, environment_column_count: int, start: tuple[int,int], goals: list[tuple[int,int]], walls: list[tuple[int,int]], object_coloring: dict, color_for_background: tuple[int],q_table: tuple[tuple[int,int]], epsilon: float, alpha:float, gamma: float, rendering: str, cell_value_to_name_map: dict, cell_reward:dict, run_index:int, coords_of_run_action_indexs: tuple[int,int]) -> tuple[list[tuple[int,int]], tuple[tuple[int,int]], float]:
    """Makes it so the agent moves through the enviornment using choose_action and update_q_table. Each action is stored in a list of actions. After all those calculations are done, if rendering = 'pygame', the function calls game_loop_manual using that list of moves to render the agent's actions using draw_agent. After all the actions, print the q table. Then, depending on rendering, it renders all actions"""
    
    chosen_actions_list = []
    current_pos = start
    action_counter = 0
    reward_gotten = 0
    
    for action in range(0,action_limit):
        
        # print(f"--------------/nThis is action {action_counter}")
        
        chosen_action = choose_action(current_pos,q_table,possible_actions,environment_row_count,environment_column_count,epsilon)[1]
    
        chosen_actions_list.append(chosen_action)
        # print(f"Chosen action: {chosen_action}")

        next_pos_calc = coordinates_after_moving(current_pos, chosen_action, possible_actions,walls)
    
        new_pos = next_pos_calc[0]
        movement_valid = next_pos_calc[1]
        # print(f"New position: {new_pos}")
    
        reward = get_reward(current_pos,chosen_action,possible_actions,environment,walls, cell_reward)[0]
        
        reward_gotten= reward_gotten + reward
    
        
        update_q_table(current_pos, chosen_action, new_pos,possible_actions,environment,environment_row_count,environment_column_count,walls,q_table,alpha,gamma,cell_reward)
        # print(f"Just updated Q table with value {reward}")
        
        current_pos = new_pos
        
        action_counter = action_counter + 1

    
    # print(f"The actions taken were {chosen_actions_list}")
    # print(f"This is the q table: {q_table}")
    
    game_loop_manual(environment,start,walls,object_coloring, color_for_background, chosen_actions_list, possible_actions,rendering, cell_value_to_name_map,q_table, run_index,coords_of_run_action_indexs)

    print(f"reward_gotten: {reward_gotten}")
    return (chosen_actions_list, q_table, reward_gotten)

def game_loop_learning_multiple_runs(runs: int, action_limit: int, possible_actions: list[tuple[int,int]],environment: tuple[tuple[int,int]], environment_row_count: int, environment_column_count: int, start: tuple[int,int], goals: list[tuple[int,int]], walls: list[tuple[int,int]], object_coloring: dict, color_for_background: tuple[int],q_table: tuple[tuple[int,int]], epsilon: float, alpha:float, gamma: float, rendering: str, cell_value_to_name_map: dict, cell_reward:dict, coords_of_run_action_indexs: tuple[int,int]):
    """Exectutes game_loop_learning_one_run multiple times, based on the number of runs"""
    
    run_index = 0
    list_of_rewards_gotten = []
    
    for i in range(0,runs):
        
        reward_gotten = game_loop_learning_one_run(action_limit, possible_actions,environment, environment_row_count, environment_column_count, start,goals,walls,object_coloring,color_for_background,q_table,epsilon,alpha,gamma,rendering,cell_value_to_name_map,cell_reward, run_index, coords_of_run_action_indexs)[2]
        
        list_of_rewards_gotten.append(reward_gotten)

        run_index = run_index + 1

    print(list_of_rewards_gotten)
    return None