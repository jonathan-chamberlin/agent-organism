from _inputs_file import *
from environment_file import *
from rendering_file import *
from coords_and_movement_file import *
from q_learning_file import *

np.set_printoptions(precision=2, suppress=True)

def game_loop_manual(environment: tuple[tuple[int,int]], start: tuple[int,int], walls: list[tuple(int,int)], object_coloring: map, color_for_background, actions_to_do: list[tuple[int,int]], possible_actions: list[tuple[int,int]],rendering: str, cell_value_to_name_map: dict, q_table: tuple[tuple[float]], run_index: int, coords_of_run_action_message: tuple[int,int], list_of_rewards_for_each_action: list[float], runs:int) -> list[bool]:
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
    total_reward_for_run = np.sum(list_of_rewards_for_each_action)


    for action in actions_to_do:
        draw_grid_and_background(full_environment, color_map, background_color, cell_value_to_name_map)
        display_run_and_action_index(run_index,action_index, coords_of_run_action_message, Font)
        
        draw_agent(agent_coords)
        
        q_value_list = display_q_values_around_agent(agent_coords,possible_actions,environment_row_count,environment_column_count,q_table)
        
        display_message_and_value("Number of Runs: ",runs, (-0.5,environment_column_count))
        
        display_message_and_value("Total Reward for Run: ",total_reward_for_run, (0, environment_column_count))

        display_message_and_value("Reward for this action: ",list_of_rewards_for_each_action[action_index], (0.5,environment_column_count))
        
        
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

def game_loop_learning_one_run(action_limit: int, possible_actions: list[tuple[int,int]],environment: tuple[tuple[int,int]], environment_row_count: int, environment_column_count: int, start: tuple[int,int], goals: list[tuple[int,int]], walls: list[tuple[int,int]], object_coloring: dict, color_for_background: tuple[int],q_table: tuple[tuple[int,int]], epsilon: float, alpha:float, gamma: float, run_indexes_to_render: list[int], cell_value_to_name_map: dict, cell_reward:dict, run_index:int, coords_of_run_action_indexs: tuple[int,int], runs: int) -> tuple[list[tuple[int,int]], tuple[tuple[int,int]], float, list[float], int]:
    """Makes it so the agent moves through the enviornment using choose_action and update_q_table. Each action is stored in a list of actions."""
    
    chosen_actions_list = []
    current_pos = start
    action_counter = 0
    total_reward_gotten = 0
    list_of_rewards_for_each_action = []
    
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
        
        
        list_of_rewards_for_each_action.append(reward)
        

        
        update_q_table(current_pos, chosen_action, new_pos,possible_actions,environment,environment_row_count,environment_column_count,walls,q_table,alpha,gamma,cell_reward)
        # print(f"Just updated Q table with value {reward}")
        
        current_pos = new_pos
        
        action_counter = action_counter + 1

    total_reward_gotten = np.sum(list_of_rewards_for_each_action)
    
    try:
        action_index_agent_first_touched_goal = list_of_rewards_for_each_action.index(cell_reward["goal"])
    except ValueError:
        action_index_agent_first_touched_goal = None

    
    # print(f"Run_index: {run_index}. Action index agent first touched goal: {action_index_agent_first_touched_goal}")
    # print(f"The rewards for each action were {list_of_rewards_for_each_action}")
    
    # print(f"The actions taken were {chosen_actions_list}")
    # print(f"This is the q table: {q_table}")
    
    if (run_index in run_indexes_to_render):
        game_loop_manual(environment,start,walls,object_coloring, color_for_background, chosen_actions_list, possible_actions,"pygame", cell_value_to_name_map,q_table, run_index,coords_of_run_action_indexs, list_of_rewards_for_each_action,runs)

    # print(f"total_reward_gotten: {total_reward_gotten}")
    
    return (chosen_actions_list, q_table, total_reward_gotten,list_of_rewards_for_each_action, action_index_agent_first_touched_goal)

def game_loop_learning_multiple_runs(runs: int, action_limit: int, possible_actions: list[tuple[int,int]],environment: tuple[tuple[int,int]], environment_row_count: int, environment_column_count: int, start: tuple[int,int], goals: list[tuple[int,int]], walls: list[tuple[int,int]], object_coloring: dict, color_for_background: tuple[int],q_table: tuple[tuple[int,int]], epsilon: float, alpha:float, gamma: float, run_indexes_to_render: list[int], cell_value_to_name_map: dict, cell_reward:dict, coords_of_run_action_indexs: tuple[int,int]):
    """Exectutes game_loop_learning_one_run multiple times, based on the number of runs"""
    
    run_index = 0
    list_of_rewards_gotten = []
    array_of_rewards_for_all_runs = []
    
    for i in range(0,runs):
        
        calculation = game_loop_learning_one_run(action_limit, possible_actions,environment, environment_row_count, environment_column_count, start,goals,walls,object_coloring,color_for_background,q_table,epsilon,alpha,gamma,run_indexes_to_render,cell_value_to_name_map,cell_reward, run_index, coords_of_run_action_indexs,runs)
        
        chosen_actions_list = calculation[0]
        reward_gotten_for_run = calculation[2]
        list_of_rewards_for_actions = calculation[3]
        
        action_index_agent_first_touched_goal = calculation[4]
        
        
        array_of_rewards_for_all_runs.append(list_of_rewards_for_actions)
        run_index = run_index + 1

    return (q_table, array_of_rewards_for_all_runs)