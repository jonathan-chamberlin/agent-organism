from _inputs_file import *
from coords_and_movement_file import *
from environment_file import *
from rendering_file import *
from q_learning_file import *
from game_loop_file import *
from test_logic import *
import matplotlib.pyplot as plt


long_calc = game_loop_learning_multiple_runs(runs, action_limit,possible_actions,  full_environment,environment_column_count,environment_row_count,start,goals,walls,color_map,background_color,q_table,epsilon,alpha,gamma,run_indexes_to_render, cell_value_to_name_map, cell_reward, coords_of_run_action_message,recording)

# print(f"Q table: {long_calc[0]}")
# print(f"array_of_rewards_for_all_runs: {long_calc[1]}")

list_of_action_index_agent_first_touched_goal= long_calc[2]

print(f"list_of_action_index_agent_first_touched_goal: {list_of_action_index_agent_first_touched_goal}")



all_run_indices = range(0,runs)
plt.figure(figsize=(14,5))
plt.bar(all_run_indices, list_of_action_index_agent_first_touched_goal)
plt.xlabel("Run index")
plt.ylabel("How many moves it took agent to reach goal")
plt.title("Run Index vs. How many moves it took agent to reach goal")
plt.show()



