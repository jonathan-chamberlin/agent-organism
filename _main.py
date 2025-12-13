from _inputs_file import *
from coords_and_movement_file import *
from environment_file import *
from rendering_file import *
from q_learning_file import *
from game_loop_file import *
from test_logic import *

# game_loop_learning_one_run(action_limit,possible_actions,  full_environment,environment_column_count,environment_row_count,start,goals,walls,color_map,background_color,q_table,epsilon,alpha,gamma,"pygame", cell_value_to_name_map, cell_reward)

game_loop_learning_multiple_runs(runs, action_limit,possible_actions,  full_environment,environment_column_count,environment_row_count,start,goals,walls,color_map,background_color,q_table,epsilon,alpha,gamma,"pygame", cell_value_to_name_map, cell_reward, coords_of_run_action_message)

print(f"Cell_name_to_value_map: {cell_name_to_value_map}")
print(f"Expect cell_name_to_value_map['goal'] to be ")


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

