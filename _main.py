from _inputs_file import *
from coords_and_movement_file import *
from environment_file import *
from rendering_file import *
from q_learning_file import *
from game_loop_file import *
from test_logic import *


# print(game_loop_manual(full_environment, start, walls, color_map, background_color, actions_to_execute,possible_actions,"pygame",cell_value_to_name_map))

game_loop_learning_one_run(actions_to_execute,action_limit,possible_actions,  full_environment,environment_column_count,environment_row_count,start,goals,walls,color_map,background_color,q_table,epsilon,alpha,gamma,"pygame", cell_value_to_name_map, cell_reward)

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

