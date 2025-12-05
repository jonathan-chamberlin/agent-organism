from logic import *
from reward import *
from setup_environment import *
from test_logic import *


actions_to_execute = [(1,0),(1,0),(1,0),(0,1),(0,1), (0,1),(-1,0),(-1,0),(-1,0),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(0,-1),(0,-1),(0,-1),(1,0),(1,0),(0,1),(0,1),(0,1)]

# print(game_loop_manual(full_environment, start, walls, cell_color_map, background_color, actions_to_execute,possible_actions,"pygame"))

game_loop_learning(actions_to_execute,action_limit,possible_actions,  full_environment,environment_column_count,environment_row_count,start,goals,walls,cell_color_map,background_color,q_table,epsilon,alpha,gamma,"pygame", cell_value_to_name_map)



running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

