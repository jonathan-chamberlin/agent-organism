from logic import *
from reward import *
from setup_environment import *
from test_logic import *


moves = ["down","down","down", "right","right", "right","up","up","up","right","right","right","right","right","right","down","down","down","down","down","down","down","left","left","left","down","down","right","right","right"]





print(game_loop_manual(full_environment, start, walls, cell_color_map, background_color, moves,"pygame"))


# LEFT OFF.
# Modify game_loop_manual, add an input 'rendering': str as an input. It could speed up my simulation. Value of 'pygame' could make it so every frame is rendered on the pygame window. Value of 'print' means that the full environment is printed in the terminal every frame. And 'none' means that the function does all the calculations without printing or rendering anything.




running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

example_blank_q_table = np.full((4,4), 0,dtype=int)
for i in range(0,100):
    print(choose_action((0,0),example_blank_q_table,example_envionment_x_length, example_environment_y_length, 0))