from logic import *
from reward import *
from setup_environment import *


moves = ["down","down","down", "right","right", "right","up","up","up","right","right","right","right","right","right","down","down","down","down","down","down","down","left","left","left","down","down","right","right","right"]

print(game_loop(full_environment, start, walls, cell_color_map, background_color, moves))


# LEFT OFF. Modify function game_loop that makes it so the agent moves through the enviornment using choose_action and update_q_table, and that the agent's moves are rendered using draw_agent. After all the moves, print the q table.



render_every_frame: bool

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False