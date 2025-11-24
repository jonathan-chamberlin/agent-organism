from logic import *
from setup_environment import *

moves = ["down", "down", "down", "right", "right","right","up","up","up","right","right"]

game_loop(full_environment, start, walls, cell_color_map, background_color, moves)

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False