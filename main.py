from logic import *
from setup_environment import *

moves = ["right","right","right"]

print(game_loop(full_environment, start, walls, cell_color_map, background_color, moves))

# left off: Since I have a working game loop and way to 

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False