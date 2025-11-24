from logic import *
from setup_environment import *





# Game loop
move_agent((0,0), "down")
pg.display.flip()


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False