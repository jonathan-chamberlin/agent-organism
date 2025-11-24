from logic import *
from setup_environment import *


move_agent((1,1), "down")

pg.display.flip()


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False