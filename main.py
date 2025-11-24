from logic import *
from setup_environment import *





running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False