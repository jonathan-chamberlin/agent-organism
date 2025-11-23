from logic import *
from visual import *
from agent import *

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False