from logic import *
from setup_environment import *



# LEFT OFF. Now I have to create the game loop, where the agent is drawn then rendered, then the agent is cleared, then it's new position is calculated, then the agent is drawn again and rendered again.

# Game loop
move_agent((0,0), "down")
pg.display.flip()


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False