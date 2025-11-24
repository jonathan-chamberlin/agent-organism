from logic import *
from setup_environment import *



# LEFT OFF. Now I have to create the game loop, where the agent is drawn then rendered, then the agent is cleared, then it's new position is calculated, then the agent is drawn again and rendered again.

# Game loop




# first frame
draw_grid_and_background(full_environment, cell_color_map, background_color)
# draw_background(background_color)
# draw_grid(full_environment, cell_color_map)
draw_agent(start)

draw_agent((3,3))
pg.display.flip()


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False