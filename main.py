from logic import *
from setup_environment import *



# LEFT OFF. Now I have to create the game loop, where the agent is drawn then rendered, then the agent is cleared, then it's new position is calculated, then the agent is drawn again and rendered again.

# Game loop

moves = ["down", "down", "down", "right", "right","right","up","up","up"]

agent_coords = start
draw_grid_and_background(full_environment, cell_color_map, background_color)
draw_agent(agent_coords)
pg.display.flip()
pg.time.delay(200)

for move in moves:
    draw_grid_and_background(full_environment, cell_color_map, background_color)
    coords_calc = coordinates_after_moving(agent_coords,move,walls)
    next_coords = coords_calc[0]
    movement_valid = coords_calc[1] 
    draw_agent(next_coords)
    pg.display.flip()
    agent_coords = next_coords
    pg.time.delay(200)
    
    
    
"""
# first frame
draw_grid_and_background(full_environment, cell_color_map, background_color)
draw_agent(start)
pg.display.flip()
pg.time.delay(500)

# second frame
draw_grid_and_background(full_environment, cell_color_map, background_color)
coords_calc = coordinates_after_moving(start,"down",walls)
next_coords = coords_calc[0]
movement_valid = coords_calc[1] 
draw_agent(next_coords)
pg.display.flip()
pg.time.delay(500)

# third frame
draw_grid_and_background(full_environment, cell_color_map, background_color)
coords_calc2 = coordinates_after_moving(next_coords,"right",walls)
next_coords2 = coords_calc2[0]
movement_valid = coords_calc2[1] 
draw_agent(next_coords2)
pg.display.flip()
"""

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False