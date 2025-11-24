from logic import *
from setup_environment import *

moves = ["down", "down", "down", "right", "right","right","up","up","up"]

# Game loop

agent_coords = start
draw_grid_and_background(full_environment, cell_color_map, background_color)
draw_agent(agent_coords)
pg.display.flip()
pg.time.delay(delay_in_ms_for_framerate)

for move in moves:
    draw_grid_and_background(full_environment, cell_color_map, background_color)
    coords_calc = coordinates_after_moving(agent_coords,move,walls)
    next_coords = coords_calc[0]
    movement_valid = coords_calc[1] 
    draw_agent(next_coords)
    pg.display.flip()
    agent_coords = next_coords
    pg.time.delay(delay_in_ms_for_framerate)
    
    
    
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