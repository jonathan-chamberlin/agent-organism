from logic import *

pg.init()

cell_x_length = 50
cell_y_length = 50
pixel_rendering_offset_x_from_top_left = 20
pixel_rendering_offset_y_from_top_left = 20

window_dimensions = (800,700)

background_color = (255,255,255)
wall_color = (20,20,20)
empty_cell_color = (100,100,100)

window = pg.display.set_mode(window_dimensions)
pg.display.set_caption("Maze Robot Simulator")
pg.draw.rect(window, background_color, (0,0,window_dimensions[0], window_dimensions[1]))

# draws a wall
pg.draw.rect(window, wall_color, (0,0,cell_x_length, cell_y_length))

# draws a empty cell
pg.draw.rect(window, empty_cell_color, (0,0,cell_x_length, cell_y_length))

example_walls_to_draw = [(0,0), (3,1),(2,3),(0,9),(9,0),(9,9)]
for cell in example_walls_to_draw:
    
    # convert cell cordinate locations to placement on the window using pixels
    
    x_coord = cell[0]
    y_coord = cell[1]
    
    x_coord_in_pixels = x_coord * cell_x_length + pixel_rendering_offset_x_from_top_left
    y_coord_in_pixels = y_coord * cell_y_length +pixel_rendering_offset_y_from_top_left
    
    pg.draw.rect(window, wall_color, (x_coord_in_pixels,y_coord_in_pixels,cell_x_length, cell_y_length))

# LEFT OFF, now create a function that renders a list of coordinates given a cell type.


# renders everything
pg.display.flip()


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False