from logic import *
from setup_environment import *

moves = ["down","down","down", "right","right", "right","up","up","up","right","right","right","right","right","right","down","down","down","down","down","down","down","left","left","left","down","down","right","right","right"]

print(game_loop(full_environment, start, walls, cell_color_map, background_color, moves))

# left off: Since I have a working game loop and way to have a list of moves, now I need to learn how to use this to make the agent learn and find it's way to the end of the maze. This will be new skillset, so paste in this claude prompt
# "Now I have a function game_loop(full_environment, start, walls, cell_color_map, background_color, moves) that actually renders everything and here is it's doc string 'Takes in a bunch of inputs, and for every move it draws the full environment (grid and background), then draws the agent, then calculates its next move nad position, then checks if that next position would be valid, then draws it, and renders it. It returns a list of booleans representing what MOVES were valid, NOT positions. So if the agent is on a valid square, and the the first move (index 0) is to an invalid square, then the output of this function will be [False, ...]. If the agent is on a valid square, and the first move (index) 0 is to a valid square, then it's second move (index 1) is to an invalid square, the output of this function will be [True, False, ...]'. Now I need to use this to make the agent learn and find it's way to the end of the maze. The full_environment variable is an array with integers of the gaol_value, wall_value, and empty_value. How can I use this stuff to make it so ultimately the agent learns from it's mistakes and finds its way to the end of the maze?

    

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False