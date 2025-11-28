from logic import *
from reward import *
from setup_environment import *
from test_logic import *

moves = ["down","down","down", "right","right", "right","up","up","up","right","right","right","right","right","right","down","down","down","down","down","down","down","left","left","left","down","down","right","right","right"]

print(game_loop(full_environment, start, walls, cell_color_map, background_color, moves))


# Should I use Continous learning? (Q table where the agent calcuates what is the next best position given it's current positionP? Or should I make episode termination on failure (it so if the agent hits a wall he has to restart)?
# Yes, continous learning with the Q table is better when failures are not catestrophic. It also takes much less time to learn. So I'll use continous learning.


for i in range(0,100,1):
    print(choose_action((2,1),example_q_table,example_envionment_x_length,example_environment_y_length,0.5))
print()
print()
print()
 
for i in range(0,100,1):
    print(choose_action((2,1),example_q_table,example_envionment_x_length,example_environment_y_length,1))

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False