from logic import *
from visual import *



# LEFT OFF. Now use the function to move the agent, and have his coords update, and have the visual rerender. 






running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False