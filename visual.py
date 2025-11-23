from logic import *

pygame.init()

window_dimensions = (800,700)

pygame.display.set_mode(window_dimensions)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False