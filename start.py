import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Hey")

directions_agent_can_move = 4
environment_x_length = 10
environment_y_length = 10

q_table_width = directions_agent_can_move

q_table = np.zeros((environment_x_length*environment_y_length,directions_agent_can_move))

print(q_table)


direction_map = {
    "up": (0,-1),
    "down":(0,1),
    "left":(-1,0),
    "right":(1,0)
    }

