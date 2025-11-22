from start import coordinates_to_q_table_index
from start import q_table_width
from start import coordinates_after_moving
from start import add_walls
import numpy as np


def test_coordinates_to_q_table_index() -> None:
    assert coordinates_to_q_table_index([0,0]) == 0
    assert coordinates_to_q_table_index([1,0]) == 1 + 0 * q_table_width
    assert coordinates_to_q_table_index([0,1]) == 0 + 1 * q_table_width
    assert coordinates_to_q_table_index([5,1]) == 5 + 1 * q_table_width
    assert coordinates_to_q_table_index([5,9]) == 5 + 9 * q_table_width
    assert coordinates_to_q_table_index([-5,0]) == -1
    assert coordinates_to_q_table_index([0,-5]) == -1
    assert coordinates_to_q_table_index([10,0]) == -1
    assert coordinates_to_q_table_index([0,11]) == -1
    assert coordinates_to_q_table_index([-5,-50]) == -1

def test_coordinates_after_moving() -> None:
    assert coordinates_after_moving((0,0),"down") == (0,1)
    assert coordinates_after_moving((0,0),"right") == (1,0)
    assert coordinates_after_moving((1,1),"left") == (0,1)
    assert coordinates_after_moving((1,1),"left") == (0,1)
    assert coordinates_after_moving((0,0),"up") == (-5,-5)
    assert coordinates_after_moving((9,0),"right") == (-5,-5)
    assert coordinates_after_moving((10,0),"left") == (-1,-1)
    assert coordinates_after_moving((0,10),"up") == (-1,-1)

def test_add_walls() -> None:
    example_maze = np.zeros((2,2))
    example_walls_1 = [(0,0)]
    example_walls_2 = [(0,0), (1,0)]
    example_walls_3 = [(0,0), (1,0), (0,1)]
    example_walls_4 = [(0,0), (1,0), (0,1), (1,1)]
    assert add_walls(example_maze, example_walls_1) == [[-1,0],[0,0]]
    assert add_walls(example_maze, example_walls_2) == [[-1,-1],[0,0]]
    assert add_walls(example_maze, example_walls_3) == [[-1,-1],[-1,0]]
    assert add_walls(example_maze, example_walls_4) == [[-1,-1],[-1,-1]]
    
    
    