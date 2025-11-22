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
    example_walls = [(1,0),(4,0), (2,3),(3,0)]
    assert coordinates_after_moving((0,0),"down", example_walls) == ((0,1),True)
    # output_valid == False because agent hits a wall
    assert coordinates_after_moving((0,0),"right", example_walls) == ((1,0), False)
    assert coordinates_after_moving((1,1),"left", example_walls) == ((0,1),True)
    assert coordinates_after_moving((1,1),"left", example_walls) == ((0,1), True)
    # output_valid == False because the movement moves the agent outside the environment
    assert coordinates_after_moving((0,0),"up", example_walls) == ((0,-1), False)
    assert coordinates_after_moving((9,0),"right", example_walls) == ((10,0), False)
     # output_valid == False because the input coordinate is outside the environment
    assert coordinates_after_moving((10,0),"left", example_walls) == ((9,0), False)
    assert coordinates_after_moving((0,10),"up", example_walls) == ((0,9), False)
    # now testing if it hits walls
    assert coordinates_after_moving((4,1), "up", example_walls) == ((4,0), False)

def test_add_walls() -> None:
    example_maze = np.zeros((2,2), dtype=int)
    example_walls_1 = [(0,0)]
    example_walls_2 = [(0,0), (1,0)]
    example_walls_3 = [(0,0), (1,0), (0,1)]
    example_walls_4 = [(0,0), (1,0), (0,1), (1,1)]
    assert np.array_equal(add_walls(example_maze, example_walls_1), [[-1,0],[0,0]])
    assert np.array_equal(add_walls(example_maze, example_walls_2), [[-1,-1],[0,0]])
    assert np.array_equal(add_walls(example_maze, example_walls_3), [[-1,-1],[-1,0]])
    assert np.array_equal(add_walls(example_maze, example_walls_4), [[-1,-1],[-1,-1]])
    
    
    