import numpy as np
import pytest
from logic import *
from reward import *


initializing_environment = np.full((4,4), empty_value,dtype=int)
example_walls = [(0,2),(0,3)]
example_goal = [(2,1)]
example_start = [(0,0)]

example_environment= add_custom_object(initializing_environment, example_walls,wall_value)
example_environment = add_custom_object(example_environment,example_goal,goal_value)
example_environment = add_custom_object(example_environment,example_start,start_value)



def test_coordinates_to_q_table_index() -> None:
    assert coordinates_to_q_table_index([0,0]) == 0
    assert coordinates_to_q_table_index([1,0]) == 1 + 0 * q_table_width
    assert coordinates_to_q_table_index([0,1]) == 0 + 1 * q_table_width
    assert coordinates_to_q_table_index([5,1]) == 5 + 1 * q_table_width
    assert coordinates_to_q_table_index([5,9]) == 5 + 9 * q_table_width
    assert coordinates_to_q_table_index([-5,0]) == -100
    assert coordinates_to_q_table_index([0,-5]) == -100
    assert coordinates_to_q_table_index([10,0]) == -100
    assert coordinates_to_q_table_index([0,11]) == -100
    assert coordinates_to_q_table_index([-5,-50]) == -100


def test_coordinates_after_moving() -> None:
    example_walls = [(1,0),(4,0), (2,3),(3,0)]
    assert coordinates_after_moving((0,0),"down", example_walls) == ((0,0),False)
    # output_valid == False because agent hits a wall
    assert coordinates_after_moving((0,0),"right", example_walls) == ((0,1), True)
    assert coordinates_after_moving((1,1),"left", example_walls) == ((1,1),False)

    assert coordinates_after_moving((0,0),"up", example_walls) == ((0,0),False)
    assert coordinates_after_moving((9,0),"right", example_walls) == ((9,1), True)
     # output_valid == False because the input coordinate is outside the environment
    assert coordinates_after_moving((10,0),"left", example_walls) == ((10,0), False)
    assert coordinates_after_moving((0,10),"up", example_walls) == ((0,10), False)
    # now testing if it hits walls
    assert coordinates_after_moving((3,0), "down", example_walls) == ((3,0), False)

def test_add_walls() -> None:
    from logic import wall_value
    example_maze = np.zeros((2,2), dtype=int)
    example_walls_1 = [(0,0)]
    example_walls_2 = [(0,0), (1,0)]
    example_walls_3 = [(0,0), (1,0), (0,1)]
    example_walls_4 = [(0,0), (1,0), (0,1), (1,1)]
    assert np.array_equal(add_walls(example_maze, example_walls_1), [[wall_value,0],[0,0]])
    assert np.array_equal(add_walls(example_maze, example_walls_2), [[wall_value,wall_value],[0,0]])
    assert np.array_equal(add_walls(example_maze, example_walls_3), [[wall_value,wall_value],[wall_value,0]])
    assert np.array_equal(add_walls(example_maze, example_walls_4), [[wall_value,wall_value],[wall_value,wall_value]])
    
def test_add_goals() -> None:
    example_maze = np.zeros((2,2), dtype=int)
    example_goals_1 = [(0,0)]
    example_goals_2 = [(0,0), (1,0)]
    example_goals_3 = [(0,0), (1,0), (0,1)]
    example_goals_4 = [(0,0), (1,0), (0,1), (1,1)]
    assert np.array_equal(add_goals(example_maze, example_goals_1), [[goal_value,0],[0,0]])
    assert np.array_equal(add_goals(example_maze, example_goals_2), [[goal_value,goal_value],[0,0]])
    assert np.array_equal(add_goals(example_maze, example_goals_3), [[goal_value,goal_value],[goal_value,0]])
    assert np.array_equal(add_goals(example_maze, example_goals_4), [[goal_value,goal_value],[goal_value,goal_value]])

def test_add_custom_object() -> None:
    example_maze = np.zeros((2,2), dtype=int)
    example_chosen_value = 5
    example_object_1 = [(0,0)]
    example_object_2 = [(0,0), (1,0)]
    example_object_3 = [(0,0), (1,0), (0,1)]
    example_object_4 = [(0,0), (1,0), (0,1), (1,1)]
    assert np.array_equal(add_custom_object(example_maze, example_object_1, example_chosen_value), [[example_chosen_value,0],[0,0]])
    assert np.array_equal(add_custom_object(example_maze, example_object_2, example_chosen_value), [[example_chosen_value,example_chosen_value],[0,0]])
    assert np.array_equal(add_custom_object(example_maze, example_object_3,example_chosen_value), [[example_chosen_value,example_chosen_value],[example_chosen_value,0]])
    assert np.array_equal(add_custom_object(example_maze, example_object_4,example_chosen_value), [[example_chosen_value,example_chosen_value],[example_chosen_value,example_chosen_value]])

def test_object_at_coordinates() -> None:
    initializing_environment = np.full((4,4), empty_value,dtype=int)
    example_walls = [(0,1),(0,2),(0,3)]
    global wall_value
    example_environment = add_custom_object(initializing_environment, example_walls,wall_value)
    
    object_at_coords((0,1),example_environment) == "wall"
    object_at_coords((1,0),example_environment) == "empty"
    object_at_coords((0,2),example_environment) == "wall"
    object_at_coords((2,0),example_environment) == "empty"
    object_at_coords((0,3),example_environment) == "wall"
    object_at_coords((3,0),example_environment) == "empty"

def test_adjacent_coords() -> None:
    assert adjacent_coords((0,0),"down") == (1,0)
    assert adjacent_coords((0,0),"up") == (-1,0)
    assert adjacent_coords((0,0),"left") == (0,-1)
    assert adjacent_coords((0,0),"right") == (0,1)

def test_get_reward() -> None:
    assert 1 == 1
    from reward import cell_reward
    global example_environment 
    
    assert get_reward((0,0),"down", example_environment, example_walls) == (cell_reward["empty"],(1,0),True)
    assert get_reward((1,2),"up", example_environment,example_walls) == (cell_reward["wall"],(0,2) ,False)
    assert get_reward((0,1),"left", example_environment,example_walls) == (cell_reward["start"], (0,0),True)
    assert get_reward((2,0),"right", example_environment,example_walls) == (cell_reward["goal"],(2,1) ,True)