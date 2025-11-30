import numpy as np
import pytest
from logic import *
from reward import *
import random


initializing_environment = np.full((4,4), empty_value,dtype=int)
example_walls = [(0,2),(0,3)]
example_goal = [(2,1)]
example_start = [(0,0)]

example_environment= add_custom_object(initializing_environment, example_walls,wall_value)
example_environment = add_custom_object(example_environment,example_goal,goal_value)
example_environment = add_custom_object(example_environment,example_start,start_value)

example_q_table = [
    [0.5, 2.3, -1.2, 3.7, 0.0],
    [1.1, -0.5, 4.2, 0.3, -2.1],
    [-1.0, 0.0, 0.0, 0.0, 0.0],
    [5.5, 3.2, 1.8, 4.1, 2.7],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [2.1, 3.5, 1.2, 0.8, 4.0],
    [-0.5, 1.5, 2.5, 3.5, 0.5],
    [0.0, 3.0, 5.0, 4.0, 0.5],
    [3.3, 2.2, 1.1, 4.4, 5.5]]
example_envionment_x_length = 3
example_environment_y_length = 3



# update the first input to be a tuple, not a list
def test_coordinates_to_q_table_index() -> None:
    test_q_table_width = 4
    test_environment_x_length = 10
    assert coordinates_to_q_table_index([0,0],test_environment_x_length,10,test_q_table_width) == 0
    assert coordinates_to_q_table_index([1,0],test_environment_x_length,10,test_q_table_width) == 0 + 1 * test_environment_x_length
    assert coordinates_to_q_table_index([0,1],test_environment_x_length,10,test_q_table_width) == 1 + 0 * test_environment_x_length
    assert coordinates_to_q_table_index([5,1],test_environment_x_length,10,test_q_table_width) == 1 + 5 * test_environment_x_length
    assert coordinates_to_q_table_index([5,9],test_environment_x_length,10,test_q_table_width) == 9 + 5 * test_environment_x_length
    assert coordinates_to_q_table_index([-5,0],test_environment_x_length,10, test_q_table_width) == -100
    assert coordinates_to_q_table_index([0,-5],test_environment_x_length,10,test_q_table_width) == -100
    assert coordinates_to_q_table_index([10,0],test_environment_x_length,10,test_q_table_width) == -100
    assert coordinates_to_q_table_index([0,11],test_environment_x_length,10,test_q_table_width) == -100
    assert coordinates_to_q_table_index([-5,-50],test_environment_x_length,10, test_q_table_width) == -100
    assert coordinates_to_q_table_index([1,2],example_envionment_x_length,example_environment_y_length,0) == 2 + 1 * example_envionment_x_length #is 2 + 1 * 3 = 5
    assert coordinates_to_q_table_index([2,1],example_envionment_x_length,example_environment_y_length,0) == 1 + 2 * example_envionment_x_length #is 1 + 2 * 3 = 7
# I already tested choose_action with coords (1,2), now I want to test (2,1) to make sure that they give different answers.

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

def test_choose_action() -> None:
    #def choose_action(current_pos: tuple(int,int), q_table: tuple[tuple[int,int]], environment_x_length: int, environment_y_length, epsilon: float) -> tuple(str,str):
    # this example q table is 3 by 3, so it's domain of coords is (0,0) -> (2,2)
    
    
    """action_map = {
    "up": (-1,0),
    "down":(1,0),
    "left":(0,-1),
    "right":(0,1),
    "remain": (0,0)
    }"""
    
    
    assert choose_action((0,0), example_q_table,example_envionment_x_length, example_environment_y_length, 0) == ("right","right")
    assert choose_action((1,2), example_q_table, example_envionment_x_length,example_environment_y_length,0) == ("remain","remain") 
    #q table index is 5, so that q table row is [2.1, 3.5, 1.2, 0.8, 4.0]. The highest value is index 4, which in action_map corresponds to remain.
    
    assert choose_action((2,1), example_q_table, example_envionment_x_length,example_environment_y_length,0) == ("left","left")
    # q table index is 7, so that q table row is [0.0, 3.0, 5.0, 4.0, 0.5].The highest value is at index 2, which in action_map corresponds to left
    
    """Testing randomness of choose_action. 
    CODE to run in main.py:

    from test_logic import *

    for i in range(0,100,1):
        print(choose_action((2,1),example_q_table,example_envionment_x_length,example_environment_y_length,0.5))
    print()
    print()
    print()
    
    for i in range(0,100,1):
        print(choose_action((2,1),example_q_table,example_envionment_x_length,example_environment_y_length,1))

    ANALYSIS of output:
    Excellent! Much better distribution now. Let me count both loops for you:
    First Loop (epsilon = 0.5, 100 iterations)
    Second position counts:

    'left': 39 times (39%)
    'remain': 24 times (24%)
    'right': 19 times (19%)
    'up': 11 times (11%)
    'down': 7 times (7%)

    Second Loop (epsilon = 1.0, 100 iterations)
    Second position counts:

    'remain': 29 times (29%)
    'down': 26 times (26%)
    'up': 24 times (24%)
    'right': 21 times (21%)
    'left': 0 times (0%)

    Analysis
    First loop: All 5 actions appear, though there's some variation (39% vs 7%). With only 100 samples, some variation is expected, but the range is wider than ideal. Still, all actions are being selected.
    Second loop: This is interesting - 'left' never appears in the second position, even though the other 4 actions are distributed fairly evenly (21-29%).
    What this tells you:
    In the second loop with epsilon=1.0 (always explore, never exploit), 'left' is excluded from the random selection entirely. But 'left' is the optimal action for position (2,1) in your test.
    Question for you: Your code removes the optimal action from the pool before randomly selecting. In the second loop, what is the optimal action? Why would that action not appear in the second position?
    This is actually correct behavior - when exploring, you're randomly choosing from non-optimal actions. The pattern matches what your code is designed to do."""
    
    """
    TESTING the functionality of if the q table row has all zeros, the function randomly chooses an action instead of just picking the first action listed in the actions list.
    
    example_blank_q_table = np.full((4,4), 0,dtype=int)
    
    for i in range(0,100):
        print(choose_action((0,0),example_blank_q_table,example_envionment_x_length, example_environment_y_length, 0))
        
    Output synthesized by claude:
    
        Counting your 98 total outputs:
    Breakdown by action:

    'right': 25 times (25.5%)
    'left': 20 times (20.4%)
    'down': 18 times (18.4%)
    'up': 17 times (17.3%)
    'remain': 18 times (18.4%)
        
        
        """

def test_update_q_table() -> None:
    # def update_q_table(old_pos: tuple[int,int], action: str, new_pos: tuple[int,int], actions_list: dict,environment: tuple[tuple[int,int]], environment_x_length: int, environment_y_length: int, walls: list[tuple[int,int]], q_table: tuple[tuple[int,int]], alpha:float, gamma: float) -> None:
    q_table_calc = update_q_table((2,2),"up",(1,2),actions, example_environment,example_envionment_x_length,example_environment_y_length,example_walls,example_q_table,0.5,0.1)
    
    example_q_table_updated_1 = q_table_calc[0]
    expected_new_q_value = q_table_calc[1]
    old_pos_q_table_index = q_table_calc[2]
    new_pos_q_table_index = q_table_calc[3]
    
    action_index = actions.index("up")
    
    assert example_q_table_updated_1[old_pos_q_table_index][action_index] == expected_new_q_value

