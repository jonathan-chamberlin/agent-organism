import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import math
from logic import *
from setup_environment import *

cell_reward = {
    "wall": -10,
    "goal": 50,
    "empty": -1,
    "start": 0 
}

wall_reward = cell_reward["wall"]
goal_reward = cell_reward["goal"]
start_reward = cell_reward["start"]
empty_reward = cell_reward["empty"]