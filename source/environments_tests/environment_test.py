import pytest
import sys
from pathlib import Path
import numpy as np
from environments.environment import Environment
from environments.states import Tile

def test_get_state():
    state = [[0, 1, 2], [3, 4, 5], [6, 0, 0]]
    state = np.ndarray(shape=(3, 3), buffer=np.array(state), dtype=int)
    environment = Environment(state, (0, 2), [(1, 0)])
    actual_state = environment.get_state()
    expected_state = np.array([[0, 1, 2], [3, 4, 5], [6, 0, 0]])
    assert np.array_equal(actual_state, expected_state)

# Create a parameterized test for the test_is_valid_move function
@pytest.mark.parametrize("new_location, expected", [
    ((-1, 0), False),
    ((0, -1), False),
    ((10, 9), False),
    ((9, 10), False),
    ((5, 5), False),
    ((4, 4), True)])

def test_is_valid_move(new_location, expected):
    state = np.zeros((10, 10))
    state[5, 5] = Tile.WALL.id
    environment = Environment(state, (0, 0), [])
    actual = environment._is_valid_move(new_location)
    assert expected == actual



