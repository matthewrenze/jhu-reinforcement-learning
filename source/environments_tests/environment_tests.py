import pytest
from unittest.mock import Mock
import numpy as np
from environments.environment import Environment
from environments.states import Tile
from agents.actions import Action

def test_reset():
    state = np.zeros((10, 10))
    environment = Environment(state, (0, 0), [])
    with pytest.raises(NotImplementedError):
        environment.reset(1)

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

@pytest.mark.parametrize("action, state_change_1, state_change_2, expected_reward, expected_game_over", [
    (Action.NONE.value, (1, 1, Tile.PACMAN.id), (1, 1, Tile.PACMAN.id), 0, False),
    (Action.UP.value, (1, 1, Tile.EMPTY.id), (0, 1, Tile.PACMAN.id), 0, False),
    (Action.DOWN.value, (1, 1, Tile.EMPTY.id), (2, 1, Tile.GHOST.id), 0, True),
    (Action.LEFT.value, (1, 1, Tile.EMPTY.id), (1, 0, Tile.PACMAN.id), 10, True),
    (Action.RIGHT.value, (1, 1, Tile.PACMAN.id), (1, 2, Tile.WALL.id), 0, False),

])
def test_execute_action(action, state_change_1, state_change_2, expected_reward, expected_game_over):
    state = [
        [Tile.EMPTY.id, Tile.EMPTY.id, Tile.EMPTY.id],
        [Tile.DOT.id, Tile.PACMAN.id, Tile.WALL.id],
        [Tile.EMPTY.id, Tile.GHOST.id, Tile.EMPTY.id]]
    state = np.ndarray(shape=(3, 3), buffer=np.array(state), dtype=int)
    expected_state = state.copy()
    expected_state[state_change_1[0]][state_change_1[1]] = state_change_1[2]
    expected_state[state_change_2[0]][state_change_2[1]] = state_change_2[2]
    environment = Environment(state, (1, 1), [(2, 1)])
    environment._get_random_action = Mock(return_value=Action.NONE.value)
    actual_state, actual_reward, actual_game_over = environment.execute_action(action)
    assert np.array_equal(expected_state, actual_state)
    assert expected_reward == actual_reward
    assert expected_game_over == actual_game_over




