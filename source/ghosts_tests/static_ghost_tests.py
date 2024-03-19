import numpy as np
from ghosts.static_ghost import StaticGhost
from houses.house import House
from states.state import State
from actions.action import Action

def test_select_action_returns_none():

    ghost = StaticGhost((0, 0), House([(0, 0)], (1, 1)))
    tiles = np.ndarray(shape=(2, 2), dtype=int, buffer=np.array([1, 2, 3, 4]))
    state = State(tiles, (0, 0), [(0, 1)], False, 1)
    actual = ghost.select_action(state)
    expected = Action.NONE
    assert actual == expected
