import numpy as np
from agents.actions import Action
from ghosts.static_ghost import StaticGhost

def test_select_action_returns_none():

    ghost = StaticGhost((0, 0))
    state = np.ndarray(shape=(2,2), dtype=int, buffer=np.array([1, 2, 3, 4]))
    actual = ghost.select_action(state, (0,0), True)
    expected = Action.NONE.value
    assert actual == expected
