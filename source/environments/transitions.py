from typing import Tuple
from agents.actions import Action

# TODO: Move all of this into the Action enum

action_transition = {
    Action.NONE.value: (0, 0),
    Action.UP.value: (-1, 0),
    Action.RIGHT.value: (0, 1),
    Action.DOWN.value: (1, 0),
    Action.LEFT.value: (0, -1)
}

def get_action_transition(action: int) -> Tuple[int, int]:
    return action_transition[action]