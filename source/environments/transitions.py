from typing import Tuple
from actions.action import Action

# TODO: Move all of this into the Action enum

action_transition = {
    Action.NONE: (0, 0),
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1)}

def get_action_transition(action: Action) -> Tuple[int, int]:
    return action_transition[action]
