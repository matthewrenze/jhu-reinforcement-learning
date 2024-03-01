import numpy as np
from agents.agent import Agent
from agents.actions import Action

key_map = {
    '': Action.NONE.value,
    'i': Action.UP.value,
    'l': Action.RIGHT.value,
    'k': Action.DOWN.value,
    'j': Action.LEFT.value
}

class HumanAgent(Agent):

    def __init__(self, location: tuple[int, int]):
        super().__init__(location)

    def _get_key(self):
        key = input("Enter a direction [i, j, k, l]: ")
        return key

    def select_action(self, state: np.ndarray):
        key = self._get_key()
        if key not in key_map:
            return Action.NONE.value
        action = key_map.get(key)
        return action
