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

def get_key():
    key = input("Enter a direction [i, j, k, l]: ")
    return key

class HumanAgent(Agent):
    def select_action(self, state: np.ndarray):
        key = get_key()
        action = key_map.get(key)
        return action
