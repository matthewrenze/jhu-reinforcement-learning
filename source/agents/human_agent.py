from agents.agent import Agent
from actions.action import Action
from states.state import State

key_map = {
    '': Action.NONE,
    'i': Action.UP,
    'l': Action.RIGHT,
    'k': Action.DOWN,
    'j': Action.LEFT
}

class HumanAgent(Agent):

    def __init__(self, location: tuple[int, int]):
        super().__init__(location)

    def _get_key(self):
        key = input("Enter a direction [i, j, k, l]: ")
        return key

    def select_action(self, state: State) -> Action:
        key = self._get_key()
        action = Action.NONE
        if key in key_map:
            action = key_map.get(key)
        if action != Action.NONE:
            self.orientation = action
        return action
