import numpy as np
from agents.agent import Agent
from states.state import State
from actions.action import Action

class RandomAgent(Agent):

    def __init__(self, location: tuple[int, int]):
        super().__init__(location)

    def _get_random_action_id(self):
        action_id = np.random.choice(5)
        return action_id

    def select_action(self, state: State) -> Action:
        action_id = self._get_random_action_id()
        action = Action(action_id)
        if action != Action.NONE:
            self.orientation = action
        return action
