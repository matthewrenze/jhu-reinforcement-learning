import numpy as np
from agents.agent import Agent
from states.state import State
from actions.action import Action

class RandomAgent(Agent):

    def __init__(self, location: tuple[int, int]):
        super().__init__(location)

    def _get_random_action(self):
        action_id = np.random.choice(5)
        return Action(action_id)


    def select_action(self, state: State) -> Action:
        action = self._get_random_action()
        self.orientation = action
        return action
