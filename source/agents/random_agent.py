import numpy as np
from agents.agent import Agent

class RandomAgent(Agent):

    def _get_random_action(self):
        return np.random.choice(5)

    def select_action(self, state: np.ndarray) -> int:
        return self._get_random_action()