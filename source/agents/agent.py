import numpy as np

class Agent:
    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError("get_action method must be implemented in the subclass.")