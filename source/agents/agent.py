from states.state import State
from actions.action import Action

class Agent:

    def __init__(self, location: tuple[int, int]):
        self.location = location

    def select_action(self, state: State) -> Action:
        raise NotImplementedError("get_action method must be implemented in the subclass.")