import numpy as np
from agents.agent import Agent
from models.model import Model
from models.q_table import QTable
from actions.action import Action
from states.state import State

class QLearningAgent(Agent):

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float]):
        pass

    def select_action(self, state: State) -> Action:
        pass

    def update(self, state: State, action: Action, reward: int, next_state: State) -> None:
        pass

    def get_model(self) -> QTable:
        pass

    def set_model(self, model: QTable) -> None:
        pass

    def _get_random_threshold(self) -> float:
        pass

    def _get_random_action_id(self) -> int:
        pass

    # TODO: Refactor this into an abstract tabular_agent superclass or state_converter class
    # TODO: To be shared by both SarsaAgent and QLearningAgent
    def _convert_state(self, state: State) -> int:
       pass
