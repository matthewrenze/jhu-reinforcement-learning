import os
import numpy as np
from tiles.tiles import Tiles
from tiles.tile import Tile
from agents.agent import Agent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.sarsa_agent import SarsaAgent
from agents.q_learning_agent import QLearningAgent
from agents.approximate_q_learning_agent import ApproximateQLearningAgent
from agents.deep_q_learning_agent import DeepQLearningAgent

class AgentFactory:

    def create(self, agent_name: str, tiles: Tiles, hyperparameters: dict[str, float]) -> Agent:

        location = self._get_agent_location(tiles)

        if agent_name == "human":
            agent = HumanAgent(location, hyperparameters)

        elif agent_name == "random":
            agent = RandomAgent(location, hyperparameters)

        elif agent_name == "sarsa":
            agent = SarsaAgent(location, hyperparameters)

        elif agent_name == "q_learning":
            agent = QLearningAgent(location, hyperparameters)

        elif agent_name == "approximate_q_learning":
            agent = ApproximateQLearningAgent(location, hyperparameters)

        elif agent_name == "deep_q_learning":
            agent = DeepQLearningAgent(location, hyperparameters)

        elif agent_name == "approximate_q_learning": 
            agent = ApproximateQLearningAgent(location, hyperparameters)
            
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")

        return agent

    def _get_agent_location(self, tiles: Tiles) -> tuple[int, int]:
        tile_id = Tile.PACMAN
        location = np.where(tiles == tile_id)
        if len(location[0]) == 0:
            raise ValueError(f"No agent found in tiles")
        if len(location[0]) > 1:
            raise ValueError(f"Multiple agents found in tiles")
        row = int(location[0][0])
        col = int(location[1][0])
        return row, col