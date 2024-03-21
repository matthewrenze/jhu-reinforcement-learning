import os
import numpy as np
from tiles.tiles import Tiles
from tiles.tile import Tile
from agents.agent import Agent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.sarsa_agent import SarsaAgent

class AgentFactory:

    def create(self, agent_name: str, tiles: Tiles, hyperparameters: dict[str, float]) -> Agent:

        location = self._get_agent_location(tiles)

        if agent_name == "human":
            return HumanAgent(location, hyperparameters)

        elif agent_name == "random":
            return RandomAgent(location, hyperparameters)

        elif agent_name == "sarsa":
            q_table = self._get_agent_q_table(agent_name)
            return SarsaAgent(location, hyperparameters, q_table)

        raise ValueError(f"Unknown agent name: {agent_name}")

    # NOTE: This method and q_table load might be better in a separate class
    def save(self, agent_name: str, q_table: np.ndarray) -> None:
        folder_path = "../q_tables"
        file_name = f"{agent_name}.csv"
        file_path = f"{folder_path}/{file_name}"
        np.savetxt(file_path, q_table, delimiter=",")

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

    def _get_agent_q_table(self, agent_name: str) -> np.ndarray:
        folder_path = "../q_tables"
        file_name = f"{agent_name}.csv"
        file_path = f"{folder_path}/{file_name}"
        if not os.path.exists(file_path):
            return None
        q_table = np.loadtxt(file_path, delimiter=",")
        return q_table