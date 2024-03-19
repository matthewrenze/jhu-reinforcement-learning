import numpy as np
from tiles.tiles import Tiles
from tiles.tile import Tile
from agents.agent import Agent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent

class AgentFactory:

    def _get_agent_location(self, tiles: Tiles) -> tuple[int, int]:
        tile_id = Tile.PACMAN
        location = np.where(tiles == tile_id)
        if len(location[0]) == 0:
            raise ValueError(f"No tile with agent found in state")
        if len(location[0]) > 1:
            raise ValueError(f"Multiple tiles with agent found in state")
        row = int(location[0][0])
        col = int(location[1][0])
        return row, col

    def create(self, agent_name: str, tiles: Tiles) -> Agent:

        location = self._get_agent_location(tiles)

        if agent_name == "human":
            return HumanAgent(location)

        elif agent_name == "random":
            return RandomAgent(location)

        raise ValueError(f"Unknown agent name: {agent_name}")