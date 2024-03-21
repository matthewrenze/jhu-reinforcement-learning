from environments.environment import Environment
from agents.agent import Agent
from ghosts.ghost import Ghost
from tiles.tiles import Tiles
from tiles.tile import Tile

class EnvironmentFactory:

    def create(self, tiles, agent, ghosts) -> Environment:
        self._clear_agents(tiles, agent, ghosts)
        return Environment(tiles, agent, ghosts)

    def _clear_agents(self, tiles: Tiles, agent: Agent, ghosts: list[Ghost]):
        tiles[agent.location] = Tile.EMPTY
        for ghost in ghosts:
            tiles[ghost.location] = Tile.EMPTY
