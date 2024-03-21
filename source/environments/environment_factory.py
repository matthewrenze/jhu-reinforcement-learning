from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment import Environment
from agents.agent import Agent
from ghosts.ghost import Ghost
from tiles.tiles import Tiles
from tiles.tile import Tile

class EnvironmentFactory:

    def __init__(
            self,
            tile_factory: TileFactory,
            agent_factory: AgentFactory,
            house_factory: HouseFactory,
            ghost_factory: GhostFactory):
        self.tile_factory = tile_factory
        self.agent_factory = agent_factory
        self.house_factory = house_factory
        self.ghost_factory = ghost_factory

    def create(self, environment_id: int, agent_name: str, hyperparameters: dict[str, float]) -> Environment:
        level_map = self._load(environment_id)
        tiles = self.tile_factory.create(level_map)
        agent = self.agent_factory.create(agent_name, tiles, hyperparameters)
        house = self.house_factory.create()
        ghosts = self.ghost_factory.create(tiles, house)
        self._clear_agents(tiles, agent, ghosts)
        return Environment(tiles, agent, ghosts)

    def _load(self, environment_id: int) -> str:
        file_name = f"level-{environment_id}.txt"
        file_path = f"levels/{file_name}"
        environment = self._read_file(file_path)
        return environment

    def _read_file(self, file_path: str) -> str:
        with open(file_path, 'r') as file:
            environment = file.read()
        return environment

    def _clear_agents(self, tiles: Tiles, agent: Agent, ghosts: list[Ghost]):
        tiles[agent.location] = Tile.EMPTY
        for ghost in ghosts:
            tiles[ghost.location] = Tile.EMPTY
