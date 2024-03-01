import numpy as np
from environments.environment import Environment
from tiles import tile
from tiles.tile import Tile
from agents.agent_factory import AgentFactory
from agents.agent import Agent
from ghosts.ghost_factory import GhostFactory

class EnvironmentFactory:

    def __init__(self, agent_factory: AgentFactory, ghost_factory: GhostFactory):
        self.agent_factory = agent_factory
        self.ghost_factory = ghost_factory

    def _read_file(self, file_path: str) -> str:
        with open(file_path, 'r') as file:
            environment = file.read()
        return environment

    def _load(self, environment_id: int) -> str:
        file_name = f"level-{environment_id}.txt"
        file_path = f"curriculum/{file_name}"
        environment = self._read_file(file_path)
        return environment

    def _convert(self, environment: str) -> np.ndarray:
        state = []
        rows = environment.split('\n')
        for row in rows:
            row = row.replace('|', '   ')
            for index, char in enumerate(row):
                if index % 3 != 0:
                    continue
                tile_id = Tile.get_id_from_symbol(char)
                state.append(tile_id)
        shape = (len(rows), len(rows))
        state = np.ndarray(shape=shape, buffer=np.array(state), dtype=int)
        return state

    def create(self, environment_id: int, agent_name: str) -> Environment:
        state_string = self._load(environment_id)
        state_matrix = self._convert(state_string)
        agent = self.agent_factory.create(agent_name, state_matrix)
        ghosts = self.ghost_factory.create(state_matrix)
        return Environment(state_matrix, agent, ghosts)