import numpy as np
from environments.environment import Environment
from environments import states
from environments.states import Tile

class EnvironmentFactory:

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

    def create(self, environment_id: int) -> Environment:
        state_string = self._load(environment_id)
        state_matrix = self._convert(state_string)
        agent_location = states.get_agent_location(state_matrix)
        ghost_locations = states.get_ghost_locations(state_matrix)
        return Environment(state_matrix, agent_location, ghost_locations)