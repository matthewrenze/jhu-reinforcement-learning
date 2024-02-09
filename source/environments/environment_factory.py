import numpy as np
from environments.environment import Environment
from environments import states
from environments.states import Tile

environment_1 = """
#  #  #  #  #
#  #  .  #  #
#  .  c  .  #
#  #  .  #  #
#  #  #  #  #
"""

environment_2 = """
#  #  #  #  #
#  .  m  .  #
#  .  #  .  #
#  .  c  .  #
#  #  #  #  #
"""

environment_99 = """
   #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  |
   #  .  .  .  .  .  .  .  .  #  .  .  .  .  .  .  .  .  #  |
   #  o  #  #  .  #  #  #  .  #  .  #  #  #  .  #  #  o  #  |
   #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #  |
   #  .  #  #  .  #  .  #  #  #  #  #  .  #  .  #  #  .  #  |
   #  .  .  .  .  #  .  .  .  #  .  .  .  #  .  .  .  .  #  |
   #  #  #  #  .  #  #  #     #     #  #  #  .  #  #  #  #  |
            #  .  #                       #  .  #           |
#  #  #  #  #  .  #     #  #  m  #  #     #  .  #  #  #  #  #
               .        #  m  m  m  #     #  .              |
#  #  #  #  #  .  #     #  #  #  #  #     #  .  #  #  #  #  #
            #  .  #                       #  .  #           |
   #  #  #  #  .  #     #  #  #  #  #     #  .  #  #  #  #  |
   #  .  .  .  .  .  .  .  .  #  .  .  .  .  .  .  .  .  #  |
   #  .  #  #  .  #  #  #  .  #  .  #  #  #  .  #  #  .  #  |
   #  o  .  #  .  .  .  .  .  c  .  .  .  .  .  #  .  o  #  |
   #  #  .  #  .  #  .  #  #  #  #  #  .  #  .  #  .  #  #  |
   #  .  .  .  .  #  .  .  .  #  .  .  .  #  .  .  .  .  #  |
   #  .  #  #  #  #  #  #  .  #  .  #  #  #  #  #  #  .  #  |
   #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #  |
   #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  |
"""

class EnvironmentFactory:

    def _load(self, environment_id: int) -> str:
        if environment_id == 1:
            return environment_1
        elif environment_id == 2:
            return environment_2
        elif environment_id == 99:
            return environment_99
        else:
            raise ValueError("Unknown environment ID")

    def _convert(self, environment: str) -> np.ndarray:
        state = []
        # NOTE: I'm temporarily removing the first and last \n until I switch to file-based loading
        environment = environment[1:]
        environment = environment[:-1]
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