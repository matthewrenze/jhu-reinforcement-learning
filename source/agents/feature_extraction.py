import numpy as np

from states.state import State
from tiles.tile import Tile
from tiles.tiles import Tiles
from ghosts.ghost import Ghost, Mode

class FeatureExtraction():
    def __init__(self, tiles:Tiles, state:State):
        self._tiles = tiles
        self._current_state = state

    def distance_closest_food(self): 
        distance = self._find_minimum_distance(self._current_state, [Tile.DOT])
        return distance

    def distance_closest_ghost(self): 
        ghost_tiles = [Tile.BLINKY, Tile.INKY, Tile.PINKY, Tile.CLYDE, Tile.STATIC]
        distance = self._find_minimum_distance(self._current_state, ghost_tiles)
        return distance 
    
    def number_active_ghosts_1step(self): 
        num_ghosts = self._ghosts_1step_away(self._current_state, Mode.CHASE)
        return num_ghosts

    def number_active_ghosts_2step(self): 
        num_ghosts = self._ghosts_2steps_away(self._current_state, Mode.CHASE)
        return num_ghosts

    def number_scared_ghosts_1step(self): 
        num_ghosts = self._ghosts_1step_away(self._current_state, Mode.FRIGHTENED)
        return num_ghosts

    def number_scared_ghosts_2step(self): 
        num_ghosts = self._ghosts_2steps_away(self._current_state, Mode.FRIGHTENED)
        return num_ghosts

    def safety_mode(self): 
        pass

    def _find_minimum_distance(self, state:State, desired_tiles:list[Tile]): 
        current_position = state.agent_location
        distance = 0
        search_locations = [(current_position[0], current_position[1], distance)]
        # Make sure only unique search locations are checked 
        search_tracker = set()
        
        while len(search_locations) != 0: 
            position_x, position_y, distance = search_locations.pop(0)
            if (position_x, position_y) in search_tracker: 
                continue
            search_tracker.add((position_x, position_y))
            for i in desired_tiles:
                if self._tiles[(position_x, position_y)] == i: 
                    return distance
            else: 
                distance += 1
                legal_positions = self._find_legal_positions((position_x, position_y))
                # All legal positions are the same distance away
                for i in legal_positions: 
                    search_locations.append((i[0], i[1], distance))
        return None

    def _find_legal_positions(self, current_position):
        legal_positions = []
        possible_positions = self._1step_positions((current_position[0], current_position[1]))
        for i in possible_positions:
            # Check new row and col are valid and not a wall tile
            if (i[0] >= 0) and (i[1] >= 0) and not self._tiles[(i[0], i[1])] == Tile.WALL :
                legal_positions.append((i[0], i[1]))
        return legal_positions

    def _ghosts_1step_away(self, state:State, mode:Mode): 
        current_position = state.agent_location
        
        ghost_modes = [i.name for i in state.ghost_mode]
        ghost_idx = list(np.where(np.array(ghost_modes) == mode.name))[0]
        ghost_locations_all = [(i[1][0], i[1][1]) for i in state.ghost_locations]
        ghost_locations = [ghost_locations_all[i] for i in ghost_idx]

        num_desired = 0
          
        new_locations = self._1step_positions((current_position[0], current_position[1]))
        for i in new_locations:
            if (i[0] >= 0) and (i[1] >= 0) and (i[0], i[1]) in ghost_locations:
                num_desired += 1
            
        return num_desired

    def _ghosts_2steps_away(self, state:State, mode:Mode): 
        current_position = state.agent_location
        found_ghosts = set()

        ghost_modes = [i.name for i in state.ghost_mode]
        ghost_idx = list(np.where(np.array(ghost_modes) == mode.name))[0]
        ghost_locations_all = [(i[1][0], i[1][1]) for i in state.ghost_locations]
        ghost_locations = [ghost_locations_all[i] for i in ghost_idx]
        
        step_locations = self._1step_positions(current_position)
        for i in step_locations: 
            step2_locations = self._1step_positions((i[0],i[1]))
            for j in step2_locations: 
                if (j[0] >= 0) and (j[1] >= 0) and (j[0], j[1]) in ghost_locations:
                    found_ghosts.add((j[0],j[1]))
        
        return len(found_ghosts)

    def _1step_positions(self, starting_position):
        new_locations = []
        transitions = [(-1,0),(0,1),(1,0),(0,-1)]
        for i in transitions: 
            new_locations.append((starting_position[0] + i[0],starting_position[1] + i[1]))
        
        return new_locations
                                 
            

        

