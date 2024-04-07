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
        
        while len(search_locations) != 0 and distance < 7: 
            position_x, position_y, distance = search_locations.pop(0)
            if (position_x, position_y) in search_tracker: 
                continue
            search_tracker.add((position_x, position_y))
            if self._tiles[(position_x, position_y)] in desired_tiles: 
                return distance
            else: 
                distance += 1
                legal_positions = self.find_legal_positions((position_x, position_y))
                # All legal positions are the same distance away
                search_locations.extend([(i[0], i[1], distance) for i in legal_positions])
        return 7
   

    def _ghosts_1step_away(self, state:State, mode:Mode): 
        current_position = state.agent_location
        ghost_locations = [(i[1][0], i[1][1]) for i in state.ghost_locations]
        ghost_mode = state.ghost_mode
        num_desired = 0
        if ghost_mode == mode.value:
            new_locations = self.find_legal_positions(current_position)
            num_desired += len(new_locations) - len(set(new_locations) - set(ghost_locations))
        return num_desired
    

    def _ghosts_2steps_away(self, state:State, mode:Mode): 
        current_position = state.agent_location
        found_ghosts = set()

        ghost_locations = [(i[1][0], i[1][1]) for i in state.ghost_locations]
        ghost_mode = state.ghost_mode
        
        if ghost_mode == mode.value:
            step_locations = self.find_legal_positions(current_position)
            for i in step_locations: 
                step2_locations = self.find_legal_positions((i[0],i[1]))
                for j in step2_locations: 
                    if (j[0] >= 0) and (j[1] >= 0) and (j[0], j[1]) in ghost_locations:
                        found_ghosts.add((j[0],j[1]))
            
        return len(found_ghosts)
            
    def find_legal_positions(self, current_position):
        legal_positions = []
        height = self._tiles.shape[0]
        width = self._tiles.shape[1]
    
        transitions = [(-1,0),(0,1),(1,0),(0,-1)]
        for i in transitions: 
            new_x = current_position[0] + i[0]
            new_y = current_position[1] + i[1]
            new_location = (new_x, new_y)

            if self._can_teleport(new_location, height, width):
                new_location = self._teleport(new_location, height, width)
        
            if self._tiles[new_location] != Tile.WALL:
                legal_positions.append(new_location)
         
        return legal_positions
   

    def _can_teleport(self, new_location: tuple[int, int], height, width) -> bool:
        if new_location[0] < 0 \
                or new_location[1] < 0 \
                or new_location[0] >= height \
                or new_location[1] >= width:
            return True
        return False


    def _teleport(self, new_location: tuple[int, int], height, width) -> tuple[int, int]:
        if new_location[0] < 0:
            new_location = (height - 1, new_location[1])
        if new_location[0] >= height:
            new_location = (0, new_location[1])
        if new_location[1] < 0:
            new_location = (new_location[0], width - 1)
        if new_location[1] >= width:
            new_location = (new_location[0], 0)
        return new_location
    
        

