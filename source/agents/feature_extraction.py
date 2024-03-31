import numpy as np

from states.state import State
from tiles.tile import Tile
from tiles.tiles import Tiles
from actions.action import Action

class FeatureExtraction():
    def __init__(self, tiles:Tiles, state:State):
        self._tiles = tiles
        self._current_state = state

    def distance_closest_food(self): 
        distance = self._find_minimum_distance(self._current_state, Tile.DOT)
        return distance

    def distance_closest_ghost(self): 
        ghost_tiles = [Tile.BLINKY, Tile.INKY, Tile.PINKY, Tile.CLYDE]
        distance = self._find_minimum_distance(self._current_state, ghost_tiles)
        return distance 
    
    def number_active_ghosts_1step(self): 
        pass

    def number_active_ghosts_2step(self): 
        pass

    def number_scared_ghosts_1step(self): 
        pass

    def number_scared_ghosts_2step(self): 
        pass

    def safety_mode(self): 
        pass

    def _find_minimum_distance(self, state:State, desired_tile:Tile): 
        current_position = state.agent_location
        search_locations = [(current_position[0], current_position[1])]
        distance = 0
        while len(search_locations) != 0: 
            position_x, position_y = search_locations.pop(0)
            if self._tiles[(position_x, position_y)] == desired_tile: 
                return distance
            else: 
                distance += 1
                legal_positions = self._find_legal_positions(current_position)
                for i in legal_positions: 
                    search_locations.append(i)
        return None


    def _find_legal_positions(self, current_position):

        legal_positions = []
        transitions = [(-1,0),(0,1),(1,0),(0,-1)]
        for i in transitions: 
            print(i)
     
            new_row = current_position[0] + i[0]
            new_col = current_position[1] + i[1]
            print((new_row, new_col))

            if not self._tiles[(new_row, new_col)] == Tile.WALL :
                print("LOOP")
                legal_positions.append((new_row, new_col))


        return legal_positions

    def _tiles_1step_away(self, current_position, desired_tile:Tile): 
        desired_1step = False

        transitions = [(-1,0),(0,1),(1,0),(0,-1)]
        for i in transitions: 
            new_row = current_position[0] + i[0]
            new_col = current_position[1] + i[1]
            if ~self._tiles[(new_row, new_col)] in desired_tile:
                desired_1step = True
            
        return desired_1step
    

    def _tiles_2steps_away(current_position, tiles): 
        pass
    

