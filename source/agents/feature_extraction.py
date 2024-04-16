import numpy as np

from states.state import State
from tiles.tile import Tile
from tiles.tiles import Tiles
from ghosts.ghost import Ghost, Mode

class FeatureExtraction():
    def __init__(self, state:State):
        self._current_state = state
        self._tiles = state.tiles

    def distance_closest_food(self): 
        distance = self._find_minimum_distance(self._current_state, [3])
        return distance

    def distance_closest_ghost(self):
        ghost_tiles = [5,6,7,8,9]
        distance = self._find_minimum_distance(self._current_state, ghost_tiles)
        return distance 

    def distance_closest_powerpellet(self): 
        pellet_tile = [4]
        distance = self._find_minimum_distance(self._current_state, pellet_tile)
        return distance
    
    def number_active_ghosts_1step(self): 
        modes = [0, 1]
        num_ghosts = self._ghosts_1step_away(self._current_state, modes, False)
        return num_ghosts

    def number_active_ghosts_2step(self): 
        modes = [0, 1]
        num_ghosts = self._ghosts_2steps_away(self._current_state, modes, False)
        return num_ghosts

    def number_scared_ghosts_1step(self): 
        modes = [1]
        num_ghosts = self._ghosts_1step_away(self._current_state, modes, True)
        return num_ghosts

    def number_scared_ghosts_2step(self): 
        modes = [1]
        num_ghosts = self._ghosts_2steps_away(self._current_state, modes, True)
        return num_ghosts
    
    def number_power_pellets_1step(self):
        pellet_tile = [4]
        num_pellets = self._tile_1step_away(self._current_state, pellet_tile)
        return num_pellets

    def number_power_pellets_2steps(self):
        pellet_tile = [4]
        num_pellets = self._tile_2steps_away(self._current_state, pellet_tile)
        return num_pellets

    def number_food_1step(self):
        food_tile = [3]
        num_food = self._tile_1step_away(self._current_state, food_tile)
        return num_food

    def number_food_2steps(self):
        food_tile = [3]
        num_food = self._tile_2steps_away(self._current_state, food_tile)
        return num_food

    def find_legal_positions(self, current_position):
        legal_positions = []
        all_positions = [current_position]
        height = self._tiles.shape[0]
        width = self._tiles.shape[1]
    
        transitions = [(-1,0),(1,0),(0,-1),(0,1)]
        for i in transitions: 
            new_x = current_position[0] + i[0]
            new_y = current_position[1] + i[1]
            new_location = (new_x, new_y)
    
            if self._can_teleport(new_location, height, width):
                new_location = self._teleport(new_location, height, width)
        
            if self._tiles[new_location] != 1:
                legal_positions.append(new_location)
            else: 
                new_location = current_position
            
            all_positions.append(new_location)

        return legal_positions, all_positions
    

    def _find_minimum_distance(self, state:State, desired_tiles:list[int]): 
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
            if self._tiles[(position_x, position_y)] in desired_tiles and distance > 0: 
                return distance
            else: 
                distance += 1
                legal_positions = self.find_legal_positions((position_x, position_y))[0]
                # All legal positions are the same distance away
                search_locations.extend([(i[0], i[1], distance) for i in legal_positions])
        return 7
   

    def _ghosts_1step_away(self, state:State, modes:list[int], need_invicibility): 
        current_position = state.agent_location
        ghost_mode = state.ghost_mode
        num_desired = 0
        if ghost_mode in modes and state.is_invincible == need_invicibility:
            new_locations = self.find_legal_positions(current_position)[0]
            for i in new_locations: 
                if self._tiles[i] in [5,6,7,8]:
                    num_desired += 1
            
        return num_desired
    

    def _ghosts_2steps_away(self, state:State, modes:list[int], need_invicibility): 
        current_position = state.agent_location
        found_ghosts = set()

        ghost_mode = state.ghost_mode
        
        if ghost_mode in modes and state.is_invincible == need_invicibility:
            step_locations = self.find_legal_positions(current_position)[0]
            for i in step_locations: 
                step2_locations = self.find_legal_positions((i[0],i[1]))[0]
                for j in step2_locations: 
                    if (j[0] >= 0) and (j[1] >= 0) and self._tiles[(j[0], j[1])] in [5,6,7,8]:
                        found_ghosts.add((j[0],j[1]))
            
        return len(found_ghosts)
    

    def _tile_1step_away(self, state:State, desired_tiles:list[int]):
        current_position = state.agent_location
        found_tiles = set()

        new_locations = self.find_legal_positions(current_position)[0]
        for i in new_locations: 
           
            if self._tiles[i] in desired_tiles:
                found_tiles.add(i)
            
        return len(found_tiles)


    def _tile_2steps_away(self, state:State, desired_tiles:list[int]):
        current_position = state.agent_location
        found_tiles = set()

        step_locations = self.find_legal_positions(current_position)[0]
        for i in step_locations: 
            step2_locations = self.find_legal_positions((i[0],i[1]))[0]
            for j in step2_locations: 
                if (j[0] >= 0) and (j[1] >= 0) and self._tiles[(j[0], j[1])] in desired_tiles:
                    found_tiles.add((j[0],j[1]))
            
        return len(found_tiles)


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
   
    
        
