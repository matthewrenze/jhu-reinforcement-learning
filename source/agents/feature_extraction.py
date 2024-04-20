import numpy as np

from states.state import State
from actions.action import Action
from tiles.tile import Tile
from tiles.tiles import Tiles
from ghosts.ghost import Ghost, Mode
from environments.transitions import get_action_transition
from environments.legal_positions import find_legal_positions

class FeatureExtraction():
    def __init__(self, state:State, action:Action):
        self._current_state = state
        self._tiles = state.tiles
        self._action = action

    def distance_closest_food(self): 
        distance = self._find_minimum_distance(self._current_state, self._action, [3], modes = None)
        return distance

    def distance_closest_ghost(self):
        ghost_tiles = [5,6,7,8,9]
        modes = [0, 1]
        distance = self._find_minimum_distance(self._current_state, self._action, ghost_tiles, modes)
        return distance 

    def distance_closest_powerpellet(self): 
        pellet_tile = [4]
        distance = self._find_minimum_distance(self._current_state, self._action, pellet_tile)
        return distance
    
    def number_active_ghosts_1step(self): 
        modes = [0, 1]
        num_ghosts = self._ghosts_1step_away(self._current_state, self._action, modes)
        return num_ghosts

    def number_active_ghosts_2step(self): 
        modes = [0, 1]
        num_ghosts = self._ghosts_2steps_away(self._current_state, self._action, modes)
        return num_ghosts

    def number_scared_ghosts_1step(self): 
        modes = [2]
        num_ghosts = self._ghosts_1step_away(self._current_state, self._action, modes)
        return num_ghosts

    def number_scared_ghosts_2step(self): 
        modes = [2]
        num_ghosts = self._ghosts_2steps_away(self._current_state, self._action, modes)
        return num_ghosts
    
    def number_power_pellets_1step(self):
        pellet_tile = [4]
        num_pellets = self._tile_1step_away(self._current_state, self._action, pellet_tile)
        return num_pellets

    def number_power_pellets_2steps(self):
        pellet_tile = [4]
        num_pellets = self._tile_2steps_away(self._current_state, self._action, pellet_tile)
        return num_pellets

    def number_food_1step(self):
        food_tile = [3]
        num_food = self._tile_1step_away(self._current_state, self._action, food_tile)
        return num_food

    def number_food_2steps(self):
        food_tile = [3]
        num_food = self._tile_2steps_away(self._current_state, self._action, food_tile)
        return num_food
    
    def food_focus(self): 
        current_position = self._current_state.agent_location
        transition = get_action_transition(self._action)
        new_position = (current_position[0]+transition[0], current_position[1]+transition[1])

        try:
            if self.number_active_ghosts_1step() == 0 and self.number_active_ghosts_2step() == 0 and self._tiles[(new_position[0], new_position[1])] == 3: 
                return 1
            else:
                return 0
        except: 
            return 0
        
    def safe_mode(self):
        current_position = self._current_state.agent_location
        transition = get_action_transition(self._action)
        new_position = (current_position[0]+transition[0], current_position[1]+transition[1])

        if self._current_state.is_invincible or self.distance_closest_ghost() >= 7:
            return 1
        else:
            return 0

    def _find_minimum_distance(self, state:State, action:Action, desired_tiles:list[int], modes): 
        current_position = state.agent_location
        transition = get_action_transition(action)
        new_position = (current_position[0]+transition[0], current_position[1]+transition[1])
        if new_position not in find_legal_positions(self._tiles, current_position): 
            new_position = current_position
        
        distance = 0
        search_locations = [(new_position[0], new_position[1], distance)]
        # Make sure only unique search locations are checked 
        search_tracker = set()
        
        while len(search_locations) != 0 and distance < 10: 
            position_x, position_y, distance = search_locations.pop(0)
            if (position_x, position_y) in search_tracker: 
                continue
            search_tracker.add((position_x, position_y))
            if self._tiles[(position_x, position_y)] in desired_tiles and distance > 0:
                if modes is not None and state.ghost_mode in modes:
                    return distance
                elif modes is None:
                    return distance
            else: 
                distance += 1
                legal_positions = find_legal_positions(self._tiles, (position_x, position_y))[0]
                # All legal positions are the same distance away
                search_locations.extend([(i[0], i[1], distance) for i in legal_positions])
        return 10
   

    def _ghosts_1step_away(self, state:State, action:Action, modes:list[int]): 
        current_position = state.agent_location
        transition = get_action_transition(action)
        new_position = (current_position[0]+transition[0], current_position[1]+transition[1])
        if new_position not in find_legal_positions(self._tiles, current_position): 
            new_position = current_position

        ghost_mode = state.ghost_mode
        num_desired = 0
        if ghost_mode in modes:
            new_locations = find_legal_positions(self._tiles, new_position)[0]
            for i in new_locations: 
                if self._tiles[i] in [5,6,7,8]:
                    num_desired += 1
            
        return num_desired
    

    def _ghosts_2steps_away(self, state:State, action:Action, modes:list[int]): 
        current_position = state.agent_location
        transition = get_action_transition(action)
        new_position = (current_position[0]+transition[0], current_position[1]+transition[1])
        if new_position not in find_legal_positions(self._tiles, current_position): 
            new_position = current_position

        found_ghosts = set()
        ghost_mode = state.ghost_mode
        
        if ghost_mode in modes:
            step_locations = find_legal_positions(self._tiles, new_position)[0]
            for i in step_locations: 
                step2_locations = find_legal_positions(self._tiles, (i[0],i[1]))[0]
                for j in step2_locations: 
                    if (j[0] >= 0) and (j[1] >= 0) and self._tiles[(j[0], j[1])] in [5,6,7,8]:
                        found_ghosts.add((j[0],j[1]))
            
        return len(found_ghosts)
    

    def _tile_1step_away(self, state:State, action:Action, desired_tiles:list[int]):
        current_position = state.agent_location
        transition = get_action_transition(action)
        new_position = (current_position[0]+transition[0], current_position[1]+transition[1])
        if new_position not in find_legal_positions(self._tiles, current_position): 
            new_position = current_position

        found_tiles = set()

        new_locations = find_legal_positions(self._tiles, new_position)[0]
        for i in new_locations: 
           
            if self._tiles[i] in desired_tiles:
                found_tiles.add(i)
            
        return len(found_tiles)


    def _tile_2steps_away(self, state:State, action:Action, desired_tiles:list[int]):
        current_position = state.agent_location
        transition = get_action_transition(action)
        new_position = (current_position[0]+transition[0], current_position[1]+transition[1])
        if new_position not in find_legal_positions(self._tiles, current_position): 
            new_position = current_position

        found_tiles = set()

        step_locations = find_legal_positions(self._tiles, new_position)[0]
        for i in step_locations: 
            step2_locations = find_legal_positions(self._tiles, (i[0],i[1]))[0]
            for j in step2_locations: 
                if (j[0] >= 0) and (j[1] >= 0) and self._tiles[(j[0], j[1])] in desired_tiles:
                    found_tiles.add((j[0],j[1]))
            
        return len(found_tiles)

