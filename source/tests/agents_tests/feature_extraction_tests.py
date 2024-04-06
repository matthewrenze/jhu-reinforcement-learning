import pytest
from unittest.mock import Mock
import numpy as np
from agents.feature_extraction import FeatureExtraction
from tiles.test_tiles import TestTiles
from tiles.tile import Tile
from ghosts.ghost import Mode

@pytest.fixture()
def setup():

    '''   #  #  #  #  #
          #  .  .  .  #
          #  .  #  s  #
          #  p  c     #
          #  #  #  #  #
    '''

    tiles = TestTiles.create([[1, 1, 1, 1, 1], [1, 3, 3, 3, 1], [1, 3, 1, 5, 1], [1, 7, 2, 0, 1], [1, 1, 1, 1, 1]])
    state = Mock()
    state.agent_location = (3,2)
    state.ghost_locations = [(Tile.PINKY.id, (3, 1)), (Tile.STATIC.id, (2, 3))]
    state.ghost_mode = [Mode.FRIGHTENED, Mode.CHASE]
   
    feature_extraction = FeatureExtraction(tiles, state)
    return tiles, feature_extraction

def test_distance_closest_food(setup): 
    _, feature_extraction = setup
    food_distance = feature_extraction.distance_closest_food()
    assert food_distance == 2

def test_distance_closest_ghost(setup):
    _, feature_extraction = setup
    ghost_distance = feature_extraction.distance_closest_ghost()
    assert ghost_distance == 1

def test_number_active_ghosts_1step(setup): 
    _, feature_extraction = setup
    num_ghosts = feature_extraction.number_active_ghosts_1step()
    assert num_ghosts == 0 

def test_number_scared_ghosts_1step(setup): 
    _, feature_extraction = setup
    num_ghosts = feature_extraction.number_scared_ghosts_1step()
    assert num_ghosts == 1
    
def test_number_active_ghosts_2step(setup): 
    tiles, feature_extraction = setup
    num_ghosts = feature_extraction.number_active_ghosts_2step()
    assert num_ghosts == 1

def test_number_scared_ghosts_2step(setup): 
    tiles, feature_extraction = setup
    num_ghosts = feature_extraction.number_scared_ghosts_2step()
    assert num_ghosts == 0

# def test_safety_mode(setup): 
#     tiles, feature_extraction = setup
#     safety = feature_extraction.safety_mode(tiles)
#     assert safety == True 