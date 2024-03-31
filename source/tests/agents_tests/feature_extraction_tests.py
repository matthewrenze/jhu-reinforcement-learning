import pytest
from unittest.mock import Mock
import numpy as np
from agents.feature_extraction import FeatureExtraction
from tiles.test_tiles import TestTiles
from tiles.tile import Tile
from states.state import State

@pytest.fixture()
def setup():

    '''   #  #  #  #  #
          #  .  s  .  #
          #  .  #  o  #
          #  .  c  .  #
          #  #  #  #  #
    '''

    tiles = TestTiles.create([[1, 1, 1, 1, 1], [1, 3, 5, 3, 1], [1, 3, 1, 4, 1], [1, 3, 2, 3, 1], [1, 1, 1, 1, 1]])
    state = Mock()
    state.agent_location = (3,2)
    feature_extraction = FeatureExtraction(tiles, state)
    return tiles, feature_extraction

def test_distance_closest_food(setup): 
    _, feature_extraction = setup
    food_distance = feature_extraction.distance_closest_food()
    assert food_distance == 1

def test_distance_closest_ghost(setup):
    _, feature_extraction = setup
    ghost_distance = feature_extraction.distance_closest_ghost()
    assert ghost_distance == 4

def test_number_active_ghosts_1step(setup): 
    tiles, feature_extraction = setup
    num_ghosts = feature_extraction.number_active_ghosts_1step(tiles)
    assert num_ghosts == 0 

def test_number_active_ghosts_2step(setup): 
    tiles, feature_extraction = setup
    num_ghosts = feature_extraction.number_active_ghosts_2step(tiles)
    assert num_ghosts == 0

def test_number_scared_ghosts_1step(setup): 
    tiles, feature_extraction = setup
    num_ghosts = feature_extraction.number_scared_ghosts_1step(tiles)
    assert num_ghosts == 0 

def test_number_scared_ghosts_2step(setup): 
    tiles, feature_extraction = setup
    num_ghosts = feature_extraction.number_scared_ghosts_2step(tiles)
    assert num_ghosts == 0

def test_safety_mode(setup): 
    tiles, feature_extraction = setup
    safety = feature_extraction.safety_mode(tiles)
    assert safety == True 