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
          #  .  #  o  #
          #  p  c  .  #
          #  #  #  #  #
    '''

    tiles = np.array([[1, 1, 1, 1, 1], [1, 3, 3, 3, 1], [1, 3, 1, 4, 1], [1, 7, 2, 3, 1], [1, 1, 1, 1, 1]])
    state = Mock()
    state.agent_location = (3,2)
    state.ghost_mode = 1
    state.is_invincible = True
    state.tiles = tiles
   
    feature_extraction = FeatureExtraction(state)
    return feature_extraction

def test_distance_closest_food(setup): 
    feature_extraction = setup
    food_distance = feature_extraction.distance_closest_food()
    assert food_distance == 1

def test_distance_closest_ghost(setup):
    feature_extraction = setup
    ghost_distance = feature_extraction.distance_closest_ghost()
    assert ghost_distance == 1

def test_number_active_ghosts_1step(setup): 
    feature_extraction = setup
    num_ghosts = feature_extraction.number_active_ghosts_1step()
    assert num_ghosts == 0

def test_number_scared_ghosts_1step(setup): 
    feature_extraction = setup
    num_ghosts = feature_extraction.number_scared_ghosts_1step()
    assert num_ghosts == 1
    
def test_number_active_ghosts_2step(setup): 
    feature_extraction = setup
    num_ghosts = feature_extraction.number_active_ghosts_2step()
    assert num_ghosts == 0

def test_number_scared_ghosts_2step(setup): 
    feature_extraction = setup
    num_ghosts = feature_extraction.number_scared_ghosts_2step()
    assert num_ghosts == 0

def test_number_power_pellets_1step(setup): 
    feature_extraction = setup
    num_pellets = feature_extraction.number_power_pellets_1step()
    assert num_pellets == 0

def test_number_power_pellets_2steps(setup): 
    feature_extraction = setup
    num_pellets = feature_extraction.number_power_pellets_2steps()
    assert num_pellets == 1

def test_number_food_1step(setup):
    feature_extraction = setup
    num_food = feature_extraction.number_food_1step()
    assert num_food == 1

def test_number_food_2steps(setup):
    feature_extraction = setup
    num_food = feature_extraction.number_food_2steps()
    assert num_food == 1