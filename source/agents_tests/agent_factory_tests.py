import pytest
import numpy as np
from agents.agent_factory import AgentFactory
from tiles.tiles import Tiles
from tiles.test_tiles import TestTiles
from states.state import State

@pytest.fixture
def setup():
    tiles = TestTiles.create([[0, 1], [2, 3]])
    factory = AgentFactory()
    return tiles, factory


def test_get_agent_location(setup):
    tiles, factory = setup
    agent_location = factory._get_agent_location(tiles)
    expected_location = (1, 0)
    assert agent_location == expected_location

def test_create_human_agent(setup):
    tiles, factory = setup
    agent = factory.create("human", tiles)
    assert agent.__class__.__name__ == "HumanAgent"
    assert agent.location == (1, 0)

def test_create_random_agent(setup):
    tiles, factory = setup
    agent = factory.create("random", tiles)
    assert agent.__class__.__name__ == "RandomAgent"
    assert agent.location == (1, 0)