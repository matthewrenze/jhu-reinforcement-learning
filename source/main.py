import time
import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from environments import environment_renderer as env_renderer

np.random.seed(42)

environment_factory = EnvironmentFactory(TileFactory(), AgentFactory(), HouseFactory(), GhostFactory())
environment = environment_factory.create(99, "human")
agent = environment.agent

max_turns = 100
total_reward = 0

env_renderer.render(environment, total_reward)
state = environment.get_state()

for i in range(100):
    action = agent.select_action(state)
    next_state, reward, is_game_over = environment.execute_action(action)
    total_reward += reward
    env_renderer.render(environment, total_reward)
    state = next_state
    time.sleep(0.5)
    if is_game_over:
        break