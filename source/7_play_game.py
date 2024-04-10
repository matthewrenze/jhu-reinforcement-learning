import time
import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from environments import environment_renderer as env_renderer
from models.model_reader import ModelReader

np.random.seed(42)

map_level = 10
agent_name = "deep_q_learning"
use_curriculum = False
hyperparameters = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.0}
max_turns = 100
is_interactive = True

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory()

tiles = tile_factory.create(map_level)
agent = agent_factory.create(agent_name, tiles, hyperparameters)
house = house_factory.create(map_level)
ghosts = ghost_factory.create(tiles, house)
environment = environment_factory.create(tiles, agent, ghosts)

model_reader = ModelReader()
model_file_name = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}"
model = model_reader.load(model_file_name)
agent.set_model(model)

total_reward = 0
if is_interactive:
    env_renderer.render(environment, total_reward)
    time.sleep(0.5)

state = environment.get_state()
while environment.game_time < max_turns:
    action = agent.select_action(state)
    next_state, reward, is_game_over = environment.execute_action(action)
    agent.update(state, action, reward, next_state)
    total_reward += reward
    if is_interactive:
        env_renderer.render(environment, total_reward)
        time.sleep(0.5)

    state = next_state
    if is_game_over:
        break