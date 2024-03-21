import time
import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from environments import environment_renderer as env_renderer

np.random.seed(42)

map_level = 1
agent_name = "sarsa"
hyperparameters = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.2}

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory(tile_factory, agent_factory, house_factory, ghost_factory)

max_turns = 100

def run_trial(is_interactive: bool):
    environment = environment_factory.create(map_level, agent_name, hyperparameters)
    agent = environment.agent
    total_reward = 0
    if is_interactive:
        env_renderer.render(environment, total_reward)
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

    if (agent_name == "sarsa"):
        agent_factory.save(agent_name, agent.q_table)

for i in range(1000):
    print(f"Trial {i + 1}")
    run_trial(False)

print("Final Trial")
run_trial(True)