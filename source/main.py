import time
import numpy as np
from environments.environment_factory import EnvironmentFactory
from environments import environment_renderer as env_renderer
from agents.agent_factory import AgentFactory

np.random.seed(42)

environment_factory = EnvironmentFactory()
agent_factory = AgentFactory()

environment = environment_factory.create(99)
agent = agent_factory.create("random")

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

print("Game Over!")