import time
import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from environments import environment_renderer as env_renderer
from experiments.results import Results
from models.model import Model
from models.q_table import QTable

np.random.seed(42)

map_level = 99
agent_name = "sarsa"
hyperparameters = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.2}
use_curriculum = False
max_turns = 100
model = None

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory()

# TODO: This is just a hack to allow you to play using a trained model
# TODO: This should be refactored to a more flexible solution
if agent_name == "sarsa" or agent_name == "q_learning":
    model = QTable()
    model.load(agent_name)

def play(model, episode: int, is_interactive: bool) -> Model:

    tiles = tile_factory.create(map_level)
    agent = agent_factory.create(agent_name, tiles, hyperparameters)
    house = house_factory.create()
    ghosts = ghost_factory.create(tiles, house)
    environment = environment_factory.create(tiles, agent, ghosts)

    agent.set_model(model)

    # details = Details()
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
        details_row = {
            "agent_name": agent_name,
            "curriculum": use_curriculum,
            "alpha": hyperparameters["alpha"],
            "gamma": hyperparameters["gamma"],
            "epsilon": hyperparameters["epsilon"],
            "mode": "train",
            "episode": episode,
            "game_level": map_level,
            "time_step": environment.game_time,
            "reward": reward,
            "total_reward": total_reward}
        # details.add(details_row)
        if is_game_over:
            break

    # details.save()
    return agent.get_model()

results = Results()
results.load()

for i in range(1000):
    print(f"Training run {i + 1}")
    play(model, i + 1, False)

print("Final run")
play(model,1, True)

results.save()