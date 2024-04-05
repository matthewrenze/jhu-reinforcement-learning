import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from models.q_table import QTable
from models.deep_q_network import DeepQNetwork

np.random.seed(42)

agent_name = "sarsa"
hyperparameters = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.1}
use_curriculum = True
num_training_steps = 100_000
training_steps_per_level = 1_000
max_game_steps = 100

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory()

model = None
# NOTE: Only load the model if you are continuing a training run
# if agent_name == "sarsa" or agent_name == "q_learning":
#     model = QTable()
#     model.load(agent_name)
#
# if agent_name == "deep_q_learning":
#     model = DeepQNetwork()
#     model.load(agent_name)


episode_id = 0
training_step_id = 0
while training_step_id < num_training_steps:
    print(f"# Episode: {episode_id + 1}")
    print(f"  Step: {training_step_id + 1}")
    map_level = min((training_step_id // training_steps_per_level + 1), 10) if use_curriculum else 10
    rotation = episode_id % 4 if (use_curriculum and map_level) != 0 else 0
    flip = (episode_id // 4) % 2 == 1 if (use_curriculum and map_level) != 10 else False

    tiles = tile_factory.create(map_level, rotation, flip)
    agent = agent_factory.create(agent_name, tiles, hyperparameters)
    house = house_factory.create(map_level)
    ghosts = ghost_factory.create(tiles, house)
    environment = environment_factory.create(tiles, agent, ghosts)

    agent.set_model(model)

    # details = Details()
    total_reward = 0
    state = environment.get_state()
    while environment.game_time < max_game_steps:
        action = agent.select_action(state)
        next_state, reward, is_game_over = environment.execute_action(action)
        agent.update(state, action, reward, next_state)
        total_reward += reward
        state = next_state
        details_row = {
            "agent_name": agent_name,
            "curriculum": "NA",
            "alpha": hyperparameters["alpha"],
            "gamma": hyperparameters["gamma"],
            "epsilon": hyperparameters["epsilon"],
            "mode": "train",
            "episode": episode_id,
            "game_level": map_level,
            "time_step": environment.game_time,
            "reward": reward,
            "total_reward": total_reward}
        # details.add(details_row)
        training_step_id += 1
        if is_game_over:
            break

    agent.learn()
    episode_id += 1

    # details.save()
    model = agent.get_model()

model.save(agent_name)
