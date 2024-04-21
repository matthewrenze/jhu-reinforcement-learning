import time
import random
import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from models.model_writer import ModelWriter
from experiments.details import Details
from environments import environment_renderer

# NOTE: Random seeds are in the main loop for reproducibility by treatment

# Note: Change agent_name and curriculum (below) to specify the treatment
agent_name = "approximate_q_learning"
use_curriculum = True
num_training_steps = 100_000
training_steps_per_level = 200
max_game_steps = 100

standard_treatments = [
    {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1},
    {"alpha": 0.05, "gamma": 0.9, "epsilon": 0.1},
    {"alpha": 0.2, "gamma": 0.9, "epsilon": 0.1},
    {"alpha": 0.1, "gamma": 0.8, "epsilon": 0.1},
    {"alpha": 0.1, "gamma": 0.95, "epsilon": 0.1},
    {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.05},
    {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.2},
]

# Note: Treatments specifically for approximate Q-learning agents
approximate_treatments = [
    {"alpha": 0.005, "gamma": 0.9, "epsilon": 0.1, "features": [0, 1, 2, 3, 4, 5]},
    {"alpha": 0.05, "gamma": 0.9, "epsilon": 0.1, "features": [0, 1, 2, 3, 4, 5]},
    {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1, "features": [0, 1, 2, 3, 4, 5]},
    {"alpha": 0.005, "gamma": 0.8, "epsilon": 0.3, "features": [0, 1, 2, 3, 4, 5]},
    {"alpha": 0.05, "gamma": 0.8, "epsilon": 0.3, "features": [0, 1, 2, 3, 4, 5]},
    {"alpha": 0.1, "gamma": 0.8, "epsilon": 0.3, "features": [0, 1, 2, 3, 4, 5]},
]

# Note: Treatments specifically for deep Q-learning agents
dqn_treatments = [
    {"alpha": 0.95, "gamma": 0.9, "epsilon": 0.1},
    {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.1},
    {"alpha": 1.0, "gamma": 0.9, "epsilon": 0.1},
    {"alpha": 0.95, "gamma": 0.8, "epsilon": 0.1},
    {"alpha": 0.95, "gamma": 0.95, "epsilon": 0.1},
    {"alpha": 0.95, "gamma": 0.9, "epsilon": 0.05},
    {"alpha": 0.95, "gamma": 0.9, "epsilon": 0.2}
]

if agent_name == "approximate_q_learning":
    treatments = approximate_treatments
elif agent_name == "deep_q_learning":
    treatments = dqn_treatments
else:
    treatments = standard_treatments

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory()
model_writer = ModelWriter()

details = Details()

for treatment in treatments:
    random.seed(42)
    np.random.seed(42)

    alpha = treatment["alpha"]
    gamma = treatment["gamma"]
    epsilon = treatment["epsilon"]
    features = treatment.get("features")
    hyperparameters = {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon, 
        "features": features}

    treatment_name = f"Alpha: {alpha} | Gamma: {gamma} | Epsilon: {epsilon}"
    print(f"Treatment: {treatment_name}")

    model = None
    episode_id = 0
    training_step_id = 0
    while training_step_id < num_training_steps:
        game_level = min((training_step_id // training_steps_per_level + 1), 10) if use_curriculum else 10
        rotation = episode_id % 4 if (use_curriculum and game_level != 10) else 0
        flip = (episode_id // 4) % 2 == 1 if (use_curriculum and game_level != 10) else False

        tiles = tile_factory.create(game_level, rotation, flip)
        agent = agent_factory.create(agent_name, tiles, hyperparameters)
        house = house_factory.create(game_level)
        ghosts = ghost_factory.create(tiles, house)
        environment = environment_factory.create(tiles, agent, ghosts)

        total_reward = 0
        agent.set_model(model)
        state = environment.get_state()
        while environment.game_time < max_game_steps:
            print(f"Agent: {agent_name} | Curriculum: {use_curriculum} | α={alpha}; γ={gamma}; ε={epsilon} | Game Level: {game_level} | Episode: {episode_id + 1} | Training Step: {training_step_id + 1}")
            start_time = time.perf_counter()
            action = agent.select_action(state)
            next_state, reward, is_game_over = environment.execute_action(action)
            agent.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            end_time = time.perf_counter()
            duration = end_time - start_time
            details_row = {
                "agent_name": agent_name,
                "curriculum": use_curriculum,
                "alpha": hyperparameters["alpha"],
                "gamma": hyperparameters["gamma"],
                "epsilon": hyperparameters["epsilon"],
                "game_level": game_level,
                "episode": episode_id + 1,
                "training_step": training_step_id + 1,
                "game_step": environment.game_time,
                "reward": reward,
                "total_reward": total_reward,
                "duration": duration}
            details.add(details_row)
            training_step_id += 1
            if is_game_over:
                break

        agent.learn()
        model = agent.get_model()
        episode_id += 1

    # model_file_name = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}"
    # model_writer.write(model_file_name, model)

folder_path = "../data/hyperparameters"
file_name = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}.csv"
file_path = f"{folder_path}/{file_name}"
details.save(file_path)
