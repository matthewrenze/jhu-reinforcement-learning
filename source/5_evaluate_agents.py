import time
import random
import numpy as np
from tiles.tile_factory import TileFactory
from tiles.tile import Tile
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from models.model_reader import ModelReader
from experiments.results import Results
from experiments.details import Details

# NOTE: Random seeds are in the main loop for reproducibility by treatment

hyperparameters = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.05, 
    "features":[0,2,3,4,5]}
game_level = 10
num_episodes = 100
max_game_steps = 100

treatments = [
    {"agent_name": "sarsa", "use_curriculum": False},
    #{"agent_name": "sarsa", "use_curriculum": True},
    {"agent_name": "q_learning", "use_curriculum": False},
    #{"agent_name": "q_learning", "use_curriculum": True},
    {"agent_name": "approximate_q_learning", "use_curriculum": False},
    #{"agent_name": "approximate_q_learning", "use_curriculum": True},
    {"agent_name": "deep_q_learning", "use_curriculum": False},
    #{"agent_name": "deep_q_learning", "use_curriculum": True}
]

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory()
model_reader = ModelReader()

details = Details()
results = Results()
# NOTE: Only load the results if you are training the agents piece-wise
# results.load("results.csv")

for treatment in treatments:
    random.seed(42)
    np.random.seed(42)
    agent_name = treatment["agent_name"]
    use_curriculum = treatment["use_curriculum"]
    treatment_name = f"{agent_name} ({'curriculum' if use_curriculum else 'baseline'})"
    print(f"Treatment: {treatment_name}")

    model_file_name = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}"
    model = model_reader.load(model_file_name)
    visited_states = set()
    for episode_id in range(num_episodes):
        print(f"Agent: {agent_name} | Curriculum: {use_curriculum} | Game Level: {game_level} | Episode: {episode_id + 1}")
        start_time = time.perf_counter()
        tiles = tile_factory.create(game_level)
        agent = agent_factory.create(agent_name, tiles, hyperparameters)
        house = house_factory.create(game_level)
        ghosts = ghost_factory.create(tiles, house)
        environment = environment_factory.create(tiles, agent, ghosts)

        # approx number of states that are not walls
        total_positions = tiles.shape[0]*tiles.shape[1]
   
        rewards = []
        total_reward = 0
        agent.set_model(model)
        state = environment.get_state()
        steps = 0
        while environment.game_time < max_game_steps:
            steps += 1
            # Tracking visited states for the various agents
            visited_states.add(agent.location)
            action = agent.select_action(state)
            next_state, reward, is_game_over = environment.execute_action(action)
            agent.update(state, action, reward, next_state)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            if is_game_over:
                break

        end_time = time.perf_counter()
        duration = end_time - start_time
        results_row = {
            "agent_name": agent_name,
            "curriculum": use_curriculum,
            "alpha": hyperparameters["alpha"],
            "gamma": hyperparameters["gamma"],
            "epsilon": hyperparameters["epsilon"],
            "game_level": game_level,
            "episode": episode_id + 1,
            "total_time": environment.game_time,
            "avg_reward": np.mean(rewards),
            "percent_states_visited": np.mean(len(visited_states)/total_positions * 100),
            "total_reward": np.sum(rewards),
            "duration": duration/steps}
        results.add(results_row)

results.save("results.csv")