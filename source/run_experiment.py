import time
import random
import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from environments import environment_renderer as env_renderer
from experiments.results import Results
from experiments.details import Details

# NOTE: Random seeds have been moved into the main loop for reproducibility by treatment

hyperparameters = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.05}
num_training_steps = 100_000
training_steps_per_level = 1_000
max_game_steps = 100

treatments = [
    {"agent_name": "sarsa", "use_curriculum": False},
    {"agent_name": "sarsa", "use_curriculum": True},
    {"agent_name": "q_learning", "use_curriculum": False},
    {"agent_name": "q_learning", "use_curriculum": True},
    {"agent_name": "deep_q_learning", "use_curriculum": False},
    {"agent_name": "deep_q_learning", "use_curriculum": True}
]

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory()

details = Details()
results = Results()
# NOTE: Only load the results if you are running the experiment piece-wise
# results.load("results.csv")

for treatment in treatments:
    random.seed(42)
    np.random.seed(42)
    agent_name = treatment["agent_name"]
    use_curriculum = treatment["use_curriculum"]
    print(f"Treatment: {agent_name} ({'curriculum' if use_curriculum else 'baseline'})")

    model = None
    episode_id = 0
    training_step_id = 0
    while training_step_id < num_training_steps:
        print(f"# Episode: {episode_id + 1}")
        map_level = min((training_step_id // 100 + 1), 10) if use_curriculum else 10

        tiles = tile_factory.create(map_level)
        agent = agent_factory.create(agent_name, tiles, hyperparameters)
        house = house_factory.create(map_level)
        ghosts = ghost_factory.create(tiles, house)
        environment = environment_factory.create(tiles, agent, ghosts)

        total_reward = 0
        agent.set_model(model)
        state = environment.get_state()
        while environment.game_time < max_game_steps:
            print(f"Training step: {training_step_id + 1}")
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
                "game_level": map_level,
                "episode": episode_id + 1,
                "training_step": training_step_id + 1,
                "game_step": environment.game_time,
                "reward": reward,
                "total_reward": total_reward}
            details.add(details_row)
            training_step_id += 1
            if is_game_over:
                break

        agent.learn()
        model = agent.get_model()

        results_row = {
            "agent_name": agent_name,
            "curriculum": use_curriculum,
            "alpha": hyperparameters["alpha"],
            "gamma": hyperparameters["gamma"],
            "epsilon": hyperparameters["epsilon"],
            "episode": episode_id + 1,
            "game_level": map_level,
            "total_time": environment.game_time,
            "avg_reward": details.table["reward"].mean(),
            "total_reward": details.table["reward"].sum()}
        results.add(results_row)
        episode_id += 1

model.save(agent_name)
details.save("details.csv")
results.save("results.csv")
