import time
import numpy as np
from tiles.tile_factory import TileFactory
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from environments import environment_renderer as env_renderer
from experiments.results import Results
from experiments.details import Details

np.random.seed(42)

hyperparameters = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.05}
num_episodes = 10000
max_turns = 100

treatments = [
    {"agent_name": "sarsa", "use_curriculum": False},
    {"agent_name": "sarsa", "use_curriculum": True}]

tile_factory = TileFactory()
agent_factory = AgentFactory()
house_factory = HouseFactory()
ghost_factory = GhostFactory()
environment_factory = EnvironmentFactory()

# NOTE: Only load the results if you are running the experiment in pieces
results = Results()
# results.load()

for treatment in treatments:
    agent_name = treatment["agent_name"]
    use_curriculum = treatment["use_curriculum"]
    model = None

    for episode_id in range(num_episodes):
        print(f"Training run {episode_id + 1}")
        is_interactive = True if episode_id == (num_episodes - 1) else False
        map_level = (episode_id // (num_episodes // 10) + 1) if use_curriculum else 10

        tiles = tile_factory.create(map_level)
        agent = agent_factory.create(agent_name, tiles, hyperparameters)
        house = house_factory.create(map_level)
        ghosts = ghost_factory.create(tiles, house)
        environment = environment_factory.create(tiles, agent, ghosts)

        agent.set_model(model)

        total_reward = 0
        if is_interactive:
            env_renderer.render(environment, total_reward)

        details = Details()
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
                "episode": episode_id,
                "game_level": map_level,
                "time_step": environment.game_time,
                "reward": reward,
                "total_reward": total_reward}
            details.add(details_row)
            if is_game_over:
                break

        # NOTE: Saving details after each episode slows down the training process
        # NOTE: Only use for debugging purposes
        # details.save()
        model = agent.get_model()

        results_row = {
            "agent_name": agent_name,
            "curriculum": use_curriculum,
            "alpha": hyperparameters["alpha"],
            "gamma": hyperparameters["gamma"],
            "epsilon": hyperparameters["epsilon"],
            "episode": episode_id,
            "game_level": map_level,
            "total_time": environment.game_time,
            "avg_reward": details.table["reward"].mean(),
            "total_reward": details.table["reward"].sum()}
        results.add(results_row)

results.save()
