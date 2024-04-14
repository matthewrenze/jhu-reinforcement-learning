import time
import random
import numpy as np
from tiles.tile_factory import TileFactory
from tiles.tile import Tile
from agents.agent_factory import AgentFactory
from ghosts.ghost_factory import GhostFactory
from houses.house_factory import HouseFactory
from environments.environment_factory import EnvironmentFactory
from models.model_writer import ModelWriter
from experiments.details import Details

# NOTE: Random seeds are in the main loop for reproducibility by treatment

num_training_steps = 10_000
training_steps_per_level = 200
max_game_steps = 100

treatments = [
    {"agent_name": "deep_q_learning", "use_curriculum": False, "alpha": 0.95, "gamma": 0.9, "epsilon": 0.1, "features":[]},
    {"agent_name": "approximate_q_learning", "use_curriculum": False, "alpha": 0.05, "gamma": 0.9, "epsilon": 0.1, "features":[0,2,3,4,5]},
    {"agent_name": "sarsa", "use_curriculum": False, "alpha": 0.1, "gamma": 0.9, "epsilon": 0.1, "features":[]},
    #{"agent_name": "sarsa", "use_curriculum": True, "alpha": 0.05, "gamma": 0.9, "epsilon": 0.1},
    {"agent_name": "q_learning", "use_curriculum": False, "alpha": 0.1, "gamma": 0.95, "epsilon": 0.1, "features":[]},
    #{"agent_name": "q_learning", "use_curriculum": True, "alpha": 0.1, "gamma": 0.95, "epsilon": 0.1},
    #{"agent_name": "approximate_q_learning", "use_curriculum": True, "alpha": 0.1, "gamma": 0.9, "epsilon": 0.1},
    #{"agent_name": "deep_q_learning", "use_curriculum": True, "alpha": 0.95, "gamma": 0.9, "epsilon": 0.1}
]

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
    agent_name = treatment["agent_name"]
    use_curriculum = treatment["use_curriculum"]
    treatment_name = f"{agent_name} ({'curriculum' if use_curriculum else 'baseline'})"
    print(f"Treatment: {treatment_name}")

    model = None
    episode_id = 0
    training_step_id = 0
    visited_states = set()
    while training_step_id < num_training_steps:
        game_level = min((training_step_id // training_steps_per_level + 1), 10) if use_curriculum else 10
        rotation = episode_id % 4 if (use_curriculum and game_level) != 0 else 0
        flip = (episode_id // 4) % 2 == 1 if (use_curriculum and game_level) != 10 else False

        hyperparameters = {
            "alpha": treatment["alpha"],
            "gamma": treatment["gamma"],
            "epsilon": treatment["epsilon"], 
            "features": treatment["features"]}

        tiles = tile_factory.create(game_level, rotation, flip)
        agent = agent_factory.create(agent_name, tiles, hyperparameters)
        house = house_factory.create(game_level)
        ghosts = ghost_factory.create(tiles, house)
        environment = environment_factory.create(tiles, agent, ghosts)
        
        # approx number of states that are not walls
        total_positions = tiles.shape[0]*tiles.shape[1]
        wall_positions = int(np.sum(tiles == Tile.WALL))

        total_reward = 0
        agent.set_model(model)
        state = environment.get_state()
        while environment.game_time < max_game_steps:
            print(f"Agent: {agent_name} | Curriculum: {use_curriculum} | Game Level: {game_level} | Episode: {episode_id + 1} | Training Step: {training_step_id + 1}")
            
            # Tracking visited states for the various agents
            if agent_name == "sarsa" or agent_name == "q_learning":
                visited_states.add(agent._convert_state(environment.get_state()))
                possible_states = np.round(20000 * (total_positions - wall_positions)/total_positions)
            elif agent_name == "deep_q_learning": 
                visited_states.add(agent._convert_state(environment.get_state()).tobytes())
                possible_states = 81
            elif agent_name == "approximate_q_learning":
                visited_states.add(agent.location)
                possible_states = total_positions - wall_positions

            num_states = len(visited_states)
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
                "percent_states_visited": num_states/possible_states * 100,
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

    model_name = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}"
    model_writer.write(model_name, model)

    folder_path = "../data/training"
    file_name = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}"
    file_path = f"{folder_path}/{file_name}.csv"
    details.save(file_path)