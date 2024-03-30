import numpy as np
from sklearn.neural_network import MLPRegressor
from agents.replay_buffer import ReplayBuffer
from models.deep_q_network import DeepQNetwork
from agents.agent import Agent
from actions.action import Action
from states.state import State

class DeepQLearningAgent(Agent):

        def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float]):
            super().__init__(location, hyperparameters)
            self.alpha = hyperparameters["alpha"]
            self.gamma = hyperparameters["gamma"]
            self.epsilon = hyperparameters["epsilon"]
            self.num_states = 251
            self.num_actions = 5
            self.batch_size = 32
            self.buffer_size = 1000
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.num_epochs = 10
            self.model = MLPRegressor(
                hidden_layer_sizes=(128),
                max_iter=1,
                warm_start=True,
                verbose=False)
            self.model.partial_fit([np.zeros(self.num_states)], [np.zeros(self.num_actions)])

        def select_action(self, state: State) -> Action:
            flat_state = self._convert_state(state)
            if self._get_random_threshold() < self.epsilon:
                action_id = self._get_random_action_id()
            else:
                q_values = self.model.predict([flat_state])[0]
                action_id = np.argmax(q_values)
            return Action(action_id)

        def update(self, state: State, action: Action, reward: int, next_state: State) -> None:
            flat_state = self._convert_state(state)
            flat_next_state = self._convert_state(next_state)
            self.replay_buffer.add(flat_state, action, reward, flat_next_state, False)

        def learn(self) -> None:
            if len(self.replay_buffer) < self.batch_size:
                return
            for epoch_id in range(self.num_epochs):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                q_values = self.model.predict(states)
                next_q_values = self.model.predict(next_states)
                for epoch_id in range(self.batch_size):
                    if dones[epoch_id]:
                        q_values[epoch_id][actions[epoch_id].value] = rewards[epoch_id]
                    else:
                        q_values[epoch_id][actions[epoch_id].value] = rewards[epoch_id] + self.gamma * np.max(next_q_values[epoch_id])

                self.model.partial_fit(states, q_values)

        def get_model(self) -> object:
            model = DeepQNetwork(self.model, self.replay_buffer)
            return model

        def set_model(self, model: DeepQNetwork) -> None:
            if model is not None:
                self.model = model.model
                self.replay_buffer = model.replay_buffer

        def _get_random_threshold(self) -> float:
            return np.random.uniform()

        def _get_random_action_id(self) -> int:
            return np.random.choice(self.num_actions)

        def _convert_state(self, state: State) -> np.ndarray:

            max_height = 5
            max_width = 5
            neighbor_tiles = np.zeros((max_height, max_width))
            agent_location = state.agent_location
            tiles = state.tiles
            height = tiles.shape[0]
            width = tiles.shape[1]
            for row in range(5):
                for col in range(5):
                    neighbor_tiles[row, col] = tiles[(agent_location[0] + row - 2) % height][(agent_location[1] + col - 2) % width]

            one_hot_tiles = np.zeros((10, max_height, max_width), dtype=int)
            for tile_id in range(10):
                one_hot_tiles[tile_id] = (neighbor_tiles == tile_id).astype(int)

            flat_tiles = one_hot_tiles.flatten()

            # Append the is_invincible flag to the end of the state
            flat_state = np.append(flat_tiles, int(state.is_invincible))

            return flat_state

