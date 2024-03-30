import os
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from models.model import Model
from agents.replay_buffer import ReplayBuffer

FOLDER_PATH = "../data/models"

class DeepQNetwork(Model):

    def __init__(self, model: MLPRegressor = None, replay_buffer: ReplayBuffer = None, ):
        self.model = model
        self.replay_buffer = replay_buffer

    def load(self, agent_name: str):
        file_name = f"{FOLDER_PATH}/{agent_name}.joblib"
        if os.path.exists(file_name):
            self.model = load(file_name)
            self.replay_buffer = ReplayBuffer()

    def save(self, agent_name: str) -> None:
        file_name = f"{FOLDER_PATH}/{agent_name}.joblib"
        dump(self.model, file_name)