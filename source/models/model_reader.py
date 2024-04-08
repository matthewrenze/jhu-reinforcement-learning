import numpy as np
from joblib import load
from models.model import Model
from models.q_table import QTable
from models.deep_q_network import DeepQNetwork
from agents.replay_buffer import ReplayBuffer

FOLDER_PATH = "../data/models"

class ModelReader:

    def load(self, file_name: str) -> Model:

        # NOTE: Deep Q-Learning needs to go first since "q_learning" is also in the file name
        if "deep_q_learning" in file_name:
            file_name = f"{FOLDER_PATH}/{file_name}.joblib"
            mlp_model = load(file_name)
            replay_buffer = ReplayBuffer()
            model = DeepQNetwork(mlp_model, replay_buffer)
            return model

        elif "sarsa" in file_name or "q_learning" in file_name:
            file_name = f"{FOLDER_PATH}/{file_name}.csv"
            table = np.loadtxt(file_name, delimiter=",")
            model = QTable(table)
            return model

        else:
            raise ValueError(f"Unknown model type: {file_name}")
