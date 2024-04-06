from sklearn.neural_network import MLPRegressor
from models.model import Model
from agents.replay_buffer import ReplayBuffer

FOLDER_PATH = "../data/models"

class DeepQNetwork(Model):

    def __init__(self, model: MLPRegressor = None, replay_buffer: ReplayBuffer = None, ):
        self.model = model
        self.replay_buffer = replay_buffer