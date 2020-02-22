# Deep Q-learning Agent
"""
We figured out that this the naive though is too hard for a short time training..
So, next we tried to create as simple network as we could
"""
import random

from agents.abstract_agent import ABSDQNAgent


def create_agent():
    return RandomAgent()


class RandomAgent(ABSDQNAgent):
    def __init__(self):
        super().__init__()  # call the __init__ of parent
        self.name = "RandomAgent"

    def _build_model(self):
        class Model:
            def save(self):
                pass

            def predict(self):
                pass

        return Model

    def init_state(self):
        return None

    def create_state(self, conf, r_locs, i_locs, c_locs, ang, reward, stp):
        return None

    def memorize(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        return random.randint(0, 3)

    def replay(self, batch_size):
        pass

    def plot_model(self, file_path):
        pass
