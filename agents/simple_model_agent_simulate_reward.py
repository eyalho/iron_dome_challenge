# Deep Q-learning Agent
"""
The flag simulate_reward tell the trainer to do so..
"""

from agents.simple_model_agent import SimpleModelDQNAgent


def create_agent():
    return SimpleModelAgentSimulateReward()


class SimpleModelAgentSimulateReward(SimpleModelDQNAgent):

    def __init__(self):
        super().__init__()  # call the __init__ of parent
        self.name = "SimpleModelAgentSimulateReward"
        self.simulate_reward = True
