# naive
"""
The flag simulate_reward tell the trainer to do so..
"""

from agents.naive_full_state_model_agent import NaiveFullStateModelAgent


def create_agent():
    return NaiveFullStateModelAgentSimulateReward()


class NaiveFullStateModelAgentSimulateReward(NaiveFullStateModelAgent):

    def __init__(self):
        super().__init__()  # call the __init__ of parent
        self.name = "NaiveFullStateModelAgentSimulateReward"
        self.simulate_reward = True
