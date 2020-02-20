# Abstract agent
# Any custom-agent should be declared by "class agent(ABSDQNAgent)"
# Then he must implement all abstractmethod.
# He may override the other methods if he want to..
#
# Moreover every agent_file should declare a func "create_agent" so trainer can load it.

import random
from collections import deque

import numpy as np
from abc import ABC, abstractmethod
from keras.utils import plot_model


class ABSDQNAgent(ABC):
    def __init__(self, actions_size=4):
        # self.state_size = state_size
        self.action_size = actions_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self):
        """ return a Neural Net for Deep-Q learning Model"""
        pass

    @abstractmethod
    def init_state(self):
        """ return a valid state for stp=0"""
        pass

    @abstractmethod
    def create_state(self, r_locs, i_locs, c_locs, ang, score, stp):
        """ convert the default env.state to a valid input for the model agent"""
        pass

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        return the decided action
        First, explore by a random/custom action..
        Later on, let the agent predict by itself (with the default epsilons it's take about 1000 calls)
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, np.array(target_f), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_model(self, file_path):
        plot_model(self.model, to_file=file_path)
