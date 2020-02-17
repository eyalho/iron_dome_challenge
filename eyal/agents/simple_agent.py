# Deep Q-learning Agent
"""
We figured out that this the naive though is too hard for a short time training..
So, next we tried to create as simple network as we could
"""
import random
from collections import deque

import numpy as np
from keras.layers import Dense, Input, concatenate
from keras.models import Model


class DQNAgent:
    def __init__(self, action_size=4):
        # self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_simple_model()

    def _build_simple_model(self):
        hidden_size = 24
        # Input Layer
        ang_input_layer = Input(shape=(1,), name="angle")  # Turret angle (ang)
        sim_score_input_layer = Input(shape=(1,), name="simulate_score")  # Turret angle (ang)
        time_input_layer = Input(shape=(1,), name="time")  # time_t
        layer = concatenate(
            [ang_input_layer, sim_score_input_layer, time_input_layer])
        layer = Dense(hidden_size, activation='relu')(layer)
        layer = Dense(hidden_size, activation='relu')(layer)
        output_layer = Dense(self.action_size, activation='linear')(layer)
        model = Model(
            inputs=[ang_input_layer, sim_score_input_layer, time_input_layer],
            outputs=output_layer, name="model_simple")
        model.compile(optimizer='adam', loss='mse')
        # from keras.utils import plot_model
        # plot_model(model, to_file=model.name)
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
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

    def init_state(self):
        state = [np.array([0]), np.array([0]), np.array([0])]
        return state
