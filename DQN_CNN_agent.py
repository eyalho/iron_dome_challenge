import random
from collections import deque

import numpy as np
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten
from keras.optimizers import Adam
from keras.models import Model


# Deep Q-learning `Agent
class DQN_CNN_agent:
    def __init__(self, action_size=4, env=None):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.env = env
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_frames = Input(shape=(self.env.py, self.env.px, 3*self.env.frame_history))
        input_ang = Input(shape=(1,))
        first_Conv2D = Conv2D(input_shape=(42,84,12) , filters=16, kernel_size=8, strides=(2,4),  activation='relu')(input_frames)
        second_Conv2D = Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')(first_Conv2D)
        flat = Flatten()(second_Conv2D)
        dense_output = Dense(256, activation='relu')(flat)
        mid_layer = concatenate([dense_output, input_ang])
        out_layer = Dense(self.action_size, activation='linear')(mid_layer)
        model = Model(
            inputs=[input_frames, input_ang],
            outputs=out_layer, name="model_full_state")
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        #from keras.utils import plot_model
        #plot_model(model, to_file='model.png')

        return model

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict([state[0], state[1]])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * \
                     np.amax(self.model.predict([next_state[0], next_state[1]]))
            target_f = self.model.predict([state[0], state[1]])

            target_f[0][action] = target
            self.model.fit([state[0], state[1]], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay