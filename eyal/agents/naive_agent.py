# naive
"""
Actually we were the naive not the agent.
Firstly, we tried just give all state data "as is" to the agent and let it figure out by itself
how to become the best player
"""
import random
from collections import deque

import numpy as np
from keras.layers import Dense, Input, LSTM, concatenate
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
        self.model = self._build_model()

    def _build_model(self):
        """
        original model was:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

        We change it in order to fit our problem:
        - 4 chanel input (r_locs, i_locs, c_locs, ang)
        - Added LSTM for remembering older locs states
        -
        :return:
        """

        # Neural Net for Deep-Q learning Model
        hidden_size = 20

        # Input Layer
        r_locs_input_layer = Input(shape=(None, 2),
                                   name="rockets")  # Location of each rocket np.array([[x,y],[x,y], ...])
        i_locs_input_layer = Input(shape=(None, 2),
                                   name="interceptor")  # Location of each interceptor np.array([[x,y],[x,y], ...])
        c_locs_input_layer = Input(shape=(None, 2), name="cities")  # Location of each city np.array([[x,y],[x,y], ...])
        ang_input_layer = Input(shape=(1,), name="angle")  # Turret angle (ang)
        features_input_layer = Input(shape=(1,), name="other_features")  # time_t

        # Add RNN with some memory to previous states
        r_locs_lstm_layer = LSTM(hidden_size)(r_locs_input_layer)  # institution: find worst rocket
        i_locs_lstm_layer = LSTM(hidden_size)(i_locs_input_layer)  # institution: find match interceptor
        c_locs_lstm_layer = LSTM(hidden_size)(c_locs_input_layer)  # institution: no real need for lstm

        layer = concatenate(
            [r_locs_lstm_layer, i_locs_lstm_layer, c_locs_lstm_layer, ang_input_layer, features_input_layer])
        layer = Dense(hidden_size, activation='linear')(layer)
        output_layer = Dense(self.action_size, activation='linear')(layer)
        model = Model(
            inputs=[r_locs_input_layer, i_locs_input_layer, c_locs_input_layer, ang_input_layer, features_input_layer],
            outputs=output_layer, name="model_full_state")
        model.compile(optimizer='adam', loss='mse')
        # from keras.utils import plot_model
        # plot_model(model, to_file='model.png')
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
        default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
        # just for the case where there are no r_locs/i_locs
        r_locs = default_val  # rocket
        i_locs = default_val  # interceptor
        c_locs = default_val  # city
        ang = 0
        normalized_t = 0
        state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                 np.array([normalized_t])]
        return state