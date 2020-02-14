# Deep Q-learning Agent
import random
from collections import deque

from Interceptor_V2 import Init, Draw, Game_step
from keras.models import Model
from keras.layers import Dense, Input, LSTM, concatenate
import numpy as np


class DQNAgent:
    def __init__(self, state_size=None, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        hidden_size = 20
        r_locs_input_layer = Input(shape=(None, 2))
        i_locs_input_layer = Input(shape=(None, 2))
        c_locs_input_layer = Input(shape=(None, 2))
        ang_input_layer = Input(shape=(1,))
        r_locs_lstm_layer = LSTM(hidden_size)(r_locs_input_layer)
        i_locs_lstm_layer = LSTM(hidden_size)(i_locs_input_layer)
        c_locs_lstm_layer = LSTM(hidden_size)(c_locs_input_layer)
        layer = concatenate([r_locs_lstm_layer, i_locs_lstm_layer, c_locs_lstm_layer, ang_input_layer])
        layer = Dense(hidden_size, activation='linear')(layer)
        output_layer = Dense(self.action_size, activation='tanh')(layer)
        model = Model(inputs=[r_locs_input_layer, i_locs_input_layer, c_locs_input_layer, ang_input_layer],
                      outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
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


if __name__ == "__main__":
    episodes = 500
    agent = DQNAgent()
    run_time = 1000
    default_val = np.array([[-1, -1]])
    for e in range(episodes):
        Init()
        r_locs = default_val
        i_locs = default_val
        c_locs = default_val
        ang = 0
        state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]
        for time_t in range(run_time):
            action = agent.act(state)
            r_locs, i_locs, c_locs, ang, score = Game_step(action)
            if e%50 == 49:
                Draw()
            r_locs = np.concatenate([default_val, r_locs])
            i_locs = np.concatenate([default_val, i_locs])
            next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]
            agent.remember(state, action, score, next_state, False)
            state = next_state
        agent.replay(min(time_t, 32))
        print(f'episode: {e+1}/{episodes}, score: {score}')