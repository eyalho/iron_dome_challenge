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
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten
from keras.optimizers import Adam
from keras.models import Model

from agents.abstract_agent import ABSDQNAgent

def create_agent():
    return CnnRAgent()

class CnnRAgent(ABSDQNAgent):
    def __init__(self, actions_size=4):
        # self.state_size = state_size
        self.action_size = actions_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.px = 84 # pixel amount along x axis of the input image
        self.py = 42 # pixel amount along y axis of the input image
        self.frame_history = 4 # amount og frame to be stacked at the network input
        self.model = self._build_model()
        self.simulate_reward = True
        self.name = "CNN_agent"
        # self.plot_model("agent3_cnn.png")



    def _build_model(self):

        """ return a Neural Net for Deep-Q learning Model"""
        # Neural Net for Deep-Q learning Model on image
        input_shape = (self.py, self.px, 3 * self.frame_history) # there are 3 channels for each image
        input_frames = Input(shape=input_shape, name="frames")
        input_ang = Input(shape=(1,), name="angle")
        first_Conv2D = Conv2D(input_shape=input_shape, filters=16, kernel_size=8, strides=(2, 4),
                              activation='relu')(input_frames)
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
        return model


    def init_state(self):
        """ return a valid state for stp=0"""
        self.state = [np.zeros([1, self.py, self.px, 3 * self.frame_history]), np.array([0])]
        return self.state


    def create_state(self, conf, r_locs, i_locs, c_locs, ang, score, stp):
        """ convert the default env.state to a valid input for the model agent"""

        # can be moved to __init__ for optimization
        # creating a grid to help accumulate rockets and interceptors in pixels
        x_grid = np.linspace(-5000, 5000, self.px) # maybe change magic 5000 to env.world.width
        x_grid_dx = x_grid[1] - x_grid[0]
        y_grid = np.linspace(0, 5000, self.py)
        y_grid_dy = y_grid[1] - y_grid[0]
        eps = np.exp(-10)  # avoid falling on grid points for patch calculation

        # counting rocket on each grid
        r_histogram_2D = np.zeros([1, self.py, self.px, 1])
        for r in range(len(r_locs)):
            x = r_locs[r, 0] + eps
            y = r_locs[r, 1] + eps
            x_loc = np.where(np.equal(x_grid >= x, x_grid + x_grid_dx >= x) == False)[0]  # between which grid point x is found
            y_loc = np.where(np.equal(y_grid >= y, y_grid + y_grid_dy >= y) == False)[0]  # between which grid point y is found
            if x_loc and y_loc:
                r_histogram_2D[0, y_loc, x_loc, 0] += 1  # add 1 to relevant patch bin

        # do same for interceptors
        i_histogram_2D = np.zeros([1, self.py, self.px, 1])
        for i in range(len(i_locs)):
            x = i_locs[i, 0] + eps
            y = i_locs[i, 1] + eps
            x_loc = np.where(np.equal(x_grid >= x, x_grid + x_grid_dx >= x) == False)[0]  # between which grid point x is found
            y_loc = np.where(np.equal(y_grid >= y, y_grid + y_grid_dy >= y) == False)[0]  # between which grid point y is found
            if x_loc and y_loc:
                i_histogram_2D[0, y_loc, x_loc, 0] += 1  # add 1 to relevant patch bin

        c_histogram_2D = np.zeros([1, self.py, self.px, 1])
        for i in range(len(c_locs)):
            x_left = c_locs[i, 0] - c_locs[i, 1] / 2 + eps
            x_right = c_locs[i, 0] + c_locs[i, 1] / 2 + eps
            y = 0
            left_city_edge = np.min(np.where(np.equal(x_grid >= x_left, x_grid + x_grid_dx >= x_left) == False)[0])  # between which grid point x is found
            right_city_edge = np.max(np.where(np.equal(x_grid > x_right, x_grid + x_grid_dx >= x_right) == False)[0])  # between which grid point y is found
            c_histogram_2D[0, y, range(left_city_edge, right_city_edge), 0] += 1  # add 1 to relevant patch bin

        r_i_c_histogram_2D = np.concatenate((r_histogram_2D, i_histogram_2D, c_histogram_2D), axis=3)
        self.state[0] = np.concatenate((r_i_c_histogram_2D, self.state[0]),
                                                           axis=3)[:, :, :, 0:(3 * self.frame_history)]
        self.state[1] = np.array([ang])

        return self.state

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

