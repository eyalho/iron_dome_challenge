import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from Interceptor_V2 import Init, Draw, Game_step
"""
Init()
max_r = 0
for stp in range(1000):
    action_button = 1# srandom.randint(0, 3)
    r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    r = len(r_locs)
    if r>max_r:
        max_r = r
print(max_r)
    #if stp%5000==0:
     #   Draw=
"""

# Deep Q-learning `Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma *\
                         np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)

            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


episodes = 500
batch_size = 32
Init()
agent = DQNAgent(state_size=285, action_size=4)

# Iterate the game
for e in range(episodes):

    # reset state in the beginning of each game
    Init()
    r_locs, i_locs, c_locs, ang, score = Game_step(1)

    state = np.concatenate((r_locs.flatten(), np.zeros((1, 140-2*np.shape(r_locs)[0])), i_locs.flatten(), np.zeros((1, 140-2*np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)
    state = np.reshape(state, [1, 285])
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time_t in range(1000):
        if e%30==0 and time_t%20==0:
            Draw()
        #    turn this on if you want to render


        # Decide action
        if np.shape(state)!=(1,285):
            a=1
        action = agent.act(state)

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        r_locs, i_locs, c_locs, ang, new_score = Game_step(action)
        next_state = np.concatenate((r_locs.flatten(), np.zeros((1, 140-2*np.shape(r_locs)[0])), i_locs.flatten(), np.zeros((1, 140-2*np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)
        reward = new_score - score
        score = new_score
        next_state = np.reshape(next_state, [1, 285])

        # memorize the previous state, action, reward, and done

        agent.memorize(state, action, reward, next_state)

        # make next_state the new current state for the next frame.
        state = next_state


        # print the score and break out of the loop
        print("episode: {}/{}, score: {}"
              .format(e, episodes, score))


    # train the agent with the experience of the episode
    if len(agent.memory) > batch_size:
       agent.replay(batch_size)

