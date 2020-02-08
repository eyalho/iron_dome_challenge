from Interceptor_V2 import Init, Draw, Game_step
from DQNAgent import DQNAgent
import numpy as np

agent = DQNAgent(state_size=285, action_size=4)
agent.model.load_weights('./models/DQN 24x24-200 episodes - model.hdf5')
Init()
r_locs, i_locs, c_locs, ang, new_score = Game_step(1)
state = np.concatenate((r_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs)[0])), i_locs.flatten(),
                        np.zeros((1, 140 - 2 * np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)

for i in range(1000):
    state = np.reshape(state, [1, 285])
    Draw()
    action = np.argmax(agent.model.predict(state))
    print(action)
    r_locs, i_locs, c_locs, ang, new_score = Game_step(action)
    state = np.concatenate((r_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs)[0])), i_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)

