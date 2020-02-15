import numpy as np

from DQNAgent import DQNAgent
from Interceptor_V2 import Init, Draw, Game_step

agent = DQNAgent(state_size=565, action_size=4)
agent.model.load_weights('./models/DQN 24x24-4950 episodes - model.hdf5')
Init()
r_locs_1, i_locs_1, c_locs, ang, score = Game_step(1)
r_locs_2, i_locs_2, c_locs, ang, score = Game_step(1)
state = np.concatenate((r_locs_1.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs_1)[0])), \
                        r_locs_2.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs_2)[0])), \
                        i_locs_1.flatten(), np.zeros((1, 140 - 2 * np.shape(i_locs_1)[0])), \
                        i_locs_2.flatten(), np.zeros((1, 140 - 2 * np.shape(i_locs_2)[0])), \
                        c_locs.flatten(), ang), axis=None)
state = np.reshape(state, [1, 565])

for i in range(998):
    Draw()
    action = np.argmax(agent.model.predict(state))
    print(action)
    r_locs, i_locs, c_locs, ang, new_score = Game_step(action)
    r_locs_1, i_locs_1 = r_locs_2, i_locs_2
    r_locs_2, i_locs_2, c_locs, ang, new_score = Game_step(action)
    state = np.concatenate((r_locs_1.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs_1)[0])), \
                            r_locs_2.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs_2)[0])), \
                            i_locs_1.flatten(), np.zeros((1, 140 - 2 * np.shape(i_locs_1)[0])), \
                            i_locs_2.flatten(), np.zeros((1, 140 - 2 * np.shape(i_locs_2)[0])), \
                            c_locs.flatten(), ang), axis=None)
    state = np.reshape(state, [1, 565])
