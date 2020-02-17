import numpy as np
from Interceptor_V2 import Init, Draw, Game_step
from agent import DQNAgent


model_path =  './models/model_full_state_e2800_2020_02_17-18_03_35.hdf5'
"""
class test_agent()
    def __init__(self, path, eval_iter_num):
        self.path = path
        self.eval_iter_num = eval_iter_num
        self.scores = []
        self.max_score = []
        self.min_score = []
        self = []"""

def create_agent(path):
    agent = DQNAgent()
    agent.model.load_weights('./models/model_full_state_e2800_2020_02_17-18_03_35.hdf5')

    return agent


Init()
agent = create_agent(model_path)


default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
# just for the case where there are no r_locs/i_locs

r_locs = default_val  # rocket
i_locs = default_val  # interceptor
c_locs = default_val  # city
ang = 0
normalized_t = 0
state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
         np.array([normalized_t])]
score = 0
scores = []

r_locs, i_locs, c_locs, ang, score = Game_step(1)
NUMBER_OF_STEPS_IN_GAME = 1000

for stp in range(NUMBER_OF_STEPS_IN_GAME):
    normalized_t = stp / NUMBER_OF_STEPS_IN_GAME
    r_locs = np.concatenate([default_val, r_locs])
    i_locs = np.concatenate([default_val, i_locs])
    state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                  np.array([normalized_t])]
    action_button = agent.model.predict(state)
    r_locs, i_locs, c_locs, ang, score = Game_step(np.argmax(action_button))
    scores.append(score)
    Draw()

print('test endded with score of: ' + scores[len(scores)-1])


