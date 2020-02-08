from Interceptor_V2 import Init, Game_step, Draw

class IronDomeEnv():
    def __init__(self):
        Init()
        self.observation_space.shape = 285
        self.action_space.n = 4

    def sample(self):
        return self.np_random.randint(self.n)

    def reset(self):
        Init()
        r_locs, i_locs, c_locs, ang, score = Game_step(1)
        state = np.concatenate((r_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs)[0])), i_locs.flatten(),
                                np.zeros((1, 140 - 2 * np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)
        state = np.reshape(state, [1, 285])
        return state
    def step(self, action):
        r_locs, i_locs, c_locs, ang, score = Game_step(action)
        state = np.concatenate((r_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs)[0])), i_locs.flatten(),
                                np.zeros((1, 140 - 2 * np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)
        state = np.reshape(state, [1, 285])
        return state