from Interceptor_V2 import Init, Game_step, Draw

class IronDomeEnv():
    def __init__(self):
        Init()
        self.observation_space.shape = 285
        self.action_space.n = 4


    def reset(self):
        Init()
        Game_step()

    def step(self, action):
        Game_step