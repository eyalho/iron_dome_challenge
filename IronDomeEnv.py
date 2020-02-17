import gym
import numpy as np
from gym import spaces

from Interceptor_V2 import Init, Game_step, Draw


class IronDomeEnv(gym.Env):
"""
This class was ment to mimic the 'gym' standard environment interface
not all properties and methods were implemented since it already worked but updates are introduces when an 'off the shelf'
agent tries to use them.
Using this Class makes it easier to use open source code with minimal adjustments
"""



    def __init__(self, state_type):
        """
        : param state_type: string. possible options are: 'flattened_sky_patches', '2D_histogram', 'flattened_locs'

            flattened_sky_patches - each rocket/interceptor is associated with one of [x_patches]x[y_patches patches] of the sky. patches are then flattened
            and concatenated - [[x_patches]x[y_patches patches] rocket patches ,  [x_patches]x[y_patches patches] interceptors patches, cities, ang]

            2D_histogram - each rocket/interceptor is associated with one of [x_patches]x[y_patches patches] of the sky, the result is a 2-channel image containing
            the amount of rocket/interceptors at each sky patch. turret angle and cities positions are returned in independent arrays.

            flattened_locs - flattening and padding the locations of rockets, interceptors, cities and turret angle
        """
        super(IronDomeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self._max_episode_steps = 1000
        self.state_type = state_type
        # Example for using image as input:
        if state_type = ,
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(285, 1), dtype=np.uint32)
        self.state = self.reset()



    def step(self, action):
        """
        : param action: index of agent's action
        :output state: the environment state
        """

        if state_type=='flattened_sky_patches':
            r_locs, i_locs, c_locs, ang, score = Game_step(action)
            next_state = np.concatenate([r_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs)[0])), i_locs.flatten(),
                                         np.zeros((1, 140 - 2 * np.shape(i_locs)[0])), c_locs.flatten(), ang], axis=None)
            next_state = np.reshape(next_state, [285])
            self.reward = score - self.score
            self.score = score
            self.t += 1
            done = False
            if self.t == self._max_episode_steps:
                done = True
            return next_state, self.reward, done, {}
        elif state_type=='flattened_sky_patches'


    def reset(self):
        Init()
        self.t = 0
        self.reward = 0
        self.score = 0
        next_state, self.reward, done, _ = self.step(1)
        return next_state

    def render(self, mode='human', close=False):
        Draw()
