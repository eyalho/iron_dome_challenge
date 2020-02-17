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
      new_r_locs, new_i_locs, c_locs, self.ang, score = Game_step(action)

      # padding
      #ast_r_locs_padded = np.concatenate((self.last_r_locs.flatten(), np.zeros((200 - 2 * np.shape(self.last_r_locs)[0]))))
      #last_i_locs_padded = np.concatenate((self.last_i_locs.flatten(), np.zeros(( 200 - 2 * np.shape(self.last_i_locs)[0]))))
      #new_r_locs_padded = np.concatenate((new_r_locs.flatten(), np.zeros((200 - 2 * np.shape(new_r_locs)[0]))))
      #new_i_locs_padded = np.concatenate((new_i_locs.flatten(), np.zeros((200 - 2 * np.shape(new_i_locs)[0]))))

      x_grid = np.array(list(range(-5000,5000, self.dx)))
      y_grid = np.array(list(range(0, 5000, self.dy)))
      eps = np.exp(-10) # avoid falling on grid points for patch calculation

      r_patches = np.zeros(len(x_grid)*len(y_grid))
      for i in range(len(new_r_locs)):
          x = new_r_locs[i, 0] + eps
          y = new_r_locs[i, 1] + eps
          x_loc = np.where(np.equal(x_grid>=x, x_grid+self.dx>=x)==False)[0] # between which grid point x is found
          y_loc = np.where(np.equal(y_grid > y, y_grid + self.dy >= y) == False)[0]# between which grid point y is found
          if x_loc and y_loc:
            r_patches[x_loc + y_loc*(len(x_grid+1))]+=1 # add 1 to relevant patch bin

    #do same for interceptors
      i_patches = np.zeros(len(x_grid) * len(y_grid))
      for i in range(len(new_i_locs)):
          x = new_i_locs[i, 0] + eps
          y = new_i_locs[i, 1] + eps
          x_loc = np.where(np.equal(x_grid > x, x_grid + self.dx >= x) == False)[0]
          y_loc = np.where(np.equal(y_grid > y, y_grid + self.dy >= y) == False)[0]
          if x_loc and y_loc:
            i_patches[x_loc + y_loc * (len(x_grid + 1))] += 1



      next_state = np.concatenate((r_patches, i_patches, c_locs.flatten(),np.array([self.ang])))



      #r_vels_padded = new_r_locs_padded - last_r_locs_padded
      #i_vels_padded = new_i_locs_padded - last_i_locs_padded

      #next_state = np.concatenate((new_r_locs_padded, new_i_locs_padded, c_locs.flatten(), self.ang), axis=None)
      #next_state = np.concatenate((r_vels_padded, new_r_locs_padded, i_vels_padded, new_i_locs_padded, c_locs.flatten(), self.ang), axis=None)

      #self.last_r_locs = new_r_locs
      #self.last_i_locs = new_i_locs
      self.reward = score - self.score
      self.score = score
      self.state = next_state
      self.t +=1
      done = False
      if self.t==self._max_episode_steps:
          done=True
      return next_state, self.reward, done, {}

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
