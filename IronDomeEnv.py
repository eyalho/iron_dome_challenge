from Interceptor_V2 import Init, Game_step, Draw
import numpy as np
import gym
from gym import spaces

class IronDomeEnv(gym.Env):


  def __init__(self,):
    super(IronDomeEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(4)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(285,1), dtype=np.uint32)
    self.reset()

  def step(self, action):

      r_locs, i_locs, c_locs, ang, score = Game_step(action)
      next_state = np.concatenate([r_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs)[0])), i_locs.flatten(),
                                   np.zeros((1, 140 - 2 * np.shape(i_locs)[0])), c_locs.flatten(), ang], axis=None)
      next_state = np.reshape(next_state, [1, 285])
      self.reward = score - self.score
      self.score = score
      self.t +=1
      done = False
      if self.t==1000:
          done=True
      return next_state, self.reward, done, {}


  def reset(self):
      Init()
      self.t = 0
      self.reward = 0
      self.score = 0
      next_state, self.reward, done, _ =self.step(1)
      return next_state

  def render(self, mode='human', close=False):
      Draw()



