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
    self._max_episode_steps = 1000
    self.ang = 0
    self.dx = 1000
    self.dy = 1000
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=np.inf, shape=(105,1), dtype=np.uint32)
    self.state = self.reset()


  def step(self, action):

      new_r_locs, new_i_locs, c_locs, self.ang, score = Game_step(action)

      # padding
      #ast_r_locs_padded = np.concatenate((self.last_r_locs.flatten(), np.zeros((200 - 2 * np.shape(self.last_r_locs)[0]))))
      #last_i_locs_padded = np.concatenate((self.last_i_locs.flatten(), np.zeros(( 200 - 2 * np.shape(self.last_i_locs)[0]))))
      #new_r_locs_padded = np.concatenate((new_r_locs.flatten(), np.zeros((200 - 2 * np.shape(new_r_locs)[0]))))
      #new_i_locs_padded = np.concatenate((new_i_locs.flatten(), np.zeros((200 - 2 * np.shape(new_i_locs)[0]))))

      x_grid = np.array(list(range(-5000,5000, self.dx)))
      y_grid = np.array(list(range(0, 5000, self.dy)))
      eps = np.exp(-10)

      r_patches = np.zeros(len(x_grid)*len(y_grid))
      for i in range(len(new_r_locs)):
          x = new_r_locs[i, 0]
          y = new_r_locs[i, 1]
          x_loc = np.where(np.equal(x_grid>=x, x_grid+self.dx>=x)==False)[0][0]
          y_loc = np.where(np.equal(y_grid > y, y_grid + self.dy >= y) == False)[0][0]
          r_patches[x_loc + y_loc*(len(x_grid+1))]+=1

      i_patches = np.zeros(len(x_grid) * len(y_grid))
      for i in range(len(new_i_locs)):
          x = new_i_locs[i, 0]
          y = new_i_locs[i, 1]
          x_loc = np.where(np.equal(x_grid > x, x_grid + self.dx >= x) == False)[0]
          y_loc = np.where(np.equal(y_grid > y, y_grid + self.dy >= y) == False)[0]
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


  def reset(self):
      Init()
      self.last_r_locs, self.last_i_locs, c_locs, ang, score = Game_step(1)
      self.t = 0
      self.reward = 0
      self.score = 0
      next_state, self.reward, done, _ =self.step(1)
      return next_state

  def render(self, mode='human', close=False):
      Draw()



