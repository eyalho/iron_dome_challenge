import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
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
        : param state_type: string. possible options are: 'flattened_sky_patches', 'histogram_2D ', 'flattened_locs', 'flattened_locs_and_vels

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
        self.dx = 1000
        self.dy = 1000
        self.px = 84
        self.py = 42
        self.frame_history = 5
        self.r_i_histogram_2D_last_frames = np.zeros([1, self.py, self.px, 3*self.frame_history])
        # Example for using image as input:
        if state_type == 'flattened_locs': # flattened_sky_patches', '2D_histogram', 'flattened_locs', 'flattened_locs_and_vels
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(285, 1), dtype=np.uint32)
        elif state_type == 'flattened_locs_and_vels':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(565, 1), dtype=np.uint32)
        elif state_type == 'flattened_sky_patches':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10000*5000/(self.dx*self.dy)+5, 1), dtype=np.uint32)
        elif state_type == 'histogram_2D ':
            self.observation_space = spaces.Tuple((spaces.Box(low=0, high=70, shape=(self.px, self.py, 8), dtype=np.uint32),
                                                   spaces.Box(low=-90,high=90, shape=(1,), dtype=np.uint32)))
            # channel for rockets and channel for interceptors, px x py image
        self.state = self.reset()



    def step(self, action):
        """
        : param action: index of agent's action
        :output state: the environment state
        """

        if self.state_type=='flattened_locs':
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

        elif self.state_type == 'flattened_locs_and_vels':
            new_r_locs, new_i_locs, c_locs, self.ang, score = Game_step(action)

            # padding
            last_r_locs_padded = np.concatenate((self.last_r_locs.flatten(), np.zeros((200 - 2 * np.shape(self.last_r_locs)[0]))))
            last_i_locs_padded = np.concatenate((self.last_i_locs.flatten(), np.zeros(( 200 - 2 * np.shape(self.last_i_locs)[0]))))
            new_r_locs_padded = np.concatenate((new_r_locs.flatten(), np.zeros((200 - 2 * np.shape(new_r_locs)[0]))))
            new_i_locs_padded = np.concatenate((new_i_locs.flatten(), np.zeros((200 - 2 * np.shape(new_i_locs)[0]))))

            r_vels_padded = new_r_locs_padded - last_r_locs_padded
            i_vels_padded = new_i_locs_padded - last_i_locs_padded

            next_state = np.concatenate((r_vels_padded, new_r_locs_padded, i_vels_padded, new_i_locs_padded, c_locs.flatten(), self.ang), axis=None)

            self.last_r_locs = new_r_locs
            self.last_i_locs = new_i_locs
            self.reward = score - self.score
            self.score = score
            self.state = next_state
            self.t += 1
            done = False
            if self.t == self._max_episode_steps:
                done = True
            return next_state, self.reward, done, {}

        elif self.state_type=='flattened_sky_patches':


            new_r_locs, new_i_locs, c_locs, self.ang, score = Game_step(action)

            # padding
            x_grid = np.array(list(range(-5000, 5000, self.dx)))
            y_grid = np.array(list(range(0, 5000, self.dy)))
            eps = np.exp(-10)  # avoid falling on grid points for patch calculation

            r_patches = np.zeros(len(x_grid) * len(y_grid))
            for i in range(len(new_r_locs)):
                x = new_r_locs[i, 0] + eps
                y = new_r_locs[i, 1] + eps
                x_loc = np.where(np.equal(x_grid >= x, x_grid + self.dx >= x) == False)[
                    0]  # between which grid point x is found
                y_loc = np.where(np.equal(y_grid > y, y_grid + self.dy >= y) == False)[
                    0]  # between which grid point y is found
                if x_loc and y_loc:
                    r_patches[x_loc + y_loc * (len(x_grid + 1))] += 1  # add 1 to relevant patch bin

            # do same for interceptors
            i_patches = np.zeros(len(x_grid) * len(y_grid))
            for i in range(len(new_i_locs)):
                x = new_i_locs[i, 0] + eps
                y = new_i_locs[i, 1] + eps
                x_loc = np.where(np.equal(x_grid > x, x_grid + self.dx >= x) == False)[0]
                y_loc = np.where(np.equal(y_grid > y, y_grid + self.dy >= y) == False)[0]
                if x_loc and y_loc:
                    i_patches[x_loc + y_loc * (len(x_grid + 1))] += 1

            next_state = np.concatenate((r_patches, i_patches, c_locs.flatten(), np.array([self.ang])))
            self.reward = score - self.score
            self.score = score
            self.state = next_state
            self.t += 1
            done = False
            if self.t == self._max_episode_steps:
                done = True
            return next_state, self.reward, done, {}

        elif self.state_type=='histogram_2D':

            new_r_locs, new_i_locs, c_locs, self.ang, score = Game_step(action)

            # creating grids
            x_grid = np.linspace(-5000,5000, self.px)
            x_grid_dx = x_grid[1] - x_grid[0]
            y_grid = np.linspace(0,5000, self.py)
            y_grid_dy = y_grid[1] - y_grid[0]
            eps = np.exp(-10)  # avoid falling on grid points for patch calculation

            #counting rocket on each grid
            r_histogram_2D = np.zeros([1, self.py, self.px, 1])
            for r in range(len(new_r_locs)):
                x = new_r_locs[r, 0] + eps
                y = new_r_locs[r, 1] + eps
                x_loc = np.where(np.equal(x_grid >= x, x_grid + x_grid_dx >= x) == False)[
                    0]  # between which grid point x is found
                y_loc = np.where(np.equal(y_grid > y, y_grid + y_grid_dy  >= y) == False)[
                    0]  # between which grid point y is found
                if x_loc and y_loc:
                    r_histogram_2D[0,y_loc, x_loc,0] += 1  # add 1 to relevant patch bin

            # do same for interceptors
            i_histogram_2D = np.zeros([1, self.py, self.px, 1])
            for i in range(len(new_i_locs)):
                x = new_i_locs[i, 0] + eps
                y = new_i_locs[i, 1] + eps
                x_loc = np.where(np.equal(x_grid >= x, x_grid + x_grid_dx>= x) == False)[0]  # between which grid point x is found
                y_loc = np.where(np.equal(y_grid > y, y_grid + y_grid_dy >= y) == False)[0]  # between which grid point y is found
                if x_loc and y_loc:
                   i_histogram_2D[0,y_loc, x_loc,0] += 1  # add 1 to relevant patch bin

            c_histogram_2D = np.zeros([1, self.py, self.px, 1])
            for i in range(len(c_locs)):
                x_left = c_locs[i, 0] - c_locs[i, 1]/2 + eps
                x_right = c_locs[i, 0] + c_locs[i, 1] / 2 + eps
                y = 0
                x_l_loc = np.min(np.where(np.equal(x_grid >= x_left, x_grid + x_grid_dx>= x_left ) == False)[0])  # between which grid point x is found
                x_r_loc = np.max(np.where(np.equal(x_grid > x_right, x_grid + x_grid_dx >= x_right) == False)[0]) # between which grid point y is found
                c_histogram_2D[0,y, range(x_l_loc, x_r_loc), 0] += 1  # add 1 to relevant patch bin

            r_i_c_histogram_2D = np.concatenate((r_histogram_2D, i_histogram_2D,c_histogram_2D), axis=3)
            self.r_i_histogram_2D_last_frames = np.concatenate((r_i_c_histogram_2D, self.r_i_histogram_2D_last_frames),
                                                               axis=3)[:,:,:,0:(3*self.frame_history)]
            next_state = (self.r_i_histogram_2D_last_frames, np.array([self.ang]))
            self.reward = score - self.score
            self.score = score
            self.state = next_state
            self.t += 1
            done = False
            if self.t == self._max_episode_steps:
                done = True
            return next_state, self.reward, done, {}


    def reset(self):
        Init()
        self.t = 0
        self.reward = 0
        self.score = 0
        next_state, self.reward, done, _ = self.step(1)
        return next_state

    def render(self, mode='human', close=False):
        #Draw()
        if self.state_type=='histogram_2D':
            plt.cla
            image = self.state[0][0,:,:,0:3]
            image = np.flipud(image)
            plt.imshow(image)
            plt.draw()
            plt.pause(0.001)

