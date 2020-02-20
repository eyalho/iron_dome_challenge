from Interceptor_V2 import Init, Draw, Game_step
import numpy as np

for e in range(100):
    Init()
    for stp in range(1000):
        action_button = np.random.randint(0,3,(1,))
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    print(score)