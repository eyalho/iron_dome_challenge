import random

from Interceptor_V2 import Init, Draw, Game_step

Init()
for stp in range(1000):
    action_button = random.randint(0, 3)
    r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    Draw()
