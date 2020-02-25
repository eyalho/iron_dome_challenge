import time

import keyboard

from Interceptor_V2 import Init, Draw, Game_step

Init()
for stp in range(1000):
    t0 = time.time()
    while True:

        if keyboard.is_pressed('a'):
            action = 0
            break
        elif keyboard.is_pressed('w'):
            action = 3
            break
        elif keyboard.is_pressed('d'):
            action = 2
            break
        else:
            action = 1
        t1 = time.time()
        if t1 - t0 > 0.2:
            break

    r_locs, i_locs, c_locs, ang, score = Game_step(action)
    Draw()
