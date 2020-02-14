import numpy as np
from Interceptor_V2 import Init, Draw, Game_step

G = 9.8
DELTA_T = 0.2
FRICTION_COEFF = 5e-7
INTERCEPTOR_VELOCITY = 800
INTERCEPTOR_X0 = -2000


def get_trajectory(v_x, v_y, x0=4800, y0=0):
    locs = [(x0, y0)]
    x, y = x0, y0
    while locs[-1][1] >= 0:
        v_squared = v_x ** 2 + v_y ** 2
        v_x = v_x * (1 - FRICTION_COEFF * DELTA_T * v_squared)
        v_y = v_y * (1 - FRICTION_COEFF * DELTA_T * v_squared) - G * DELTA_T
        x, y = x + v_x * DELTA_T, y + v_y * DELTA_T
        locs.append((x, y))
    return locs


def get_velocity(x1, y1, x0=4800, y0=0):
    a = -np.power((x0 - x1) / DELTA_T, 2) - np.power((y1 - y0) / DELTA_T + G * DELTA_T, 2)
    b = np.power((x0 - x1) / DELTA_T, 2) / FRICTION_COEFF / DELTA_T
    c = np.power((x0 - x1) / DELTA_T, 3) / FRICTION_COEFF / DELTA_T
    v_x = np.roots([a, 0, b, c])[-1]
    v_y = np.sqrt(
        1 / FRICTION_COEFF / DELTA_T + (x0 - x1) / FRICTION_COEFF / np.power(DELTA_T, 2) / v_x - np.power(v_x, 2))
    return v_x, v_y


def is_dest_city(x_dest, c_locs):
    return any([x_dest - loc[0] <= loc[1] for loc in c_locs])


Init()
action_button = 3
r_locs = []

while not len(r_locs):
    r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    if len(r_locs):
        v_x, v_y = get_velocity(r_locs[0][0], r_locs[0][1])
print(get_trajectory(v_x, v_y)[-1][0], c_locs[0])
print(any([get_trajectory(v_x, v_y)[-1][0] - loc[0] <= loc[1] for loc in c_locs]))