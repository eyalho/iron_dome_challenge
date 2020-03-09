import numpy as np

from envs import Interceptor_V2
from envs.Interceptor_V2 import Init, Game_step

g = Interceptor_V2.World.g  # Gravity [m/sec**2]
dt = Interceptor_V2.World.dt  # [sec]
FRICTION_COEFF = Interceptor_V2.World.fric  # Air friction [Units of Science]


def get_trajectory(vx, vy):
    x0 = Interceptor_V2.Turret.x_hostile
    y0 = Interceptor_V2.Turret.y_hostile
    # self.v0 = 700 + np.random.rand() * 300  # [m/sec]
    # self.vx = self.v0 * np.sin(np.deg2rad(self.ang))
    # self.vy = self.v0 * np.cos(np.deg2rad(self.ang))
    r_locs = [(x0, y0)]
    x, y = x0, y0
    while r_locs[-1][1] >= 0:
        v_loss = (vx ** 2 + vy ** 2) * FRICTION_COEFF * dt
        vx = vx * (1 - v_loss)
        vy = vy * (1 - v_loss) - g * dt
        x = x + vx * dt
        y = y + vy * dt
        r_locs.append((x, y))
    return r_locs


def get_rocket_second_velocity(x1, y1):
    """
    No math, and get vx_1, vy_1 based on x_1, y_1 and const x_0, y_0..
    :return: vx_1, vy_1
    """
    x0 = Interceptor_V2.Turret.x_hostile
    y0 = Interceptor_V2.Turret.y_hostile
    dx = x0 - x1
    dy = y1 - y0
    vx_1 = dx / dt
    vy_1 = dy / dt
    return vx_1, vy_1


def get_rocket_init_velocity(x1, y1):
    """
    Calc too much math, and get vx_0, vy_0 based on x_1, y_1 and const x_0, y_0..
    :return: vx_0, vy_0
    """
    x0 = Interceptor_V2.Turret.x_hostile
    y0 = Interceptor_V2.Turret.y_hostile
    dx = x0 - x1
    dy = y1 - y0
    a = -np.power(dx / dt, 2) - np.power(dy / dt + g * dt, 2)
    b = np.power(dx / dt, 2) / FRICTION_COEFF / dt
    c = np.power(dx / dt, 3) / FRICTION_COEFF / dt
    vx_0 = np.roots([a, 0, b, c])[-1]
    vy_0 = np.sqrt(1 / FRICTION_COEFF / dt + dx / FRICTION_COEFF / np.power(dt, 2) / vx - np.power(vx, 2))
    return vx_0, vy_0


def is_dest_city(x_dest, c_locs):
    return any([x_dest - loc[0] <= loc[1] for loc in c_locs])


if __name__ == "__main__":
    Init()
    action_button = 3
    r_locs = []

    # run until a rocket was shoot
    while len(r_locs) == 0:
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    print(f"r_locs {r_locs}")
    r1 = r_locs[0]
    x1 = r1[0]
    y1 = r1[1]
    v_x1, v_y1 = get_rocket_init_velocity(x1, y1)
    print(f"velocities v_x1, v_y1 =  {v_x1, v_y1}")
    trajectory = get_trajectory(v_x1, v_y1)
    print(f"len(trajectory)={len(trajectory)}")
    trajectory_final_x = trajectory[-1][0]
    print(f"trajectory final x ={trajectory_final_x}")
    print(f"c_locs = {c_locs}")
    city1 = c_locs[0]
    city2 = c_locs[1]
    print([trajectory_final_x - c_loc[0] <= c_loc[1] for c_loc in c_locs])
