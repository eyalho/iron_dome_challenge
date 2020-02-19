# -*- coding: utf-8 -*-
"""
Simulate the Interceptor_V2 game by deepcopy all the globals which were declared on Init
API:
Simulate(world, turret, rocket_list, interceptor_list, city_list, explosion_list)
will deepcopy the game state into the new s_* globals

"""
import copy
import matplotlib.pyplot as plt
import numpy as np

class SWorld():
    def __init__(self, world):
        self.width = copy.deepcopy(world.width)  # 10000 # [m]
        self.height = copy.deepcopy(world.height)  # 4000 # [m]
        self.dt = copy.deepcopy(world.dt)  # 0.2 # [sec]
        self.time = copy.deepcopy(world.time)  # 0 # [sec]
        self.score = copy.deepcopy(world.score)  # 0
        self.reward_city = copy.deepcopy(world.reward_city)  # -15
        self.reward_open = copy.deepcopy(world.reward_open)  # -1
        self.reward_fire = copy.deepcopy(world.reward_fire)  # -1
        self.reward_intercept = copy.deepcopy(world.reward_intercept)  # 4
        self.g = copy.deepcopy(world.g)  # 9.8 # Gravity [m/sec**2]
        self.fric = copy.deepcopy(world.fric)  # 5e-7 # Air friction [Units of Science]
        self.rocket_prob = copy.deepcopy(world.rocket_prob)  # 1 # expected rockets per sec

class STurret():
    def __init__(self, turrent):
        self.x = copy.deepcopy(turrent.x)
        self.y = copy.deepcopy(turrent.y)  # 0  # [m]
        self.x_hostile = copy.deepcopy(turrent.x_hostile)  # 4800
        self.y_hostile = copy.deepcopy(turrent.y_hostile)  # 0
        self.ang_vel = copy.deepcopy(turrent.ang_vel)  # 30  # Turret angular speed [deg/sec]
        self.ang = copy.deepcopy(turrent.ang)  # 0  # Turret angle [deg]
        self.v0 = copy.deepcopy(turrent.v0)  # 800  # Initial speed [m/sec]
        self.prox_radius = copy.deepcopy(turrent.prox_radius)  # 150  # detonation proximity radius [m]
        self.reload_time = copy.deepcopy(turrent.reload_time)  # 1.5  # [sec]
        self.last_shot_time = copy.deepcopy(turrent.last_shot_time)  # -3  # [sec]

    def update(self, action_button):
        if action_button == 0:
            self.ang = self.ang - self.ang_vel * s_world.dt
            if self.ang < -90: self.ang = -90

        if action_button == 1:
            pass

        if action_button == 2:
            self.ang = self.ang + self.ang_vel * s_world.dt
            if self.ang > 90: self.ang = 90

        if action_button == 3:
            if s_world.time - self.last_shot_time > self.reload_time:
                Interceptor()
                self.last_shot_time = s_world.time  # [sec]


class Interceptor():
    def __init__(self):
        self.x = s_turret.x
        self.y = s_turret.y
        self.vx = s_turret.v0 * np.sin(np.deg2rad(s_turret.ang))
        self.vy = s_turret.v0 * np.cos(np.deg2rad(s_turret.ang))
        s_world.score = s_world.score + s_world.reward_fire
        SInterceptor(self)

class SInterceptor():
    def __init__(self, interceptor):
        self.x = copy.deepcopy(interceptor.x)
        self.y = copy.deepcopy(interceptor.y)
        self.vx = copy.deepcopy(interceptor.vx)
        self.vy = copy.deepcopy(interceptor.vy)
        # s_world.score = s_world.score + s_world.reward_fire
        s_interceptor_list.append(self)

    def update(self):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * s_world.fric * s_world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - s_world.g * s_world.dt
        self.x = self.x + self.vx * s_world.dt
        self.y = self.y + self.vy * s_world.dt
        if self.y < 0:
            Explosion(self.x, self.y)
            s_interceptor_list.remove(self)
        if np.abs(self.x) > s_world.width / 2:
            s_interceptor_list.remove(self)


class SRocket():
    def __init__(self, rocket):
        self.x = copy.deepcopy(rocket.x)  # [m]
        self.y = copy.deepcopy(rocket.y)  # [m]
        self.v0 = copy.deepcopy(rocket.v0)  # [m/sec]
        self.ang = copy.deepcopy(rocket.ang)  # [deg]
        self.vx = copy.deepcopy(rocket.vx)
        self.vy = copy.deepcopy(rocket.vy)
        s_rocket_list.append(self)

    def update(self):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * s_world.fric * s_world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - s_world.g * s_world.dt
        self.x = self.x + self.vx * s_world.dt
        self.y = self.y + self.vy * s_world.dt


class SCity():
    def __init__(self, city):
        self.x = copy.deepcopy(city.x)
        self.width = copy.deepcopy(city.width)
        s_city_list.append(self)
        self.img = np.zeros((200, 800))
        # for b in range(60):
        #     h = np.random.randint(30, 180)
        #     w = np.random.randint(30, 80)
        #     x = np.random.randint(1, 700)
        #     self.img[0:h, x:x + w] = np.random.rand()
        # self.img = np.flipud(self.img)


class Explosion():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 500
        self.duration = 0.4  # [sec]
        self.verts1 = (np.random.rand(30 ,2)- 0.5) * self.size
        self.verts2 = (np.random.rand(20 ,2)- 0.5) * self.size / 2
        self.verts1[:, 0] = self.verts1[:, 0] + x
        self.verts1[:, 1] = self.verts1[:, 1] + y
        self.verts2[:, 0] = self.verts2[:, 0] + x
        self.verts2[:, 1] = self.verts2[:, 1] + y
        self.hit_time = s_world.time
        s_explosion_list.append(self)

    def update(self):
        if s_world.time - self.hit_time > self.duration:
            s_explosion_list.remove(self)


class SExplosion():
    def __init__(self, explosion):
        self.x = copy.deepcopy(explosion.x)
        self.y = copy.deepcopy(explosion.y)
        self.size = copy.deepcopy(explosion.size)
        self.duration = copy.deepcopy(explosion.duration)  # [sec]
        self.verts1 = copy.deepcopy(explosion.verts1)
        self.verts2 = copy.deepcopy(explosion.verts2)
        self.hit_time = copy.deepcopy(explosion.hit_time)
        s_explosion_list.append(self)

    def update(self):
        if s_world.time - self.hit_time > self.duration:
            s_explosion_list.remove(self)


def Check_interception():
    for intr in s_interceptor_list:
        for r in s_rocket_list:
            if ((r.x - intr.x) ** 2 + (r.y - intr.y) ** 2) ** 0.5 < s_turret.prox_radius:
                s_rocket_list.remove(r)
                Explosion(intr.x, intr.y)
                if intr in s_interceptor_list: s_interceptor_list.remove(intr)
                s_world.score = s_world.score + s_world.reward_intercept


def Check_ground_hit():
    for r in s_rocket_list:
        if r.y < 0:
            city_hit = False
            for c in s_city_list:
                if np.abs(r.x - c.x) < c.width:
                    city_hit = True
            if city_hit == True:
                s_world.score = s_world.score + s_world.reward_city
            else:
                s_world.score = s_world.score + s_world.reward_open
            Explosion(r.x, r.y)
            s_rocket_list.remove(r)


def Draw():
    plt.cla()
    plt.rcParams['axes.facecolor'] = 'black'
    for r in s_rocket_list:
        plt.plot(r.x, r.y, '.y')
    for intr in s_interceptor_list:
        plt.plot(intr.x, intr.y, 'or')
        C1 = plt.Circle((intr.x, intr.y), radius=s_turret.prox_radius, linestyle='--', color='gray', fill=False)
        ax = plt.gca()
        ax.add_artist(C1)
    for c in s_city_list:
        plt.imshow(c.img, extent=[c.x - c.width / 2, c.x + c.width / 2, 0, c.img.shape[0]])
        plt.set_cmap('bone')
    for e in s_explosion_list:
        P1 = plt.Polygon(e.verts1, True, color='yellow')
        P2 = plt.Polygon(e.verts2, True, color='red')
        ax = plt.gca()
        ax.add_artist(P1)
        ax.add_artist(P2)
    plt.plot(s_turret.x, s_turret.y, 'oc', markersize=12)
    plt.plot([s_turret.x, s_turret.x + 100 * np.sin(np.deg2rad(s_turret.ang))],
             [s_turret.y, s_turret.y + 100 * np.cos(np.deg2rad(s_turret.ang))], 'c', linewidth=3)
    plt.plot(s_turret.x_hostile, s_turret.y_hostile, 'or', markersize=12)
    plt.axes().set_aspect('equal')
    plt.axis([-s_world.width / 2, s_world.width / 2, 0, s_world.height])
    plt.title('Score: ' + str(s_world.score))
    plt.draw()
    plt.pause(0.001)


def Simulate(world, turret, rocket_list, interceptor_list, city_list, explosion_list):
    global s_world, s_turret, s_rocket_list, s_interceptor_list, s_city_list, s_explosion_list
    s_rocket_list = []
    s_interceptor_list = []
    s_city_list = []
    s_explosion_list = []

    s_world = SWorld(world)

    # For debug purpose
    s_world.score = 0  # float(s_world.score)
    # s_world.reward_city = 0  # float(s_world.reward_city) + 0.01
    # s_world.reward_open = 0  # float(s_world.reward_open) + 0.001
    # s_world.reward_fire = -10  # float(s_world.reward_fire) + 0.0001
    # s_world.reward_intercept = 100  # float(s_world.reward_intercept) + 0.00001

    s_turret = STurret(turret)
    s_rocket_list = [SRocket(rocket) for rocket in rocket_list]
    s_interceptor_list = [SInterceptor(interceptor) for interceptor in interceptor_list]
    s_city_list = [SCity(city) for city in city_list]
    s_explosion_list = [SExplosion(explosion) for explosion in explosion_list]


def simulate_game_step(action_button):
    """
    simulate a step where:
     - no new rockets
    """
    s_world.time = s_world.time + s_world.dt

    # we dont want to check about rocekts which were not exsits in the real env
    # if np.random.rand() < s_world.rocket_prob * s_world.dt:
    #     Rocket(s_world)

    for r in s_rocket_list:
        r.update()

    for intr in s_interceptor_list:
        intr.update()

    for e in s_explosion_list:
        e.update()

    s_turret.update(action_button)
    Check_interception()
    Check_ground_hit()

    r_locs = np.zeros(shape=(len(s_rocket_list), 2))
    for ind in range(len(s_rocket_list)):
        r_locs[ind, :] = [s_rocket_list[ind].x, s_rocket_list[ind].y]

    i_locs = np.zeros(shape=(len(s_interceptor_list), 2))
    for ind in range(len(s_interceptor_list)):
        i_locs[ind, :] = [s_interceptor_list[ind].x, s_interceptor_list[ind].y]

    c_locs = np.zeros(shape=(len(s_city_list), 2))
    for ind in range(len(s_city_list)):
        c_locs[ind, :] = [s_city_list[ind].x, s_city_list[ind].width]

    return r_locs, i_locs, c_locs, s_turret.ang, s_world.score


def simulate_peace_step():
    """
    simulate a step where:
     - no new rockets
     - no new interceptors
    :return:
    """
    PEACE=1
    return simulate_game_step(PEACE)