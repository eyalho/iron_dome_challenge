import simulate_Interceptor_V2 as sim_env
from debug_logger import create_logger

import Interceptor_V2 as env
from Interceptor_V2 import Init, Draw, Game_step

logger = create_logger("train")
debug = logger.debug


def choose_action(steps_to_sim):
    rewards = calc_fancy_rewards(steps_to_sim)
    # rewards = np.asarray(rewards)
    shoot_r = rewards[0]
    wait_r = rewards[1]

    # if it worth shooting, shoot:
    if shoot_r > wait_r:
        debug("shoot!")
        action_button = 3
    else:
        debug("skip")
        action_button = 1

    return action_button


def calc_fancy_rewards(steps_to_sim):
    """
    :param steps_to_sim: how many step until end of game (1000-time_t)
    :param action_button: deserved action
    :return: the score of the game
    """
    shoot = 3
    wait = 1
    actions = [shoot, wait]
    rewards = []
    for action_button in actions:
        # init new simulate game
        sim_env.Simulate(env.world, env.turret, env.rocket_list, env.interceptor_list, env.city_list,
                         env.explosion_list)
        # act
        sim_env.Game_step(action_button)

        # peace steps until end of game
        for i in range(min(steps_to_sim - 2, 200)):
            sim_env.peace_step()

        # last step : save score
        r_locs, i_locs, c_locs, ang, score = sim_env.peace_step()

        # append score in end of peace game for choosing specific action
        rewards.append(score)

    debug(f"steps_to_sim = {steps_to_sim}\nrewards={rewards}")
    return rewards


if __name__ == "__main__":
    Init()
    max_stp = 1000
    init_stp = 4

    # move turent to best angle
    for stp in range(init_stp):
        action_button = 2
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)

    # shoot only if it's worth it
    for stp in range(stp, max_stp):
        action_button = choose_action(max_stp - stp)
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
        debug(f"{stp}.score = {score}")
        if stp % 30 == 0:
            Draw()
