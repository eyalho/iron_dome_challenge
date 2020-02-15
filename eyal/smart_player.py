import simulate_Interceptor_V2 as sim_env
from debug_logger import create_logger

import Interceptor_V2 as env
from Interceptor_V2 import Init, Draw, Game_step

# from env_for_training import Init, Draw, Game_step
# import env_for_training as env

logger = create_logger("smart_player")
debug = logger.debug


def choose_action(steps_to_sim):
    SHOOT = 3
    WAIT = 1
    diff_score = predict_shoot_score(steps_to_sim)
    # if it worth shooting, shoot:
    if diff_score > 0:
        debug("shoot!")
        action_button = SHOOT
    else:
        debug("skip")
        action_button = WAIT
    return action_button


def predict_shoot_score(steps_to_sim):
    """
    :param steps_to_sim: how many step until end of game (1000-stp)
    :param action_button: deserved action
    :return: the score of the game
    """
    SHOOT = 3
    WAIT = 1
    MAX_STEPS = 300
    actions = [SHOOT, WAIT]
    scores = []
    steps_to_sim = min(steps_to_sim, MAX_STEPS)
    for action_button in actions:
        # init new simulate game
        sim_env.Simulate(env.world, env.turret, env.rocket_list, env.interceptor_list, env.city_list,
                         env.explosion_list)
        # act
        sim_env.Game_step(action_button)

        # peace steps until end of game
        for i in range(steps_to_sim):
            _, _, _, _, score = sim_env.peace_step()
        # last step : save score in end of peace game
        scores.append(score)

    shoot_score = scores[0] - scores[1]
    if shoot_score != 0:
        debug(f"steps_to_simulate = {steps_to_sim}\n diff={shoot_score}")
    return shoot_score


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

        if action_button == 3 or stp % 1 == 0:
            Draw()
