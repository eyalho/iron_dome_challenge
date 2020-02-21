# TODO MAKE IT POSSIBLE USE OTHER ENVS
from simulator import simulate_Interceptor_V2 as sim_env


def predict_score(conf, action, steps_to_sim):
    MAX_STEPS = 150
    final_score = 0
    env = conf.env

    steps_to_sim = max(0, min(steps_to_sim - 1, MAX_STEPS))

    # init new simulate game
    sim_env.Simulate(env.world, env.turret, env.rocket_list, env.interceptor_list, env.city_list,
                     env.explosion_list)
    # act
    sim_env.simulate_game_step(action)

    # peace steps until end of game
    for i in range(steps_to_sim):
        _, _, _, _, final_score = sim_env.simulate_peace_step()
    return final_score


def predict_scores(conf, stp):
    """
    simulate 1 shoot and other peace steps : calc predicted_shoot_score
    simulate all peace: calc predicted_wait_score
    :param steps_to_sim: how many step until end of game (1000-stp)
    :return: predicted_shoot_score, predicted_wait_score
    """
    steps_to_sim = conf.NUMBER_OF_STEPS_IN_GAME - stp

    SHOOT = 3
    WAIT = 1
    predicted_shoot_score = predict_score(conf, SHOOT, steps_to_sim)
    predicted_wait_score = predict_score(conf, WAIT, steps_to_sim)
    return predicted_shoot_score, predicted_wait_score
