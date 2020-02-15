import env_for_training as env
import numpy as np
import simulate_Interceptor_V2 as sim_env
from agent import DQNAgent
from debug_logger import create_logger
from env_for_training import Init, Draw, Game_step

logger = create_logger("train")
debug = logger.debug


def eval_score(predicted_action, steps_to_sim):
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
        # if action_button == SHOOT: debug("try shoot")
        # if action_button == WAIT: debug("try wait")
        sim_env.Game_step(action_button)

        # peace steps until end of game
        for i in range(steps_to_sim):
            _, _, _, _, score = sim_env.peace_step()
        # last step : save score in end of peace game
        scores.append(score)

    shoot_score = scores[0]
    wait_score = scores[1]
    if shoot_score != wait_score:
        debug(
            f"steps_to_simulate = {steps_to_sim}\nshoot={shoot_score}, wait={wait_score} diff={shoot_score - wait_score}")
    if predicted_action == SHOOT:
        return shoot_score - wait_score
    else:
        return wait_score - shoot_score


if __name__ == "__main__":
    NUMBER_OF_GAMES = 2
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    BATCH_SIZE = int(NUMBER_OF_STEPS_IN_GAME / 2)
    render = True
    agent = DQNAgent()
    scores = []

    debug(f"start train of {NUMBER_OF_GAMES} episodes, with batch size {BATCH_SIZE}\n")
    default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
    # just for the case where there are no r_locs/i_locs

    for e in range(NUMBER_OF_GAMES):
        Init()
        r_locs = default_val  # rocket
        i_locs = default_val  # interceptor
        c_locs = default_val  # city
        ang = 0
        normalized_t = 0
        state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                 np.array([normalized_t])]
        for stp in range(NUMBER_OF_STEPS_IN_GAME):
            stp_left = NUMBER_OF_STEPS_IN_GAME - stp
            normalized_t = stp / NUMBER_OF_STEPS_IN_GAME

            action = agent.act(state)
            r_locs, i_locs, c_locs, ang, score = Game_step(action)
            r_locs = np.concatenate([default_val, r_locs])
            i_locs = np.concatenate([default_val, i_locs])
            next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                          np.array([normalized_t])]
            is_done = stp == NUMBER_OF_STEPS_IN_GAME

            score = eval_score(action, stp_left)
            agent.memorize(state, action, score, next_state, is_done)

            state = next_state

            # turn this on if you want to render
            if render:
                # once in _ episodes play on x_ fast forward
                if stp % 100 == 0:
                    Draw()

        # TODO figure out about right way to use replay
        agent.replay(BATCH_SIZE)
        debug(f'episode: {e + 1}/{NUMBER_OF_GAMES}, score: {score}')
        scores.append(score)
