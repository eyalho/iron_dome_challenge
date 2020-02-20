import os
import time

import numpy as np
# import simulate_Interceptor_V2 as sim_env
from agents.naive_full_state_model_agent import DQNAgent
from savers.debug_logger import create_logger
from envs.env_for_training import Init, Draw, Game_step
from simulator.simulate_shoot import simulate_shoot_score

logger = create_logger("train")
debug = logger.debug


def eval_score(predicted_action, ang, score, steps_to_sim):
    """
    :param steps_to_sim: how many step until end of game (1000-stp)
    :param action_button: deserved action
    :return: the score of the game
    """
    SHOOT = 3
    ANGLE_SCORE_PUNISHMENT = 10000

    ############### SHOOT reward ###############
    # We want the agent to shoot when he's going to gain score.
    # But the score will be received in the far future.
    # So we simulate next steps (with no other shooting) and calc
    # how the score will be difference if he chose to shoot..
    shoot_score = simulate_shoot_score(steps_to_sim)
    if predicted_action == SHOOT:
        score += shoot_score
    else:
        score -= shoot_score

    ############### ANGLE reward ###############
    # While playing we noticed that optimal angel should be
    # between 12 to 72
    if ang < 12 or score > 72:
        score -= ANGLE_SCORE_PUNISHMENT

    return score


if __name__ == "__main__":
    NUMBER_OF_GAMES = 50000
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    BATCH_SIZE = int(NUMBER_OF_STEPS_IN_GAME / 2)
    render = False
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
            sim_score = eval_score(action, ang, score, stp_left)

            agent.memorize(state, action, sim_score, next_state, is_done)

            state = next_state

            # turn this on if you want to render
            if render:
                # once in _ episodes play on x_ fast forward
                if stp % 10 == 0 and e % 10 == 0:
                    Draw()

        # TODO figure out about right way to use replay
        agent.replay(BATCH_SIZE)

        debug(f'episode: {e + 1}/{NUMBER_OF_GAMES}, score: {score}, sim_score: {sim_score}')
        scores.append(score)

        if e % 50 == 0:
            directory = "models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, f"{agent.model.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(file_path)
            debug("Saved model to disk")
