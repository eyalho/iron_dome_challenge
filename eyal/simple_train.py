import os
import time
import uuid

import numpy as np
from debug_logger import create_logger
from env_for_training import Init, Draw, Game_step, Save_draw
# import simulate_Interceptor_V2 as sim_env
from simple_agent import DQNAgent
from smart_player import simulate_shoot_score

logger = create_logger("simple train")
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
    unique = str(uuid.uuid4())[:5]
    SAVE_RATE = 50
    STP_RATE  = 10
    NUMBER_OF_GAMES = 50000
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    BATCH_SIZE = int(NUMBER_OF_STEPS_IN_GAME / 2)
    render = True
    agent = DQNAgent()
    scores = []

    debug(f"simple start train of {NUMBER_OF_GAMES} episodes, with batch size {BATCH_SIZE}\n")
    default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
    # just for the case where there are no r_locs/i_locs

    for e in range(NUMBER_OF_GAMES):
        MAX_SIM_SCORE = 100000
        Init()
        ang = 0.0
        normalized_t = 0.0
        sim_score = 0.0
        MAX_ANG = 360
        state = [np.array([ang]), np.array([0]), np.array([normalized_t])]
        for stp in range(NUMBER_OF_STEPS_IN_GAME):
            stp_left = NUMBER_OF_STEPS_IN_GAME - stp
            action = agent.act(state)
            r_locs, i_locs, c_locs, ang, score = Game_step(action)

            sim_score = eval_score(action, ang, score, stp_left)

            normalized_ang = ang / MAX_ANG
            normalized_sim_score = sim_score / MAX_SIM_SCORE
            normalized_t = stp / NUMBER_OF_STEPS_IN_GAME
            next_state = [np.array([ang]), np.array([normalized_sim_score]), np.array([normalized_t])]
            is_done = stp == NUMBER_OF_STEPS_IN_GAME
            agent.memorize(state, action, score, next_state, is_done)
            state = next_state

            # turn this on if you want to render
            if render:
                # once in _ episodes play on x_ fast forward
                if stp % STP_RATE == 0 and e % SAVE_RATE == 0:
                    directory1 = "plots"
                    directory2 = f"{agent.model.name}_{unique}"
                    directory3 = f"e{e}"
                    directory = os.path.join(directory1,directory2,directory3)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    file_path = os.path.join(directory, f"{stp}.png")
                    Save_draw(file_path)


        # TODO figure out about right way to use replay
        agent.replay(BATCH_SIZE)

        debug(f'episode: {e + 1}/{NUMBER_OF_GAMES}, score: {score}, sim_score: {sim_score}')

        scores.append(score)

        if e % SAVE_RATE == 0:
            directory = "models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, f"{agent.model.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(file_path)
            debug("Saved model to disk")

