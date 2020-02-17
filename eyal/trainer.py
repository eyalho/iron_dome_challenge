import os
import time
import uuid

import numpy as np
import importlib

from savers.debug_logger import create_logger
from envs.env_for_training import Init, Game_step, Save_draw
from simulator.simulate_action import predict_scores
from simulator.simulate_shoot import simulate_shoot_score


def eval_score(predicted_action, ang, score, steps_to_sim):
    """
    :param steps_to_sim: how many step until end of game (1000-stp)
    :param action_button: deserved action
    :return: the score of the game
    """
    # SHOOT = 3
    # ANGLE_SCORE_PUNISHMENT = 100

    ############### SHOOT reward ###############
    # We want the agent to shoot when he's going to gain score.
    # But the score will be received in the far future.
    # So we simulate next steps (with no other shooting) and calc
    # how the score will be difference if he chose to shoot..
    shoot_score = simulate_shoot_score(steps_to_sim)
    # if predicted_action == SHOOT:
    #    score += shoot_score
    # else:
    #    score -= shoot_score
    #
    ############### ANGLE reward ###############
    # While playing we noticed that optimal angel should be
    # between 12 to 72
    # if ang < 12 or score > 72:
    #    score -= ANGLE_SCORE_PUNISHMENT
    score = shoot_score
    return score


if __name__ == "__main__":
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    unique_id = str(uuid.uuid4())[:5]

    ################################################
    # Command Line Interface
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_filename')
    parser.add_argument('--simulate')
    parser.add_argument('--max_episodes')
    parser.add_argument('--episodes_save_period')
    parser.add_argument('--game_step_save_period')
    parser.add_argument('--batch_size')
    #TODO refactor reward
    #TODO refactor model
    #TODO refactor features
    args = parser.parse_args()

    # play at most max_episodes
    if args.agent_filename:
        agent_filename = args.agent_filename
        full_module_name = "agents." + agent_filename
        agent_module = importlib.import_module(full_module_name)
        agent = agent_module.DQNAgent()
    else:
        raise Exception("MUST CHOOSE AN AGENT")

    # True or False: should we use the simulator on each step to evaluate the action score
    if args.simulate:
        simulate = args.simulate
    else:
        simulate = True

    # run for at most max_episodes
    if args.max_episodes:
        max_episodes = int(args.max_episodes)
    else:
        max_episodes = 50000

    # play at most max_episodes
    if args.max_episodes:
        max_episodes = int(args.max_episodes)
    else:
        max_episodes = 50000

    # save weights each int(episodes_save_period) episodes
    if args.episodes_save_period:
        episodes_save_period = int(args.episodes_save_period)
    else:
        episodes_save_period = 50

    # When save weights, save each (game_step_save_period)th step
    if args.game_step_save_period:
        game_step_save_period = int(args.game_step_save_period)
    else:
        game_step_save_period = 10

    # When save weights, save each (game_step_save_period)th step
    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = int(NUMBER_OF_STEPS_IN_GAME)
    ################################################

    logger = create_logger(unique_id)
    debug = logger.debug

    # print values given by cli
    for kwarg in args._get_kwargs():
        debug(f"{kwarg[0]}, {eval(kwarg[0])}")

    Init()
    state = agent.init_state()
    debug(f"\nstart train of {max_episodes} episodes, with batch size {batch_size}")
    for e in range(max_episodes):
        for stp in range(NUMBER_OF_STEPS_IN_GAME):
            stp_left = NUMBER_OF_STEPS_IN_GAME - stp
            action = agent.act(state)
            r_locs, i_locs, c_locs, ang, score = Game_step(action)

            if simulate:
                predicted_shoot_score, predicted_wait_score = predict_scores(stp_left)
                diff_sim_score = predicted_shoot_score - predicted_wait_score

            MAX_ANG = 360
            MAX_DIFF_SIM_SCORE = 10
            normalized_ang = ang / MAX_ANG
            normalized_sim_score = diff_sim_score / MAX_DIFF_SIM_SCORE
            normalized_t = stp / NUMBER_OF_STEPS_IN_GAME
            next_state = [np.array([ang]), np.array([normalized_sim_score]), np.array([normalized_t])]
            is_done = stp == NUMBER_OF_STEPS_IN_GAME
            agent.memorize(state, action, score, next_state, is_done)
            state = next_state

            # once in _ episodes play on x_ fast forward
            if stp % game_step_save_period == 0 and e % episodes_save_period == 0:
                directory0 = "results"
                directory1 = "plots"
                directory2 = f"{agent.model.name}_{unique_id}"
                directory3 = f"e{e}"
                directory = os.path.join(directory0, directory1, directory2, directory3)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_path = os.path.join(directory, f"{stp}.png")
                Save_draw(file_path)

        # TODO figure out about right way to use replay
        agent.replay(batch_size)

        debug(f'episode: {e + 1}/{max_episodes}, score: {score}, sim_score: {diff_sim_score}')

        if e % episodes_save_period == 0:
            directory = "models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, f"{agent.model.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(file_path)
            debug("Saved model to disk")
