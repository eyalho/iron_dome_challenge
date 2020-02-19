import os
import time
import uuid

import numpy as np
import importlib

from savers.debug_logger import create_logger
from envs.env_for_training import Init, Game_step, Save_draw
from simulator.simulate_action import predict_scores
from simulator.simulate_shoot import simulate_shoot_score




if __name__ == "__main__":
    NUMBER_OF_STEPS_IN_GAME = 100  # total frames in a game
    unique_id = str(uuid.uuid4())[:5]

    ################################################
    # Command Line Interface
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_filename')
    parser.add_argument('--simulate')
    parser.add_argument('--simulate_reward')
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
        if simulate == "Fasle":
            simulate = False
    else:
        simulate = True

    # True or False: reward = last_score + simulator_diff_score(action)
    if args.simulate_reward:
        simulate_reward = args.simulate_reward
        if simulate_reward == "Fasle":
            simulate_reward = False
    else:
        simulate_reward = True


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

    score=0
    debug(f"\nstart train of {max_episodes} episodes, with batch size {batch_size}")
    for e in range(max_episodes):
        Init()
        state = agent.init_state()
        sum_diff_sim_score = 0
        for stp in range(NUMBER_OF_STEPS_IN_GAME):
            stp_left = NUMBER_OF_STEPS_IN_GAME - stp

            # simulate next step
            last_score = score
            if simulate:
                predicted_shoot_score, predicted_wait_score = predict_scores(stp_left)
                diff_sim_score = predicted_shoot_score - predicted_wait_score
                sum_diff_sim_score += diff_sim_score
            else:
                predicted_shoot_score = predicted_wait_score = diff_sim_score = 0

            # let agent act next step
            action = agent.act(state)
            r_locs, i_locs, c_locs, ang, score = Game_step(action)

            MAX_ANG = 360
            MAX_DIFF_SIM_SCORE = 10
            normalized_ang = ang / MAX_ANG
            normalized_sim_score = diff_sim_score / MAX_DIFF_SIM_SCORE
            normalized_t = stp / NUMBER_OF_STEPS_IN_GAME

            #TODO REFACTOR
            if agent_filename == "simple_agent":
                next_state = [np.array([ang]), np.array([normalized_sim_score]), np.array([normalized_t])]
            if agent_filename == "naive_agent":
                default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
                r_locs = np.concatenate([default_val, r_locs])
                i_locs = np.concatenate([default_val, i_locs])
                next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                              np.array([normalized_t])]

            if simulate_reward:
                if action == 3:
                    score = last_score + diff_sim_score
                else:
                    score = last_score - diff_sim_score

            is_done = stp == NUMBER_OF_STEPS_IN_GAME

            agent.memorize(state, action, score, next_state, is_done)
            state = next_state

            # once in _ episodes play on x_ fast forward
            if stp % game_step_save_period == 0 and e % episodes_save_period == 0 and stp != 0:
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

        debug(f'episode: {e + 1}/{max_episodes}, score: {score}, sum_diff_sim_score: {sum_diff_sim_score}')

        if e % episodes_save_period == 0:
            directory0 = "results"
            directory1 = "models"
            directory = os.path.join(directory0, directory1)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, f"{agent.model.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(file_path)
            debug("Saved model to disk")
