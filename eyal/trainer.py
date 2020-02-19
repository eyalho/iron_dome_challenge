import os
import time
import uuid

import numpy as np
import importlib

from savers.debug_logger import create_logger
from envs.env_for_training import Init, Game_step, Save_draw
from simulator.simulate_action import predict_scores


#TODO complete
class Conf:
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    unique_id = str(uuid.uuid4())[:5]
    MAX_DIFF_SIM_SCORE = 10
    MAX_ANG = 360
    game_step_save_period = 10
    
    def __init__(self):
        self.logger = create_logger(Conf.unique_id)
        self.debug = self.logger.debug


    # def parse_command_line(conf):
    #     pass


if __name__ == "__main__":
    conf = Conf()
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
    # TODO refactor reward
    # TODO refactor model
    # TODO refactor features
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

    # When save weights, save each (game_step_save_period) step
    if args.game_step_save_period:
        Conf.game_step_save_period = int(args.game_step_save_period)

    # When save weights, save each (Conf.game_step_save_period)th step
    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = int(Conf.NUMBER_OF_STEPS_IN_GAME)
    ################################################




    conf.debug(f"\nstart train of {max_episodes} episodes, with batch size {batch_size}")
    for e in range(max_episodes):
        score = 0
        last_score = 0
        Init()
        state = agent.init_state()
        for stp in range(Conf.NUMBER_OF_STEPS_IN_GAME):
            stp_left = Conf.NUMBER_OF_STEPS_IN_GAME - stp
            predicted_shoot_score = predicted_wait_score = 0

            # save last score
            last_score = score

            # let agent act next step
            action = agent.act(state)

            # if FLAG, simulate next step in order to understand if that was a good action
            if simulate:
                predicted_shoot_score, predicted_wait_score = predict_scores(stp_left)
                diff_sim_score = predicted_shoot_score - predicted_wait_score
                # calc reward based on last_score + how much the action was good compared to other action
                if simulate_reward:
                    if action == 3:
                        reward = last_score + (predicted_shoot_score - predicted_wait_score)
                    else:
                        reward = last_score - (predicted_shoot_score - predicted_wait_score)

            # play next step
            r_locs, i_locs, c_locs, ang, score = Game_step(action)

            # reformat the state to model input
            next_state = agent.create_state(r_locs, i_locs, c_locs, ang, score, stp,
                                                predicted_shoot_score, predicted_wait_score)

            is_done = stp == Conf.NUMBER_OF_STEPS_IN_GAME

            agent.memorize(state, action, score, next_state, is_done)
            state = next_state

            # Apply savers
            # once in _ episodes play on x_ fast forward
            if stp % Conf.game_step_save_period == 0 and e % episodes_save_period == 0 and stp != 0:
                directory0 = "results"
                directory1 = "plots"
                directory2 = f"{agent.model.name}_{Conf.unique_id}"
                directory3 = f"e{e}"
                directory = os.path.join(directory0, directory1, directory2, directory3)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_path = os.path.join(directory, f"{stp}.png")
                Save_draw(file_path)

        ################# END of game #################
        # TODO figure out about right way to use replay
        agent.replay(batch_size)
        conf.debug(f'episode: {e + 1}/{max_episodes}, score: {score}')

        if e % episodes_save_period == 0:
            directory0 = "results"
            directory1 = "models"
            directory = os.path.join(directory0, directory1)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, f"{agent.model.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(file_path)
            conf.debug("Saved model to disk")
