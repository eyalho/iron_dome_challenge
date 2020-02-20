import os
import time
import uuid

import importlib

from savers.debug_logger import create_logger
from savers.episode_saver import EpisodeSaver
from envs.env_for_training import Init, Game_step, Save_draw
from simulator.simulate_action import predict_scores


class Conf:
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    MAX_DIFF_SIM_SCORE = 10
    MAX_ANG = 360
    unique_id = str(uuid.uuid4())[:5]
    results_folder = "results"

    def __init__(self):
        # set default values:
        self.logger = create_logger(Conf.unique_id)
        self.simulate = False
        self.simulate_reward = False
        self.max_episodes = 50000
        self.episodes_save_period = 50
        self.game_step_save_period = 10
        self.batch_size = int(Conf.NUMBER_OF_STEPS_IN_GAME)
        self.agent = None
        
        # parse CLI to update values
        self.parse_command_line()
        
    def parse_command_line(self):
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
            self.agent = agent_module.DQNAgent()
        else:
            raise Exception("MUST CHOOSE AN AGENT")
    
        # True or False: should we use the simulator on each step to evaluate the action score
        if args.simulate:
            simulate = args.simulate
            if str(simulate).lower() == "true":
                self.simulate = True
    
        # True or False: reward = last_score + simulator_diff_score(action)
        if args.simulate_reward:
            simulate_reward = args.simulate_reward
            if str(simulate_reward).lower() == "true":
                self.simulate = True
                self.simulate_reward = True
    
        # run for at most max_episodes
        if args.max_episodes:
            self.max_episodes = int(args.max_episodes)
    
        # save weights each int(episodes_save_period) episodes
        if args.episodes_save_period:
            self.episodes_save_period = int(args.episodes_save_period)
    
        # When save weights, save each (game_step_save_period) step
        if args.game_step_save_period:
            self.game_step_save_period = int(args.game_step_save_period)
    
        if args.batch_size:
            self.batch_size = int(args.batch_size)


if __name__ == "__main__":
    conf = Conf()
    agent = conf.agent
    debug = conf.logger.debug

    debug(f"\nstart train of {conf.max_episodes} episodes, with batch size {conf.batch_size}")
    for e in range(conf.max_episodes):
        score = 0
        last_score = 0
        Init()
        state = agent.init_state()
        for stp in range(conf.NUMBER_OF_STEPS_IN_GAME):
            stp_left = conf.NUMBER_OF_STEPS_IN_GAME - stp
            predicted_shoot_score = predicted_wait_score = 0

            # save last score
            last_score = score

            # let agent act next step
            action = agent.act(state)

            # TODO MOVE ALL SIMULATE LOGIC TO AGENT FILE
            # if FLAG, simulate next step in order to understand if that was a good action
            if conf.simulate:
                predicted_shoot_score, predicted_wait_score = predict_scores(stp_left)
                diff_sim_score = predicted_shoot_score - predicted_wait_score
                # calc reward based on last_score + how much the action was good compared to other action
                if conf.simulate_reward:
                    if action == 3:
                        reward = last_score + (predicted_shoot_score - predicted_wait_score)
                    else:
                        reward = last_score - (predicted_shoot_score - predicted_wait_score)

            # play next step
            r_locs, i_locs, c_locs, ang, score = Game_step(action)

            # reformat the state to model input
            next_state = agent.create_state(r_locs, i_locs, c_locs, ang, score, stp,
                                                predicted_shoot_score, predicted_wait_score)

            is_done = stp == conf.NUMBER_OF_STEPS_IN_GAME

            agent.memorize(state, action, score, next_state, is_done)
            state = next_state

            # Apply savers
            # once in _ episodes play on x_ fast forward
            if stp % conf.game_step_save_period == 0 and e % conf.episodes_save_period == 0 and e != 0:
                if stp == 0:
                    epi_saver = EpisodeSaver()  # init an empty saver
                directory1 = "plots"
                directory2 = f"{agent.model.name}_{conf.unique_id}"
                directory3 = f"e{e}"
                directory = os.path.join(conf.results_folder, directory1, directory2, directory3)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                file_path = os.path.join(directory, f"{stp}.png")
                Save_draw(file_path)

        ################# END of game #################
        # TODO figure out about right way to use replay
        debug(f'episode: {e + 1}/{conf.max_episodes}, score: {score}')

        # train based on memory
        agent.replay(conf.batch_size)

        # Apply savers
        if e % conf.episodes_save_period == 0:
            directory1 = "models"
            directory = os.path.join(conf.results_folder, directory1)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, f"{agent.model.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(file_path)
            debug("Saved model to disk")
