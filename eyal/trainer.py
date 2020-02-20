import os
import time
import uuid

import importlib

from savers.debug_logger import create_logger
from savers.episode_saver import EpisodeSaver
from envs.env_for_training import Init, Game_step, Save_draw
from savers.games_saver import GamesSaver
from savers.python_files_saver import save_program_files


class Conf:
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    MAX_DIFF_SIM_SCORE = 10
    MAX_ANG = 360

    def __init__(self):
        # set default values:
        self.running_id = str(uuid.uuid4())[:5]
        self.logger = create_logger(self.running_id)
        self.results_folder = os.path.join("results_folder", self.running_id)
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
        parser.add_argument('--max_episodes')
        parser.add_argument('--episodes_save_period')
        parser.add_argument('--game_step_save_period')
        parser.add_argument('--batch_size')
        args = parser.parse_args()

        # play at most max_episodes
        if args.agent_filename:
            agent_filename = args.agent_filename
            full_module_name = "agents." + agent_filename
            agent_module = importlib.import_module(full_module_name)
            self.agent = agent_module.create_agent()
        else:
            raise Exception("MUST CHOOSE AN AGENT")

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
    conf.NUMBER_OF_STEPS_IN_GAME = 100
    conf.batch_size = 100
    agent = conf.agent
    debug = conf.logger.debug

    save_program_files(os.path.join(conf.results_folder, "py"))

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
            # if conf.simulate:
            #     predicted_shoot_score, predicted_wait_score = predict_scores(stp)
            #     diff_sim_score = predicted_shoot_score - predicted_wait_score
            #     # calc reward based on last_score + how much the action was good compared to other action
            #     if conf.simulate_reward:
            #         if action == 3:
            #             reward = last_score + (predicted_shoot_score - predicted_wait_score)
            #         else:
            #             reward = last_score - (predicted_shoot_score - predicted_wait_score)

            # play next step
            r_locs, i_locs, c_locs, ang, score = Game_step(action)

            # reformat the state to model input
            next_state = agent.create_state(r_locs, i_locs, c_locs, ang, score, stp)

            is_done = stp == conf.NUMBER_OF_STEPS_IN_GAME

            agent.memorize(state, action, score, next_state, is_done)
            state = next_state

            # Apply Game savers
            # once in _ episodes play on x_ fast forward
            conf.episodes_save_period = 1
            if stp % conf.game_step_save_period == 0 and e % conf.episodes_save_period == 0:
                if stp == 0:
                    games_saver = GamesSaver(conf.results_folder)  # init an empty saver
                games_saver.save_screen_shot(e, stp, Save_draw)
                games_saver.update(r_locs, i_locs, c_locs, ang, score, stp, action)


        ################# END of game #################
        games_saver.save_game(e)
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
            weights_file_path = os.path.join(directory,
                                             f"{agent.model.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(weights_file_path)
            debug("Saved model to disk")
