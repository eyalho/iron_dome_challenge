import importlib
import os
import time
import uuid

from keras.engine.saving import load_model

from savers.debug_logger import create_logger
from savers.episodes_saver import EpisodesSaver
from savers.game_saver import GameSaver
from savers.python_files_saver import save_program_files
from simulator.simulate_action import ActionsPredictor


class Conf:
    NUMBER_OF_STEPS_IN_GAME = 1000  # total frames in a game
    MAX_DIFF_SIM_SCORE = 10
    MAX_ANG = 90

    def __init__(self):
        # set default values:
        self.running_id = str(uuid.uuid4())[:5]
        self.logger = create_logger(self.running_id)
        self.results_folder = os.path.join("results_folder", self.running_id)
        self.max_episodes = 50000
        self.episodes_save_period = 10
        self.game_step_save_period = 10
        self.MAX_STEP_FOR_SIMULATE = 150
        self.SHOOT = 3
        self.stop_render = False
        self.batch_size = int(Conf.NUMBER_OF_STEPS_IN_GAME)
        self.saved_model_path = None
        self.agent = None
        self.env = None

        # parse CLI to update values
        self.parse_command_line()
        if self.saved_model_path is not None:
            self.model = load_model(self.saved_model_path)

    def parse_command_line(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--agent_filename')
        parser.add_argument('--env_filename')
        parser.add_argument('--saved_model_path')
        parser.add_argument('--max_episodes')
        parser.add_argument('--episodes_save_period')
        parser.add_argument('--game_step_save_period')
        parser.add_argument('--batch_size')
        parser.add_argument('--stop_render')
        args = parser.parse_args()

        # play at most max_episodes
        if args.agent_filename:
            agent_filename = args.agent_filename
            full_module_name = "agents." + agent_filename
            agent_module = importlib.import_module(full_module_name)
            self.agent = agent_module.create_agent()
        else:
            raise Exception("MUST CHOOSE AN AGENT:\n"
                            "Usage python trainer.py --agent_filename=<XXX> (filename without .py)")

        if args.env_filename:
            env_module_name = "envs." + args.env_filename
            self.env = importlib.import_module(env_module_name)
        else:
            raise Exception("MUST CHOOSE AN ENV:\n"
                            "Usage python trainer.py --env_filename=<XXX> (filename without .py)")

        if args.saved_model_path:
            self.saved_model_path = args.saved_model_path

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

        if args.stop_render:
            self.stop_render = True


if __name__ == "__main__":
    conf = Conf()
    # conf.NUMBER_OF_STEPS_IN_GAME = 100
    # conf.batch_size = 100
    agent = conf.agent
    debug = conf.logger.debug

    save_program_files(os.path.join(conf.results_folder, "py"))

    debug(f"{agent.name}")
    debug(f"{conf.__dict__}")
    debug(f"\nstart train of {conf.max_episodes} episodes, with batch size {conf.batch_size}")
    for e in range(conf.max_episodes):
        conf.env.Init()
        state = agent.init_state()
        score = 0
        reward = 0
        last_score = 0
        for stp in range(conf.NUMBER_OF_STEPS_IN_GAME):
            # save last score
            last_score = score
            stp_left = conf.NUMBER_OF_STEPS_IN_GAME - stp

            # run simulate for calc reward
            if agent.simulate_reward:
                actions_predictor = ActionsPredictor(conf, stp)
                predicted_shoot_score, predicted_wait_score = actions_predictor.predict_scores()
                reward = predicted_shoot_score - predicted_wait_score

            # let agent act next step
            action = agent.act(state)

            if agent.simulate_reward and action != conf.SHOOT:
                reward = -reward

            # play next step
            r_locs, i_locs, c_locs, ang, score = conf.env.Game_step(action)

            # calc reward
            if agent.simulate_reward:
                if action != conf.SHOOT:
                    reward = -reward
            else:
                reward = score - last_score

            # reformat the state to model input
            next_state = agent.create_state(conf, r_locs, i_locs, c_locs, ang, reward, stp)

            is_done = stp == conf.NUMBER_OF_STEPS_IN_GAME

            agent.memorize(state, action, reward, next_state, is_done)
            state = next_state

            # load games saver with data
            # once in _ episodes play on x_ fast forward
            # conf.episodes_save_period = 1
            if stp % conf.game_step_save_period == 0 and e % conf.episodes_save_period == 0:
                if stp == 0:
                    game_saver = GameSaver(e, conf.results_folder)  # init an empty saver
                game_saver.update(r_locs, i_locs, c_locs, ang, score, stp, action, reward)
                # screenshots take longer, so save once in a 10 saver
                if not conf.stop_render:
                    if stp % (conf.episodes_save_period * 10) == 0 and stp % (conf.game_step_save_period * 10) == 0:
                        game_saver.save_screen_shot(stp, conf.env.Save_draw)

        ################# END of game #################
        debug(f'episode: {e + 1}/{conf.max_episodes}, score: {score}')

        # train based on memory
        agent.replay(conf.batch_size)

        # Apply savers
        if e % conf.episodes_save_period == 0:
            if e == 0:
                episodes_saver = EpisodesSaver(conf.results_folder)  # init an empty saver

            # save games saver to files
            game_saver.save_game()
            episodes_saver.update(game_saver)
            episodes_saver.save_episodes()

            weights_directory = os.path.join(conf.results_folder, "models")
            if not os.path.exists(weights_directory):
                os.makedirs(weights_directory)
            weights_file_path = os.path.join(weights_directory,
                                             f"{agent.name}_e{e}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
            agent.model.save(weights_file_path)
            debug("Saved model to disk")
