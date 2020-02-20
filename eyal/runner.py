import importlib
import os
import uuid

from keras.engine.saving import load_model

from envs.env_for_training import Init, Game_step, Save_draw
from savers.debug_logger import create_logger
from savers.episodes_saver import EpisodesSaver
from savers.game_saver import GameSaver
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
        self.max_episodes = 50000
        self.episodes_save_period = 50
        self.game_step_save_period = 10
        self.saved_model_path = None
        self.agent = None

        # parse CLI to update values
        self.parse_command_line()
        self.model = load_model(self.saved_model_path)

    def parse_command_line(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--agent_filename')
        parser.add_argument('--saved_model_path')
        parser.add_argument('--max_episodes')
        parser.add_argument('--episodes_save_period')
        parser.add_argument('--game_step_save_period')
        args = parser.parse_args()

        # play at most max_episodes
        if args.agent_filename:
            agent_filename = args.agent_filename
            full_module_name = "agents." + agent_filename
            agent_module = importlib.import_module(full_module_name)
            self.agent = agent_module.create_agent()
        else:
            raise Exception("MUST CHOOSE AN AGENT\n"
                            "Usage python runner.py --agent_filename=<XXX> --saved_model_path=<YYY>")

        if args.saved_model_path:
            self.saved_model_path = args.saved_model_path
        else:
            raise Exception("MUST GIVE SAVED_MODEL_PATH\n"
                            "Usage python runner.py --agent_filename=<XXX> --saved_model_path=<YYY>")

        # run for at most max_episodes
        if args.max_episodes:
            self.max_episodes = int(args.max_episodes)

        # save weights each int(episodes_save_period) episodes
        if args.episodes_save_period:
            self.episodes_save_period = int(args.episodes_save_period)

        # When save weights, save each (game_step_save_period) step
        if args.game_step_save_period:
            self.game_step_save_period = int(args.game_step_save_period)


if __name__ == "__main__":
    conf = Conf()
    # conf.NUMBER_OF_STEPS_IN_GAME = 100
    # conf.batch_size = 100
    agent = conf.agent
    debug = conf.logger.debug

    save_program_files(os.path.join(conf.results_folder, "py"))

    debug(f"\nload {conf.saved_model_path} into agents model")
    agent.model = conf.model

    debug(f"\nstart run of {conf.max_episodes} episodes")

    for e in range(conf.max_episodes):
        Init()
        state = agent.init_state()
        for stp in range(conf.NUMBER_OF_STEPS_IN_GAME):

            action = agent.act(state)

            r_locs, i_locs, c_locs, ang, score = Game_step(action)

            # reformat the state to model input
            next_state = agent.create_state(r_locs, i_locs, c_locs, ang, score, stp)

            is_done = stp == conf.NUMBER_OF_STEPS_IN_GAME

            agent.memorize(state, action, score, next_state, is_done)
            state = next_state

            # load games saver with data
            # once in _ episodes play on x_ fast forward
            # conf.episodes_save_period = 1
            if stp % conf.game_step_save_period == 0 and e % conf.episodes_save_period == 0:
                if stp == 0:
                    game_saver = GameSaver(e, conf.results_folder)  # init an empty saver
                # game_saver.save_screen_shot(stp, Save_draw)
                game_saver.update(r_locs, i_locs, c_locs, ang, score, stp, action)

        ################# END of game #################
        debug(f'episode: {e + 1}/{conf.max_episodes}, score: {score}')


        # Apply savers
        if e % conf.episodes_save_period == 0:
            if e == 0:
                episodes_saver = EpisodesSaver(conf.results_folder)  # init an empty saver

            # save games saver to files
            game_saver.save_game()
            episodes_saver.update(game_saver)
            episodes_saver.save_episodes()
