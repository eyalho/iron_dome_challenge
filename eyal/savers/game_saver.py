#### Games Savers ####
# [X] screenshot vs stp
# [X] score vs stp
# [X] count(shoot) vs stp
# [X] angle vs stp
# [X] histogram(angle)
# [ ] simulated_score vs stp
# [ ] histogram(simulated_score)
# [ ] json all data vs stp
# [ ] count(agent_listen to simulator) vs stp
# [ ] count(success destroy missile) vs stp
# [ ] count(non-success city hit) vs stp
import json
import os

import matplotlib.pyplot as plt


class GameSaver:
    def __init__(self, e, base_folder):
        self.e = e
        self.steps_dict = dict()
        self.base_folder = base_folder
        self.plots_folder = os.path.join(self.base_folder, "plots")
        self.steps = []
        self.scores = []
        self.shoots_counter = 0
        self.count_shoots_list = []
        self.angles_list = []

    def update(self, r_locs, i_locs, c_locs, ang, score, stp, action):
        self.steps.append(stp)
        self.scores.append(score)
        if action == 3:
            self.shoots_counter += 1
        self.count_shoots_list.append(self.shoots_counter)
        self.angles_list.append(ang)

    def save_game(self):
        self.save_scores_files()
        self.save_shoots_counter_files()
        self.save_angle_files()

    def save_scores_files(self):
        self.save_generic_plot(self.scores, "Score", "Score vs Step", "score.png", "score.txt")

    def save_shoots_counter_files(self):
        self.save_generic_plot(self.count_shoots_list, "count(Shoots)", "count(Shoots) vs Step",
                               "count_shoots.png", "count_shoots.txt")

    def save_angle_files(self):
        self.save_generic_plot(self.angles_list, "Turret's Angle", "Turret Angle vs Step",
                               "turret_angel.png", "turret_angel.txt")
        self.save_histogram_plot(self.angles_list, "Turret's Angle", "Turret's Angle histogram",
                                 "turret_angel_hist.png")

    def save_generic_plot(self, data, y_name, title, fig_filename, data_filename):
        save_dir = os.path.join(self.plots_folder, f"e{self.e}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig_path = os.path.join(save_dir, fig_filename)
        data_path = os.path.join(save_dir, data_filename)
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'white'
        plt.title(title)
        plt.plot(self.steps, data, 'bo')
        plt.xlabel("Steps")
        plt.ylabel(y_name)
        plt.savefig(fig_path)
        plt.close

        with open(data_path, "w") as f:
            f.write(str(data))

    def save_histogram_plot(self, data, data_name, title, score_fig_filename):
        score_fig_path = os.path.join(self.plots_folder, f"e{self.e}", score_fig_filename)
        plt.figure()
        plt.title(title)
        plt.hist(data, density=1, bins=30)
        plt.axis([-90, 90, 0, 1])
        # axis([xmin,xmax,ymin,ymax])
        plt.xlabel("Distribution")
        plt.ylabel(data_name)
        plt.savefig(score_fig_path)
        plt.close

    def save_screen_shot(self, stp, Save_draw_func):
        screen_shots_dir = os.path.join(self.plots_folder, f"e{self.e}", "screen_shots")
        if not os.path.exists(screen_shots_dir):
            os.makedirs(screen_shots_dir)
        file_path = os.path.join(screen_shots_dir, f"{stp}.png")
        Save_draw_func(file_path)

    def init_dict(self):
        self.steps_dict = dict()

    def update_dict(self, stp, data):
        if len(self.steps_dict) != stp - 1:
            raise ValueError(f"expected stp={len(self.steps_dict)}, but got {stp}")
        if self.steps_dict.get(stp, False):
            raise ValueError(f"key {stp} already exists in EpisodeSaver.steps_dict")
        self.steps_dict[stp] = data

    def save_episode_stats_to_json_file(self, file_path):
        """
        create a dict of the form {stp_0: step_stats, ..., stp_1000: stp_stats}.
        When episode end, save this dict as json file
        This make a later on processing possible.
        """
        with open(file_path, 'w') as f:
            json.dump(self.steps_dict, f)
