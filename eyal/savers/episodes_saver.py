#### Episodes Saver ####
# [X] final_score vs episode
# [X] count(shoot) vs episode
# [X] average(angle) vs episode
# [ ] final_simulated_score vs episode
# [ ] average(simulated_score) vs episode
# [ ] count(agent_listen to simulator) vs episode
import os
import statistics

import matplotlib.pyplot as plt

from savers.game_saver import GameSaver


class EpisodesSaver:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.plots_folder = os.path.join(self.base_folder, "plots", "episodes")
        self.episodes = []
        self.final_scores = []
        self.count_shoots = []
        self.avg_angles = []
        self.variance_angles = []

    def update(self, game_saver: GameSaver):
        # Episodes
        self.episodes.append(game_saver.e)
        # final score
        final_score = game_saver.scores[-1]
        self.final_scores.append(final_score)
        # count shoots
        count_shoot = game_saver.count_shoots_list[-1]
        self.count_shoots.append(count_shoot)
        # angles
        avg_angle = statistics.mean(game_saver.angles_list)
        variance_angle = statistics.variance(game_saver.angles_list)
        self.avg_angles.append(avg_angle)
        self.variance_angles.append(variance_angle)

    def save_episodes(self):
        self.save_generic_plot(self.final_scores, "Final Score", "Final Score vs Episode",
                               "e_final_score.png", "e_final_score.txt")
        self.save_generic_plot(self.count_shoots, "count(Shoots)", "count(Shoots) vs Episode",
                               "e_count_shoots.png", "e_count_shoots.txt")
        self.save_generic_plot(self.avg_angles, "avg(Turret's Angle)", "avg(Turret's Angle) vs Episode",
                               "e_turret_angel.png", "e_turret_angel.txt")
        self.save_avg_std_plots(self.avg_angles, self.variance_angles, "avg(Turret's Angle)",
                                "avg(Turret's Angle) vs Episode","e_turret_angel2.png", "e_turret_angel2.txt")

    def save_generic_plot(self, data, y_name, title, fig_filename, data_filename):
        save_dir = os.path.join(self.plots_folder, f"e{self.episodes[-1]}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig_path = os.path.join(save_dir, fig_filename)
        data_path = os.path.join(save_dir, data_filename)

        plt.figure()
        plt.rcParams['axes.facecolor'] = 'white'
        plt.title(title)
        plt.plot(self.episodes, data, 'bo')
        plt.xlabel("Episodes")
        plt.ylabel(y_name)
        plt.savefig(fig_path)
        plt.close

        with open(data_path, "w") as f:
            f.write(str(data))

    def save_avg_std_plots(self, avg, std, y_name, title, score_fig_filename, score_data_filename):
        score_fig_path = os.path.join(self.plots_folder, f"e{self.episodes[-1]}", score_fig_filename)
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'white'
        plt.title(title)
        plt.errorbar(self.episodes, avg, std, linestyle='None', marker='^')
        plt.xlabel("Episodes")
        plt.ylabel(y_name)
        plt.savefig(score_fig_path)
        plt.close

        score_data_path = os.path.join(self.plots_folder, f"e{self.episodes[-1]}", score_data_filename)
        with open(score_data_path, "w") as f:
            f.write(str(avg))
            f.write("\n")
            f.write(str(std))

