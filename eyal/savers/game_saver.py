#### Games Savers ####
# [ ] score vs stp
# [ ] count(shoot) vs stp
# [ ] angle vs stp
# [ ] histogram(angle)
# [ ] simulated_score vs stp
# [ ] histogram(simulated_score)
# [ ] json all data vs stp
# [ ] count(agent_listen to simulator) vs stp
import json


class GameSaver:
    def __init__(self):
        self.steps_dict = dict()
        pass

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
