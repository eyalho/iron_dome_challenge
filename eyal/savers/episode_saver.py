
class EpisodeSaver:
    def __init__(self):
        self.r_locs = None
        self.i_locs = None
        self.c_locs = None
        self.ang = None
        self.score = None

    def save_episode_stats_to_json_file(self):
        """
        create a dict of the form {stp_0: step_stats, ..., stp_1000: stp_stats}.
        When episode end, save this dict as json file
        This make a later on processing possible.
        """
        pass