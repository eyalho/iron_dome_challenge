#### Episodes Saver ####
# [ ] final_score vs episode
# [ ] count(shoot) vs episode
# [ ] average(angle) vs episode
# [ ] final_simulated_score vs episode
# [ ] average(simulated_score) vs episode
# [ ] count(agent_listen to simulator) vs episode

class EpisodeSaver:
    def __init__(self):
        self.r_locs = None
        self.i_locs = None
        self.c_locs = None
        self.ang = None
        self.score = None

