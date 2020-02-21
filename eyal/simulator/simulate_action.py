# TODO MAKE IT POSSIBLE USE OTHER ENVS
import os

from simulator import simulate_Interceptor_V2 as sim_env


class ActionsPredictor:
    SHOOT = 3
    WAIT = 1

    def __init__(self, conf, stp):
        self.conf = conf
        self.stp = stp
        self.env = conf.env
        self.steps_left = conf.NUMBER_OF_STEPS_IN_GAME - stp
        self.steps_to_sim = min(self.steps_left, self.conf.MAX_STEP_FOR_SIMULATE)

    def predict_score(self, action, debug=False):
        filename = None
        if self.stp < 101 or self.stp > 150:
            debug = False

        if debug:
            save_dir = "debug_sim"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            basename = os.path.join(save_dir, f"stp{self.stp}_action{action}_simstep_")
        final_score = 0

        # init new simulate game
        sim_env.Simulate(self.env.world, self.env.turret, self.env.rocket_list, self.env.interceptor_list,
                         self.env.city_list, self.env.explosion_list)
        # act
        sim_env.simulate_game_step(action)

        # peace steps until end of game
        for i in range(max(self.steps_to_sim, 0)):
            if debug:
                filename = basename + str(i) + "png"
            _, _, _, _, final_score = sim_env.simulate_peace_step(filename)
        return final_score

    def predict_scores(self):
        """
        simulate 1 shoot and other peace steps : calc predicted_shoot_score
        simulate all peace: calc predicted_wait_score
        :param steps_to_sim: how many step until end of game (1000-stp)
        :return: predicted_shoot_score, predicted_wait_score
        """
        predicted_shoot_score = self.predict_score(self.SHOOT)
        predicted_wait_score = self.predict_score(self.WAIT)
        return predicted_shoot_score, predicted_wait_score
