# Deep Q-learning Agent
"""
We figured out that this the naive though is too hard for a short time training..
So, next we tried to create as simple network as we could
"""

import numpy as np
from keras.layers import Dense, Input, concatenate
from keras.models import Model

from agents.abstract_agent import ABSDQNAgent
from simulator.simulate_action import predict_scores


def create_agent():
    return SimpleModelDQNAgent()


class SimpleModelDQNAgent(ABSDQNAgent):

    def _build_model(self):
        hidden_size = 24
        # Input Layer
        ang_input_layer = Input(shape=(1,), name="angle")  # Turret angle (ang)
        sim_score_input_layer = Input(shape=(1,), name="simulate_score")  # Turret angle (ang)
        time_input_layer = Input(shape=(1,), name="time")  # time_t
        layer = concatenate(
            [ang_input_layer, sim_score_input_layer, time_input_layer])
        layer = Dense(hidden_size, activation='relu')(layer)
        layer = Dense(hidden_size, activation='relu')(layer)
        output_layer = Dense(self.action_size, activation='linear')(layer)
        model = Model(
            inputs=[ang_input_layer, sim_score_input_layer, time_input_layer],
            outputs=output_layer, name="model_simple")
        model.compile(optimizer='adam', loss='mse')

        return model

    def init_state(self):
        state = [np.array([0]), np.array([0]), np.array([0])]
        return state

    def create_state(self, conf, r_locs, i_locs, c_locs, ang, score, stp):
        predicted_shoot_score, predicted_wait_score = predict_scores(conf, stp)
        normalized_sim_score = (predicted_shoot_score - predicted_wait_score) / conf.MAX_DIFF_SIM_SCORE
        normalized_t = stp / conf.NUMBER_OF_STEPS_IN_GAME
        normalized_ang = ang / conf.MAX_ANG
        next_state = [np.array([normalized_ang]), np.array([normalized_sim_score]), np.array([normalized_t])]
        return next_state
