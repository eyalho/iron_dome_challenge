# naive
"""
Actually we were the naive not the agent.
Firstly, we tried just give all state data "as is" to the agent and let it figure out by itself
how to become the best player
"""

import numpy as np
from keras.layers import Dense, Input, LSTM, concatenate
from keras.models import Model

from agents.abstract_agent import ABSDQNAgent


def create_agent():
    return NaiveFullStateModelAgent()


class NaiveFullStateModelAgent(ABSDQNAgent):

    # if you want to call parent init + add your __init__ do so:
    # def __init__(self):
    #     super().__init__()  # call the __init__ of parent
    #     self.name = "naive_agent"

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        hidden_size = 20

        # Input Layer
        r_locs_input_layer = Input(shape=(None, 2),
                                   name="rockets")  # Location of each rocket np.array([[x,y],[x,y], ...])
        i_locs_input_layer = Input(shape=(None, 2),
                                   name="interceptor")  # Location of each interceptor np.array([[x,y],[x,y], ...])
        c_locs_input_layer = Input(shape=(None, 2), name="cities")  # Location of each city np.array([[x,y],[x,y], ...])
        ang_input_layer = Input(shape=(1,), name="angle")  # Turret angle (ang)
        features_input_layer = Input(shape=(1,), name="other_features")  # time_t

        # Add RNN with some memory to previous states
        r_locs_lstm_layer = LSTM(hidden_size)(r_locs_input_layer)  # institution: find worst rocket
        i_locs_lstm_layer = LSTM(hidden_size)(i_locs_input_layer)  # institution: find match interceptor
        c_locs_lstm_layer = LSTM(hidden_size)(c_locs_input_layer)  # institution: no real need for lstm

        layer = concatenate(
            [r_locs_lstm_layer, i_locs_lstm_layer, c_locs_lstm_layer, ang_input_layer, features_input_layer])
        layer = Dense(hidden_size, activation='linear')(layer)
        output_layer = Dense(self.action_size, activation='linear')(layer)
        model = Model(
            inputs=[r_locs_input_layer, i_locs_input_layer, c_locs_input_layer, ang_input_layer, features_input_layer],
            outputs=output_layer, name="model_full_state")
        model.compile(optimizer='adam', loss='mse')
        return model

    def init_state(self):
        default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
        # just for the case where there are no r_locs/i_locs
        r_locs = default_val  # rocket
        i_locs = default_val  # interceptor
        c_locs = default_val  # city
        ang = 0
        normalized_t = 0
        state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                 np.array([normalized_t])]
        return state

    def create_state(self, conf, r_locs, i_locs, c_locs, ang, score, stp,
                     predicted_shoot_score=None, predicted_wait_score=None):
        normalized_t = stp / conf.NUMBER_OF_STEPS_IN_GAME
        normalized_ang = ang / conf.MAX_ANG
        default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
        r_locs = np.concatenate([default_val, r_locs])
        i_locs = np.concatenate([default_val, i_locs])
        next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([normalized_ang]),
                      np.array([normalized_t])]
        return next_state
