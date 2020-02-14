# -*- coding: utf-8 -*-
import numpy as np
from keras.layers import Dense, Input, LSTM, concatenate
from keras.models import Model

from Interceptor_V2 import Init, Game_step


def build_model():
    hidden_size = 20
    output_size = 4
    r_locs_input_layer = Input(shape=(None, 2))
    i_locs_input_layer = Input(shape=(None, 2))
    c_locs_input_layer = Input(shape=(None, 2))
    ang_input_layer = Input(shape=(1,))
    r_locs_lstm_layer = LSTM(hidden_size)(r_locs_input_layer)
    i_locs_lstm_layer = LSTM(hidden_size)(i_locs_input_layer)
    c_locs_lstm_layer = LSTM(hidden_size)(c_locs_input_layer)
    layer = concatenate([r_locs_lstm_layer, i_locs_lstm_layer, c_locs_lstm_layer, ang_input_layer])
    layer = Dense(hidden_size, activation='linear')(layer)
    output_layer = Dense(output_size, activation='tanh')(layer)
    model = Model(inputs=[r_locs_input_layer, i_locs_input_layer, c_locs_input_layer, ang_input_layer],
                  outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model


model = build_model()
Init()
action_button = 3
# r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
# r_locs = [[-1, -1]] + r_locs
# i_locs = [[-1, -1]] + i_locs
default_val = np.array([[-1, -1]])
r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
r_locs = np.concatenate([default_val, r_locs])
i_locs = np.concatenate([default_val, i_locs])
next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]

# print(model.predict([r_locs, i_locs, c_locs, ang]))
state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]
model.fit(state, np.array([[0, 0, 0, 0]]))
print(model.predict(state))
# for stp in range(1000):
#     action_button = 3
#     r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
#     Draw()
#     print(r_locs, i_locs, c_locs, ang, score)
