# -*- coding: utf-8 -*-

def build_model():
    from keras.layers import Dense, Input, LSTM, concatenate
    from keras.models import Model
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


def test_model():
    import numpy as np
    from Interceptor_V2 import Init, Game_step
    model = build_model()
    Init()
    action_button = 3
    default_val = np.array([[-1, -1]])
    r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    r_locs = np.concatenate([default_val, r_locs])
    i_locs = np.concatenate([default_val, i_locs])
    next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]
    state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]
    model.fit(state, np.array([[0, 0, 0, 0]]))
    prediction = model.predict(state)
    print(prediction)
    assert prediction is not None


def test_globals():
    """
    test: can we use the globals from Interceptor_V2
    (global world, turret, rocket_list, interceptor_list, city_list, explosion_list)
    :return:
    """
    import Interceptor_V2
    from Interceptor_V2 import Init
    # global world, turret, rocket_list, interceptor_list, city_list, explosion_list
    Init()
    assert Interceptor_V2.world is not None
    cl = Interceptor_V2.city_list
    print(f" city list is {[(c.x, c.width) for c in cl]}")


if __name__ == "__main__":
    # test_model()
    test_globals()
