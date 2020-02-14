import numpy as np
from eyal.agent import DQNAgent
from Interceptor_V2 import Init, Draw, Game_step

if __name__ == "__main__":
    NUMBER_OF_GAMES = 500
    NUMBER_OF_FRAMES_IN_GAME = 1000  # total frames in a game
    BATCH_SIZE = int(NUMBER_OF_FRAMES_IN_GAME / 10)
    agent = DQNAgent()

    default_val = np.array([[-1, -1]])  # init with invalid (x,y)
    for e in range(NUMBER_OF_GAMES):
        Init()
        r_locs = default_val  # rocket
        i_locs = default_val  # interceptor
        c_locs = default_val  # city
        ang = 0
        state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]
        for time_t in range(NUMBER_OF_FRAMES_IN_GAME):
            action = agent.act(state)
            r_locs, i_locs, c_locs, ang, score = Game_step(action)
            r_locs = np.concatenate([default_val, r_locs])
            i_locs = np.concatenate([default_val, i_locs])
            next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang])]
            agent.memorize(state, action, score, next_state, False)
            state = next_state

            if e % 50 == 49:
                Draw()
            # TODO figure out about right way to use replay
            if time_t % BATCH_SIZE == 0 and len(agent.memory) >= BATCH_SIZE:
                agent.replay(min(time_t, BATCH_SIZE))

        print(f'episode: {e + 1}/{NUMBER_OF_GAMES}, score: {score}')
