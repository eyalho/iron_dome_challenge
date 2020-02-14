import matplotlib.pyplot as plt
import numpy as np
from agent import DQNAgent

from Interceptor_V2 import Init, Draw, Game_step

if __name__ == "__main__":
    NUMBER_OF_GAMES = 500
    NUMBER_OF_FRAMES_IN_GAME = 1000  # total frames in a game
    BATCH_SIZE = int(NUMBER_OF_FRAMES_IN_GAME / 10)
    render = False
    agent = DQNAgent()
    scores = []

    with open("score.txt", "a") as f:
        f.write(f"start train of {NUMBER_OF_GAMES} episodes, with batch size {BATCH_SIZE}\n")

    default_val = np.array([[-1, -1]])  # init always with invalid (x,y)
    # just for the case where there are no r_locs/i_locs
    try:
        for e in range(NUMBER_OF_GAMES):
            Init()
            r_locs = default_val  # rocket
            i_locs = default_val  # interceptor
            c_locs = default_val  # city
            ang = 0
            normalized_t = 0
            state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                     np.array([normalized_t])]
            for time_t in range(NUMBER_OF_FRAMES_IN_GAME):
                normalized_t = time_t / NUMBER_OF_FRAMES_IN_GAME
                action = agent.act(state)
                r_locs, i_locs, c_locs, ang, score = Game_step(action)
                r_locs = np.concatenate([default_val, r_locs])
                i_locs = np.concatenate([default_val, i_locs])
                next_state = [np.array([r_locs]), np.array([i_locs]), np.array([c_locs]), np.array([ang]),
                              np.array([normalized_t])]
                is_done = time_t == NUMBER_OF_FRAMES_IN_GAME
                agent.memorize(state, action, score, next_state, is_done)

                state = next_state

                # turn this on if you want to render
                if render:
                    # once in _ episodes play on x_ fast forward
                    if e % 3 == 0 and time_t % 20 == 0:
                        Draw()

            # TODO figure out about right way to use replay
            agent.replay(BATCH_SIZE)

            with open("score.txt", "a") as f:
                f.write(f'episode: {e + 1}/{NUMBER_OF_GAMES}, score: {score}\n')
            print(f'episode: {e + 1}/{NUMBER_OF_GAMES}, score: {score}')
            scores.append(score)
    except KeyboardInterrupt:
        plt.plot(scores)
        import time

        time.sleep(1000)
