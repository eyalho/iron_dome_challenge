import env_for_training as env
import matplotlib.pyplot as plt
import numpy as np
import simulate_Interceptor_V2 as sim_env
from agent import DQNAgent
from debug_logger import create_logger
from env_for_training import Init, Draw, Game_step

logger = create_logger("train")
debug = logger.debug


def calc_fancy_reward(steps_to_sim):
    """
    :param steps_to_sim: how many step until end of game (1000-time_t)
    :param action_button: deserved action
    :return: the score of the game
    """
    sim_env.Simulate(env.world, env.turret, env.rocket_list, env.interceptor_list, env.city_list, env.explosion_list)
    actions = [0, 1, 2, 3]
    rewards = []
    for action_button in actions:
        sim_env.Game_step(action_button)
        for i in range(steps_to_sim - 2):
            sim_env.peace_step()
        r_locs, i_locs, c_locs, ang, score = sim_env.peace_step()
        rewards.append(score)

    debug(f"steps_to_sim = {steps_to_sim}\nrewards={rewards}")


if __name__ == "__main__":
    NUMBER_OF_GAMES = 500
    NUMBER_OF_FRAMES_IN_GAME = 1000  # total frames in a game
    BATCH_SIZE = int(NUMBER_OF_FRAMES_IN_GAME / 10)
    render = False
    agent = DQNAgent()
    scores = []

    debug(f"start train of {NUMBER_OF_GAMES} episodes, with batch size {BATCH_SIZE}\n")
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
                calc_fancy_reward(NUMBER_OF_FRAMES_IN_GAME - time_t)

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
            debug(f'episode: {e + 1}/{NUMBER_OF_GAMES}, score: {score}')
            scores.append(score)
    except KeyboardInterrupt:
        plt.plot(scores)
        import time

        time.sleep(1000)
