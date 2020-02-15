import matplotlib as plt
import numpy as np

from DQNAgent import DQNAgent
from eyal.env_for_training import Init, Draw, Game_step

# run configurations
episodes = 1000
model_name = 'DQN 24x24'
render = True
batch_size = 32

# env and agent initialization
Init()
agent = DQNAgent(state_size=285, action_size=4)
scores = []


# Iterate the game
for e in range(episodes):

    # reset state in the beginning of each game
    Init()
    r_locs, i_locs, c_locs, ang, score = Game_step(1)
    state = np.concatenate((r_locs.flatten(), np.zeros((1, 140 - 2 *np.shape(r_locs)[0])), i_locs.flatten(), np.zeros((1, 140 - 2 *np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)
    state = np.reshape(state, [1, 285])

    # time_t represents each frame of the game
    for time_t in range(1000):
        # turn this on if you want to render
        if render:
            # once in 50 episodes play on x20 fast forward
            if e% 50 == 0 and time_t % 20 == 0:
                Draw()

        # Decide action
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        r_locs, i_locs, c_locs, ang, new_score = Game_step(action)
        next_state = np.concatenate((r_locs.flatten(), np.zeros((1, 140 - 2 * np.shape(r_locs)[0])), i_locs.flatten(),
                                     np.zeros((1, 140 - 2 * np.shape(i_locs)[0])), c_locs.flatten(), ang), axis=None)
        reward = new_score - score
        score = new_score
        next_state = np.reshape(next_state, [1, 285])

        # memorize the previous state, action, reward, and done

        agent.memorize(state, action, reward, next_state)

        # make next_state the new current state for the next frame.
        state = next_state
    # train the agent with the experience of the episode
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    scores.append(score)

    if e % 10 == 0:
        print("episode: {}/{}, score: {}".format(e, episodes, score))

        # save model once in a while
        if e % 50 == 0:
            agent.model.save("models/{}-{} episodes - model.hdf5".format(model_name, e))
            print("Saved model to disk")

# on end, plot scores vs episodes
plt.figure()
plt.plot(range(e + 1), scores)