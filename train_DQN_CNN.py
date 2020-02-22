import matplotlib as plt
import os
import time
from IronDomeEnv import IronDomeEnv
from debug_logger import create_logger
from DQN_CNN_agent import DQN_CNN_agent

logger = create_logger("train")
debug = logger.debug

load_weights = True
episode_ofset = 0
if load_weights:
    weights_path = './models/DQN CNN 42x84x12_e1150_2020_02_22-07_06_26.hdf5'
    episode_ofset = 1150


# run configurations
episodes = 50000
model_name = 'DQN CNN 42x84x30'
render = True
batch_size = 32

# env and agent initialization
env = IronDomeEnv(state_type='histogram_2D')
agent = DQN_CNN_agent(action_size=4, env=env)
if load_weights:
  agent.model.load_weights(weights_path)
  agent.epsilon = agent.epsilon_min
scores = []


# Iterate the game
for e in range(episodes-episode_ofset):

    # reset state in the beginning of each game
    state = env.reset()

    # time_t represents each frame of the game
    for time_t in range(env._max_episode_steps):
        # turn this on if you want to render
        if render:
            # once in 50 episodes play on x20 fast forward
            if e% 100 == 0 and time_t % 5 == 0:
                env.render()

        # Decide action
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        next_state, reward, done, _ = env.step(action)
        # memorize the previous state, action, reward, and done
        agent.memorize(state, action, reward, next_state)
        # make next_state the new current state for the next frame.
        state = next_state
    # train the agent with the experience of the episode
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        print("replaying memory")
    scores.append(env.score)


    print("episode: {}/{}, score: {}".format(e, episodes, env.score))

    # save model once in a while
    if e % 50 == 0:
        directory = "models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory,
                                 f"{model_name}_e{e + episode_ofset}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.hdf5")
        agent.model.save(file_path)
        debug("Saved model to disk")

# on end, plot scores vs episodes
plt.figure()
plt.plot(range(e + 1), scores)