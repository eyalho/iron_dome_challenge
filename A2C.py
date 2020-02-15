import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.python.framework.ops import disable_eager_execution
from IronDomeEnv import IronDomeEnv

disable_eager_execution()

env = IronDomeEnv()
state_size = env.observation_space.shape[0]
num_of_actions = env.action_space.n





class VNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size):
        self.v_target = tf.compat.v1.placeholder(shape=None, dtype=tf.float32, name='dvn_v_target')
        self.r = tf.compat.v1.placeholder(shape=None, dtype=tf.float32, name='dvn_r')
        self.states = tf.compat.v1.placeholder(shape=(None, input_size), dtype=tf.float32, name='dvn_states')

        _layer = self.states
        for l in hidden_layers_size:
            _layer = tf.compat.v1.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.predictions = tf.compat.v1.layers.dense(inputs=_layer, units=1, activation=None,  # Linear activation
                                           kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.labels = self.r + gamma * self.v_target
        self.cost = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


class PolicyGradient:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, num_of_actions):
        self.states = tf.compat.v1.placeholder(shape=(None, input_size), dtype=tf.float32, name='pg_states')
        self.r = tf.compat.v1.placeholder(shape=None, dtype=tf.float32, name='pg_r')
        self.v = tf.compat.v1.placeholder(shape=None, dtype=tf.float32, name='pg_v')
        self.v_target = tf.compat.v1.placeholder(shape=None, dtype=tf.float32, name='pg_v_target')
        self.actions = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='pg_actions')

        _layer = self.states
        for l in hidden_layers_size:
            _layer = tf.compat.v1.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.last_layer = tf.compat.v1.layers.dense(inputs=_layer, units=num_of_actions, activation=None,  # Linear activation
                                          kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.action_prob = tf.nn.softmax(self.last_layer)
        self.a = self.r + gamma * self.v_target - self.v
        self.log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.last_layer, labels=self.actions)
        self.cost = tf.reduce_mean(self.a * self.log_policy)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


class A2C:
    def __init__(self, input_size, num_of_actions, actor_hidden_layers_size, critic_hidden_layers_size,
                 actor_gamma, critic_gamma, actor_learning_rate, critic_learning_rate):
        self.critic = VNetwork(critic_hidden_layers_size, critic_gamma, critic_learning_rate, input_size)
        self.actor = PolicyGradient(actor_hidden_layers_size, actor_gamma, actor_learning_rate, input_size,
                                    num_of_actions)
        self.memory = []
        self.num_of_actions = num_of_actions

    def _extract_from_batch(self, batch, key):
        return np.array(list(map(lambda x: x[key], batch)))

    def _update_terminal_states(self, v_target, terminals):
        for i in range(len(terminals)):
            if terminals[i]:
                v_target[i] = 0.0
        return v_target

    def remember(self, **kwargs):
        self.memory.append(kwargs)

    def actions_prob(self, session, state):
        return session.run(self.actor.action_prob,
                           feed_dict={self.actor.states: np.expand_dims(state, axis=0)}).flatten()

    def act(self, session, state, greedy=False):
        actions_prob = self.actions_prob(session, state)
        if greedy:
            action = np.argmax(actions_prob)
        else:
            action = np.random.choice(self.num_of_actions, p=actions_prob)
        return action

    def learn(self, session):
        batch = shuffle(self.memory)
        next_states = self._extract_from_batch(batch, 'next_state')
        states = self._extract_from_batch(batch, 'state')
        rewards = self._extract_from_batch(batch, 'reward')
        actions = self._extract_from_batch(batch, 'action')
        terminals = self._extract_from_batch(batch, 'game_over')

        v = session.run(self.critic.predictions, feed_dict={self.critic.states: states})
        v_t = session.run(self.critic.predictions, feed_dict={self.critic.states: next_states})
        v_t = self._update_terminal_states(v_t, terminals=terminals)

        actor_cost, _, critic_cost, _ = session.run([self.actor.cost, self.actor.optimizer,
                                                     self.critic.cost, self.critic.optimizer],
                                                    feed_dict={self.actor.states: states,
                                                               self.actor.r: r,
                                                               self.actor.v: v,
                                                               self.actor.v_target: v_t,
                                                               self.actor.actions: actions,
                                                               self.critic.v_target: v_t,
                                                               self.critic.r: rewards,
                                                               self.critic.states: states})
        if np.isnan(actor_cost) or np.isnan(critic_cost): raise Exception('NaN cost!')
        self.memory = []
        return actor_cost, critic_cost


actor_hidden_layers = [24]
critic_hidden_layers = [24,24]
learning_rate = 0.001
gamma = 0.99

sess = tf.compat.v1.Session()
ac = A2C(input_size=state_size, num_of_actions=num_of_actions,
         actor_hidden_layers_size=actor_hidden_layers, critic_hidden_layers_size=critic_hidden_layers,
         actor_gamma=gamma, critic_gamma=gamma,
         actor_learning_rate=learning_rate, critic_learning_rate=learning_rate)


game_df = pd.DataFrame(columns=['game','steps','actor_cost','critic_cost'])
sess.run(tf.compat.v1.global_variables_initializer())


def print_stuff(s, every=50):
    if game % every == 0 or game == 1:
        print(s)

games = 1000

for g in range(games):
    game = g + 1
    game_over = False
    next_state = env.reset()
    steps = 0
    while not game_over:
        steps += 1
        state = np.copy(next_state)
        action = ac.act(sess, state)
        next_state, r, game_over, _ = env.step(action)
        if game_over and steps < env._max_episode_steps: r = 0
        ac.remember(state=state, action=action, reward=r, next_state=next_state, game_over=game_over)
        actor_cost, critic_cost = ac.learn(sess)
    print_stuff('Game {g} ended after {s} steps | Actor cost: {a:.2e}, Critic cost: {c:.2e}'.format(g=game, s=steps, a=actor_cost, c=critic_cost))
    game_df = game_df.append({'game':game, 'steps':steps, 'actor_cost':actor_cost, 'critic_cost':critic_cost},
                             ignore_index=True)
    if not g%50:
        next_state = env.reset()
        env.render()
        game_over = False
        steps = 0
        while not game_over:
            steps += 1
            state = np.copy(next_state)
            action = ac.act(sess, state, greedy=True)
            action_prob = ac.actions_prob(sess, state).tolist()
            next_state, _, game_over, _ = env.step(action)
            if not steps%15:
                env.render()
        print('Ended after {} steps'.format(steps))


game_df['steps_moving_average'] = game_df['steps'].rolling(window=50).mean()
ax = game_df.plot('game','steps_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Steps')
plt.show()

game_df['actor_cost_moving_average'] = game_df['actor_cost'].rolling(window=50).mean()
ax = game_df.plot('game','actor_cost_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Actor Cost')
plt.show()

game_df['critic_cost_moving_average'] = game_df['critic_cost'].rolling(window=50).mean()
ax = game_df.plot('game','critic_cost_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Critic Cost')
plt.show()