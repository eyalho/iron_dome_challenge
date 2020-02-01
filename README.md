# iron_dome_challenge
[Official challenge](http://portal.rafael.co.il/mlchallenge2019/Documents/index.html)

Refael posted an simplified simulation of Iron Dom:
- Enemy’s missiles are shoot toward the city
- Iron Dom shoots interceptors in order to destroy the the missiles before they hit
the city.

The goal of the game: Getting highest score in 100 games  each running for
1000 steps. At each step, The player can choose one of the follow actions:
- 0: Change turret angle one step left
- 1: Do nothing in the current game step
- 2: Change turret angle one step right
- 3: Fire



Referenced articles:

[Reinforcement Learning (CartPole game with Keras)](https://keon.github.io/deep-q-learning) - notice work until python 3.7.4