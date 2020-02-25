from Interceptor_V2 import Init, Draw, Game_step
import matplotlib.pyplot as plt
import numpy as np
final_scores=[]
scores = np.zeros([1000,1000])
for e in range(1000):
    Init()
    for stp in range(1000):
        action_button = np.random.randint(0,3,(1,))
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
        scores[e,stp] = score
    final_scores.append(score)
    print(score)
print("max score is {}".format(np.max(final_scores)))
print("mean score is {}".format(np.mean(final_scores)))
print("scores std is {}".format(np.std(final_scores)))
plt.figure()
plt.plot(range(1000), scores)
