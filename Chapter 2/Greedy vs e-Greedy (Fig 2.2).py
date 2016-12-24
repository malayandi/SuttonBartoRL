''' Runs the Ten-Armed Bandit, comparing the performance of the greedy method
with e-greedy methods (both are examples of sample-average methods) on the
stationary problem.

Used to generate Figure 2.2 '''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Number of arms of bandit
k = 10

# Probabilities with which we explore instead of exploit
eps = [0, 0.1, 0.01]

num_runs = 1000
run_length = 1000

bestAvReward = 0
rewardAtStep = np.zeros((len(eps), run_length))
numOptimalAtStep = np.zeros((len(eps), run_length))

for l in range(len(eps)):
    for i in range(num_runs):
        trueValue = np.random.normal(0, 1, k)
        bestAction = np.argmax(trueValue)
        bestAvReward += trueValue[bestAction]

        estimatedValue = np.zeros(k)
        count = np.zeros(k)

        for j in range(run_length):
            # Randomly choose to explore with probability e or exploit with probability 1-e
            exploit = np.random.choice(2, p = [eps[l], 1 - eps[l]])
            if exploit:
                action = np.argmax(estimatedValue)
            else:
                action = np.random.randint(k)
            reward = np.random.normal(trueValue[action], 1)
            count[action] += 1
            estimatedValue[action] += (reward - estimatedValue[action])/count[action]

            rewardAtStep[l, j] += reward
            if action == bestAction:
                numOptimalAtStep[l, j] += 1

bestAvReward /= num_runs
rewardAtStep /= num_runs
numOptimalAtStep /= (num_runs / 100)

print("Best Possible Return: " + str(bestAvReward))

plt.plot(range(run_length), rewardAtStep[0,:], label = "Greedy")
plt.plot(range(run_length), rewardAtStep[1,:], label = "e = 0.1")
plt.plot(range(run_length), rewardAtStep[2,:], label = "e = 0.01")
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.show()

plt.plot(range(run_length), numOptimalAtStep[0,:], label = "Greedy")
plt.plot(range(run_length), numOptimalAtStep[1,:], label = "e = 0.1")
plt.plot(range(run_length), numOptimalAtStep[2,:], label = "e = 0.01")
plt.xlabel('Steps')
plt.ylabel('% Optimal Reward')
plt.show()
