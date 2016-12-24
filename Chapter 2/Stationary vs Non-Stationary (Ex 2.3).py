''' EX 2.3: Runs the Ten-Armed Bandit with the goal of demonstrating the
difficulties that sample average methods have for nonstationary problems. '''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Number of arms of bandit
k = 10

# Step-size parameter
alpha = 0.1

# Probabilities with which we explore instead of exploit
eps = 0.1

num_runs = 100
run_length = 2000

rewardAtStep = np.zeros((2, run_length))
numOptimalAtStep = np.zeros((2, run_length))

for s in range(2):
    for i in range(num_runs):
        trueValue = np.random.normal(0, 1, k)
        bestAction = np.argmax(trueValue)

        estimatedValue = np.zeros(k)
        count = np.zeros(k)

        for j in range(run_length):
            # Randomly choose to explore with probability e or exploit with probability 1-e
            if j and j % 50 == 0:
                trueValue += np.random.normal(0, 1, k)
                bestAction = np.argmax(trueValue)
            exploit = np.random.choice(2, p = [eps, 1 - eps])
            if exploit:
                action = np.argmax(estimatedValue)
            else:
                action = np.random.randint(k)
            reward = np.random.normal(trueValue[action], 1)
            count[action] += 1
            if s == 0:
                estimatedValue[action] += (reward - estimatedValue[action])/count[action]
            else:
                estimatedValue[action] += ((reward - estimatedValue[action]) * alpha)

            rewardAtStep[s, j] += reward
            if action == bestAction:
                numOptimalAtStep[s, j] += 1

rewardAtStep /= num_runs
numOptimalAtStep /= (num_runs / 100)

plt.plot(range(run_length), rewardAtStep[0,:], label = "Stationary")
plt.plot(range(run_length), rewardAtStep[1,:], label = "Non-Stationary")
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.show()

plt.plot(range(run_length), numOptimalAtStep[0,:], label = "Stationary")
plt.plot(range(run_length), numOptimalAtStep[1,:], label = "Non-Stationary")
plt.xlabel('Steps')
plt.ylabel('% Optimal Reward')
plt.legend()
plt.show()
