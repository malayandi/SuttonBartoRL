''' Fig 2.3: Compares the performance of greedy algorithms when using optmistic
initial values as opposed to realistic initial values.

Uses constant step-size parameter. '''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Number of arms of bandit
k = 10

# Step-size parameter
alpha = 0.1

# Optimistic Initial Value
init = 5

# Probabilities with which we explore instead of exploit
eps = [0, 0.1]

num_runs = 1000
run_length = 1000

rewardAtStep = np.zeros((2, run_length))
numOptimalAtStep = np.zeros((2, run_length))

for r in range(2):
    for i in range(num_runs):
        trueValue = np.random.normal(0, 1, k)
        bestAction = np.argmax(trueValue)

        if r == 1:
            estimatedValue = np.zeros(k)
        else:
            estimatedValue = np.ones(k) * init
        count = np.zeros(k)

        for j in range(run_length):
            # Randomly choose to explore with probability e or exploit with probability 1-e
            exploit = np.random.choice(2, p = [eps[r], 1 - eps[r]])
            if exploit:
                action = np.argmax(estimatedValue)
            else:
                action = np.random.randint(k)
            reward = np.random.normal(trueValue[action], 1)
            count[action] += 1
            estimatedValue[action] += ((reward - estimatedValue[action]) * alpha)

            rewardAtStep[r, j] += reward
            if action == bestAction:
                numOptimalAtStep[r, j] += 1

rewardAtStep /= num_runs
numOptimalAtStep /= (num_runs / 100)

plt.plot(range(run_length), numOptimalAtStep[0,:], label = "Optimistic, Greedy (Q1 = " + str(init) + ")")
plt.plot(range(run_length), numOptimalAtStep[1,:], label = "Realistic, e-Greedy (Q1 = 0, e = 0.1)")
plt.xlabel('Steps')
plt.ylabel('% Optimal Reward')
plt.legend()
plt.show()
