''' Fig 2.4: Compares the performance of the UCB action selection algorithm with
an e-Greedy algorithm. '''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Number of arms of bandit
k = 10

# UCB parameter
c = 2

# Probabilities with which we explore instead of exploit
eps = 0.1

num_runs = 1000
run_length = 1000

rewardAtStep = np.zeros((2, run_length))
numOptimalAtStep = np.zeros((2, run_length))

# e-Greedy for r = 0; UCB for r = 1
for r in range(2):
    for i in range(num_runs):
        trueValue = np.random.normal(0, 1, k)
        bestAction = np.argmax(trueValue)

        estimatedValue = np.zeros(k)
        count = np.zeros(k)

        for j in range(run_length):
            if not r:
                exploit = np.random.choice(2, p = [eps, 1 - eps])
                if exploit:
                    action = np.argmax(estimatedValue)
                else:
                    action = np.random.randint(k)
            else:
                if 0 in count:
                    action = np.where(count == 0)[0][0]
                else:
                    ucbEst = estimatedValue + c * np.sqrt(np.log(j)/count)
                    action = np.argmax(ucbEst)
            reward = np.random.normal(trueValue[action], 1)
            count[action] += 1
            estimatedValue[action] += (reward - estimatedValue[action])/count[action]

            rewardAtStep[r, j] += reward
            if action == bestAction:
                numOptimalAtStep[r, j] += 1

rewardAtStep /= num_runs
numOptimalAtStep /= (num_runs / 100)

plt.plot(range(run_length), rewardAtStep[0,:], label = "e-Greedy (e = 0.1)")
plt.plot(range(run_length), rewardAtStep[1,:], label = "UCB (c = 2)")
plt.xlabel('Steps')
plt.ylabel('Avg Reward')
plt.legend(loc = 4)
plt.show()
