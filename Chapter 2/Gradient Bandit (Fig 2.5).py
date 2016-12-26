''' Fig 2.5: Compares the performance of various gradient bandit algorithms. '''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Number of arms of bandit
k = 10

# Parameters
param = [(0.1, "with"), (0.1, "without"), (0.4, "with"), (0.4, "without")]

num_runs = 100
run_length = 1000

rewardAtStep = np.zeros((4, run_length))
numOptimalAtStep = np.zeros((4, run_length))

def computeProbability(pref):
    ''' Returns a nd-array containing the action probability of each of the 10
    actions using a soft-max distribution parameterised by the entries of PREF.
    '''
    return np.exp(pref)/np.sum(np.exp(pref))

for p in range(len(param)):
    for i in range(num_runs):
        trueValue = np.random.normal(4, 1, k)
        bestAction = np.argmax(trueValue)

        avReward = 0
        pref = np.zeros(k)

        currParam = param[p]
        alpha = currParam[0]
        baseline = (currParam[1] == "with")

        for j in range(run_length):
            prob = computeProbability(pref)
            action = np.random.choice(k, p = prob)
            reward = np.random.normal(trueValue[action], 1)

            pref[action] += alpha * (reward - avReward) * (1 - prob[action])
            for i in range(k):
                if i != action:
                    pref[i] -= alpha * (reward - avReward) * (prob[i])

            if baseline:
                avReward += (reward - avReward)/(j + 1)
            rewardAtStep[p, j] += reward
            if action == bestAction:
                numOptimalAtStep[p, j] += 1

    print(param[p])
    print(avReward)

rewardAtStep /= num_runs
numOptimalAtStep /= (num_runs / 100)

plt.plot(range(run_length), numOptimalAtStep[0,:], label = "alpha = 0.1, with baseline")
plt.plot(range(run_length), numOptimalAtStep[1,:], label = "alpha = 0.1, without baseline")
plt.plot(range(run_length), numOptimalAtStep[2,:], label = "alpha = 0.4, with baseline")
plt.plot(range(run_length), numOptimalAtStep[3,:], label = "alpha = 0.4, without baseline")
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend(loc = 4)
plt.show()
