''' Fig 2.6: Compares the performance of four different algorithms at varying
parameter settings: e-Greedy, UCB, gradient bandit, greedy with optimistic
initialisation. '''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def computeProbability(pref):
    ''' Returns a nd-array containing the action probability of each of the 10
    actions using a soft-max distribution parameterised by the entries of PREF.
    '''
    return np.exp(pref)/np.sum(np.exp(pref))


def tenArmedBandit(alg, param, k = 10, run_length = 1000, num_runs = 250):
    ''' Computes the average reward in 1000 steps of the Ten-Armed Bandit
    problem using the algorithm, ALG, across various parameter settings, as
    given in PARAM.

    Returns an array of the same length as param, containing the average reward
    over 10 steps.

    @ALG: Options - eGreedy, gradient, UCB, optGreedy'''
    averages = np.zeros(len(param))
    for p in range(len(param)):
        totalReward = 0
        for i in range(num_runs):
            trueValue = np.random.normal(0, 1, k)
            avReward = 0

            if alg == "gradient":
                pref = np.zeros(k)
            elif alg == "optGreedy":
                estimatedValue = np.ones(k) * param[p]
            else:
                estimatedValue = np.zeros(k)

            count = np.zeros(k)

            for j in range(run_length):
                # Picking action
                if alg == "eGreedy":
                    exploit = np.random.choice(2, p = [param[p], 1 - param[p]])
                    if exploit:
                        action = np.argmax(estimatedValue)
                    else:
                        action = np.random.randint(k)
                elif alg == "optGreedy":
                    action = np.argmax(estimatedValue)
                elif alg == "UCB":
                    if 0 in count:
                        action = np.where(count == 0)[0][0]
                    else:
                        ucbEst = estimatedValue + param[p] * np.sqrt(np.log(j)/count)
                        action = np.argmax(ucbEst)
                elif alg == "gradient":
                    prob = computeProbability(pref)
                    action = np.random.choice(k, p = prob)

                # Getting reward
                reward = np.random.normal(trueValue[action], 1)
                count[action] += 1

                # Updating estimates
                if alg == "gradient":
                    pref[action] += param[p] * (reward - avReward) * (1 - prob[action])
                    for i in range(k):
                        if i != action:
                            pref[i] -= param[p] * (reward - avReward) * (prob[i])
                elif alg == "optGreedy":
                    estimatedValue[action] += (reward - estimatedValue[action]) * 0.1
                else:
                    estimatedValue[action] += (reward - estimatedValue[action])/count[action]

                avReward += (reward - avReward)/(j + 1)

            totalReward += avReward

        averages[p] = totalReward/num_runs

    return averages

eGreedy = tenArmedBandit("eGreedy", [1/128, 1/64, 1/32, 1/16, 1/8, 1/4])
gradient = tenArmedBandit("gradient", [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2])
UCB = tenArmedBandit("UCB", [1/16, 1/8, 1/4, 1/2, 1, 2, 4])
optGreedy = tenArmedBandit("optGreedy", [1/4, 1/2, 1, 2, 4])

ax = plt.axes()
ax.set_xscale('log', basex=2)
plt.ylim(1, 1.5)
plt.plot([1/128, 1/64, 1/32, 1/16, 1/8, 1/4], eGreedy, label = "e-Greedy")
plt.plot([1/32, 1/16, 1/8, 1/4, 1/2, 1, 2], gradient, label = "gradient bandit")
plt.plot([1/16, 1/8, 1/4, 1/2, 1, 2, 4], UCB, label = "UCB")
plt.plot([1/4, 1/2, 1, 2, 4], optGreedy, label = "greedy with optimistic initialisation")
plt.xlabel('Parameter')
plt.ylabel('Average Reward')
plt.legend(loc = 4)
plt.show()
