# -*- coding: UTF-8 -*-

from ZUI_MDP import GridWorld
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    name = "3x4"
    algorithm_rtol = 1e-4  # relative tolerance for comparing two floats
    algorithm_atol = 1e-08  # absolute tolerance for comparing two floats

    valueV = []
    changes = []
    old_policy = np.zeros((13,), dtype=int)

    p = np.linspace(0, 1, 100)
    for i in p:
        gw = GridWorld.get_world(name, action_proba=i)
        # Q = gw.value_iteration(rtol=algorithm_rtol, atol=algorithm_atol)
        Q = gw.policy_iteration(rtol=algorithm_rtol, atol=algorithm_atol)
        policy = gw.Q2policy(Q)
        V = gw.Q2Vbypolicy(Q, policy)

        valueV.append(V.tolist())

        # compare old and current policy
        policy_compare = policy == old_policy
        if not policy_compare.all():
            changes.append(i)
        old_policy = policy

    valueV = np.transpose(valueV)

    # p for which there occurs a change in the optimal policy
    for ch in changes:
        plt.axvline(ch, 0, 1, c='r', ls='--', lw=0.7)

    print('Action probabilities in which policy change occurs: ', [float('%0.2f' % i) for i in changes])

    # plot graph for each state
    states = [0, 3, 6, 8, 9, 10, 11]
    for state in states:
        plt.plot(p, valueV[state], label=str(state))
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.xlabel('p')
    plt.ylabel('value')
    plt.legend()
    plt.show()
