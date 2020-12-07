# -*- coding: UTF-8 -*-

from ZUI_MDP import GridWorld
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    name = "6x12"
    algorithm_rtol = 1e-4  # relative tolerance for comparing two floats
    algorithm_atol = 1e-08  # absolute tolerance for comparing two floats

    # intervals
    c1 = np.linspace(0, 0.9, 10)
    c2 = np.linspace(1, 5, 5)
    c = np.concatenate((c1, c2), axis=0)
    p = np.linspace(0, 0.9, 10)
    data = np.zeros((len(c), len(p)))

    # given state (I also used 5 and 67)
    state = 58

    # compute value for each value of cost and action probability
    for i in range(len(c)):
        for j in range(len(p)):
            gw = GridWorld.get_world(name, action_proba=p[j], action_cost=c[i])
            Q = gw.policy_iteration(rtol=algorithm_rtol, atol=algorithm_atol)
            policy = gw.Q2policy(Q)
            V = gw.Q2Vbypolicy(Q, policy)
            data[i][j] = V[state]

    # formatting values for printing
    p_list = ["%0.1f" % i for i in p.tolist()]
    c_list = ["%0.1f" % i for i in c.tolist()]

    # heatmap
    ax = sns.heatmap(data, annot=False, cmap="coolwarm", xticklabels=p_list, yticklabels=c_list)

    plt.xlabel('action cost')
    plt.ylabel('action probability')
    plt.show()
