# -*- coding: UTF-8 -*-

from ZUI_MDP import GridWorld
import numpy as np


if __name__ == "__main__":
    name = "5x5"
    algorithm_rtol = 1e-4  # relative tolerance for comparing two floats
    algorithm_atol = 1e-08  # absolute tolerance for comparing two floats

    changes = []
    old_policy = np.zeros((26,), dtype=int)

    c1 = np.linspace(0, 0.9, 100)
    c2 = np.linspace(1, 9, 10)
    c = np.concatenate((c1, c2), axis=0)
    for i in c:
        gw = GridWorld.get_world(name, action_cost=i)
        Q = gw.policy_iteration(rtol=algorithm_rtol, atol=algorithm_atol)
        policy = gw.Q2policy(Q)

        # compare old and current policy
        policy_compare = policy == old_policy
        if not policy_compare.all():
            changes.append(i)
        old_policy = policy

    print('Costs in which policy change occurs: ', [float('%0.2f' % i) for i in changes])

    cost = 2
    gw = GridWorld.get_world(name, action_cost=cost)
    Q = gw.policy_iteration(rtol=algorithm_rtol, atol=algorithm_atol)
    policy = gw.Q2policy(Q)
    V = gw.Q2Vbypolicy(Q, policy)
    gw.plot(V=V, policy=policy)
