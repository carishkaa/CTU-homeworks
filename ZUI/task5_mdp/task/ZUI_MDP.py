# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import Tuple
import matplotlib
import numpy as np
import warnings
from abc import ABC


class MDP(ABC):
    def Q_from_V(self, V: np.ndarray) -> np.ndarray:
        """ Computes `Q` from `V`.

            Estimates the values of individual action in individual states
            i.e. `Q`) for given valuation of states `V`.
            `` Q[s,a] = rewards[s] + \\sum_{s'} transition_proba[s,a,s'] * discount_factor * V[s']  ``
            

            Parameters
            ----------
            V : :obj:`ndarray`
                A valuation of individual states.

            Returns
            -------
            :obj:`ndarray`
                A valuation of individual action in individual states `Q`.

        """
        Q = np.zeros((self.n_states + 1, self.n_actions))

        for action in self.actions:
            # ----------------------
            Q[:, action] = self.rewards + self.discount_factor * np.matmul(self.transition_proba[:, action, :], V)
            # ----------------------
        return Q

    def Q2V(self, Q: np.ndarray) -> np.ndarray:
        """ Computes `V`  from `Q`.

            Computes the valuation `V` from `Q` assuming optimal decision is made at each state.   
            `` V[s] = \\max_{a} Q[s,a] ``


            Parameters
            ----------
            Q : :obj:`ndarray`
                A valuation of individual states and actions.

            Returns
            -------
            :obj:`ndarray`
                A valuation of individual action for optimal policy.

        """
        return np.amax(Q, axis=1)

    def Q2Vbypolicy(self, Q: np.ndarray, policy: np.ndarray) -> np.ndarray:
        """ Computes `V`  from `Q` for given policy.

            Computes the valuation `V` from `Q` for given policy describing optimal action in each state.   
            `` V[s] = Q[s,policy[s]] ``


            Parameters
            ----------
            Q : :obj:`ndarray`
                A valuation of individual states and actions.
                
            policy : :obj:`ndarray` of `int`
                Policy describing optimal action in each state.

            Returns
            -------
            :obj:`ndarray`
                A valuation of individual action for optimal policy.

        """
        # print(np.choose(policy, Q.T))
        trQ = np.transpose(Q)
        return np.choose(policy, trQ)

    def Q2policy(self, Q: np.ndarray) -> np.ndarray:
        """ Computes optimal policy from `Q`.

            Computes the optimal policy from `Q`.   
            `` policy[s] = \\argmax_{a} Q[s,a] ``


            Parameters
            ----------
            Q : :obj:`ndarray`
                A valuation of individual states and actions.

            Returns
            -------
            :obj:`ndarray`
                The optimal policy.
        """
        return np.argmax(Q, axis=1)

    def evaluate_policy(
        self,
        policy: np.ndarray,
        init_V: np.ndarray = None,
        max_iter: int = int(1e3),
        rtol: float = 1e-06,
        atol: float = 1e-08,
    ):
        """ Evaluate policy.

            Evaluate policy by iteratively recomputing `Q` and `V` using given
            policy until convergence.
            

            Parameters
            ----------
            policy : :obj:`ndarray`
                A policy to be evaluated.
                
            init_V : :obj:`ndarray`, optional
                An intial valuation of states.
                
            max_iter : int, optional
                Maximum number of iterations.
                
            rtol : float, optional
                Relative tolerance for checking convergence.
                
            atol : float, optional
                Absolute tolerance for checking convergence.

            Returns
            -------
            :obj:`ndarray`
                The valuation `V`.
                
            Notes
            -----
            Convergence is checked using `numpy.allclose`.
        """
        old_V = self.rewards if init_V is None else init_V
        for i in range(max_iter):
            # ----------------------
            Q = self.Q_from_V(old_V)
            V = self.Q2Vbypolicy(Q, policy)
            # ----------------------
            if np.allclose(old_V, V, rtol=rtol, atol=atol):
                return V
            old_V = V
        warnings.warn(
            f"Maximum number of iterations ({max_iter}) has been exceeded."
            f"The iterative evaluation of a policy might have not converged"
        )
        return V

    def value_iteration(
        self, init_V: np.ndarray = None, max_iter: int = int(1e4), rtol: float = 1e-06, atol: float = 1e-08
    ):
        """ Estimates the valuation of individual states and action using
            value iteration.

            Estimates the valuation of individual states and action using
            value iteration - i.e. by iteratively recomputing `Q` and `V`
            for optimal policy in each iteration.


            Parameters
            ----------

            init_V : :obj:`ndarray`, optional
                An intial valuation of states.

            max_iter : int, optional
                Maximum number of iterations.

            rtol : float, optional
                Relative tolerance for checking convergence.

            atol : float, optional
                Absolute tolerance for checking convergence.

            Returns
            -------
            :obj:`ndarray`
                The valuation `Q`.

            Notes
            -----
            Convergence is checked using `numpy.allclose`.
        """
        old_V = self.rewards if init_V is None else init_V
        for i in range(max_iter):
            # ----------------------
            Q = self.Q_from_V(old_V)
            V = self.Q2V(Q)
            # ----------------------
            if np.allclose(old_V, V, rtol=rtol, atol=atol):
                return Q
            old_V = V
        warnings.warn(
            f"Maximum number of iterations ({max_iter}) has been exceeded."
            f" Value iteration might have not converged."
        )
        return Q

    def policy_iteration(
        self,
        init_policy: np.ndarray = None,
        max_iter: int = int(1e4),
        rtol: float = 1e-08,
        atol: float = 1e-08,
        max_iter_eval_iteration: int = int(1e4),
    ):
        """ Estimates the valuation of individual states and action using
            policy iteration.

            Estimates the valuation of individual states and action using
            policy iteration.


            Parameters
            ----------

            init_V : :obj:`ndarray`, optional
                An intial valuation of states.

            max_iter : int, optional
                Maximum number of iterations.

            rtol : float, optional
                Relative tolerance for checking convergence.

            atol : float, optional
                Absolute tolerance for checking convergence.
                
            max_iter_eval_iteration : int, optional
                The maximum number of iterations for evaluating the policy in each iteration.

            Returns
            -------
            :obj:`ndarray`
                The valuation `Q`.

            Notes
            -----
            Convergence is checked using `numpy.allclose`.
        """
        Q = np.zeros((self.n_states + 1, self.n_actions))
        old_Q = Q
        if init_policy is None:
            policy = np.zeros((self.n_states + 1,), dtype=int)
        else:
            policy = init_policy
        old_policy = policy
        for i in range(max_iter):
            # ----------------------
            # FIXME:
            V = self.evaluate_policy(old_policy, None, max_iter_eval_iteration, rtol, atol)
            Q = self.Q_from_V(V)
            policy = self.Q2policy(Q)
            # ----------------------
            if np.allclose(old_Q, Q, rtol=rtol, atol=atol) or np.array_equal(policy, old_policy):
                return Q
            old_Q = Q
            old_policy = policy
        warnings.warn(
            f"Maximum number of iterations ({max_iter}) has been exceeded."
            f"Policy iteration might have not converged."
        )
        return Q


class GridWorld(MDP):
    """ GridWorld MDP.

        The GridWorld describes and MarkovDecisionProblem (MDP) as a rectangular
        grid of states with actions that allows go to the 4 neighbouring cells.
        
        Possible actions are NORTH, EAST, SOUTH, and WEST.

        Attributes
        ----------
        NORTH : int
            Code of action go to north.
        EAST : int
            Code of action go to east.
        SOUTH : int
            Code of action go to south.
        WEST : int
            Code of action go to west.
        actions : :obj:`dictionary` with key `int` and values :obj:`ndarray`
            Describes coordinate changes for each action.
        n_actions : int
            Number of available actions.
        mismatched_actions : :obj:`dictionary` with key `int` and values :obj:`tuple` of `int`
            Describes the possibile erroneous actions an agent migh make instaed of the action given by the key.
        AVAILABLE_WORLDS : :obj:`list` of :obj:`str`
            List of available predefined GridWorlds.
        """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    actions = {NORTH: np.array((-1, 0)), EAST: np.array((0, 1)), SOUTH: np.array((1, 0)), WEST: np.array((0, -1))}
    n_actions = len(actions)
    mismatched_actions = {NORTH: (EAST, WEST), EAST: (NORTH, SOUTH), SOUTH: (EAST, WEST), WEST: (NORTH, SOUTH)}

    AVAILABLE_WORLDS = ["2x2", "3x3", "3x4", "4x4", "5x5", "6x8", "6x12"]

    @staticmethod
    def get_world(name: str, action_proba: float = None, action_cost: float = None) -> MDP:
        """Factory function returning one of the predefined GridWorlds.

        Returns one of the predefined GridWorlds. The names of predefined
        GridWorlds are in `AVAILABLE_WORLDS`.

        Parameters
        ----------
        name : str
            The name of the predefined instance.
        action_proba : float, optional
            The probability of executing intended action overriding the
            predefined value for easier experimenting.
        action_cost : float, optional
            The cost of executing action (as described in `__init__` method )
            overriding the predefined value for easier experimenting.
        Returns
        -------
        GridWorld
            A predefined GridWorld instance.

        """

        if name == "2x2":
            ab = 0.8 if action_proba is None else action_proba
            ac = 1 / 25 if action_cost is None else action_cost
            n_rows = 2
            n_columns = 2
            discount_factor = 1
            obstacles = np.zeros((n_rows, n_columns))
            grid_rewards = np.zeros((n_rows, n_columns))
            grid_rewards[0, n_columns - 1] = 1
            grid_rewards[1, n_columns - 1] = -1

            terminals = np.zeros((n_rows, n_columns))
            terminals[0, n_columns - 1] = 1
            terminals[1, n_columns - 1] = 1

            gw = GridWorld(
                n_rows,
                n_columns,
                obstacles,
                terminals,
                grid_rewards,
                discount_factor=discount_factor,
                action_proba=ab,
                action_cost=ac,
            )
            return gw

        elif name == "3x3":
            ab = 0.6 if action_proba is None else action_proba
            ac = 1 / 25 if action_cost is None else action_cost
            n_rows = 3
            n_columns = 3
            discount_factor = 1
            obstacles = np.zeros((n_rows, n_columns))
            grid_rewards = np.zeros((n_rows, n_columns))
            grid_rewards[0, n_columns - 1] = 1
            grid_rewards[1, n_columns - 1] = -1

            terminals = np.zeros((n_rows, n_columns))
            terminals[0, n_columns - 1] = 1
            terminals[1, n_columns - 1] = 1

            return GridWorld(
                n_rows,
                n_columns,
                obstacles,
                terminals,
                grid_rewards,
                discount_factor=discount_factor,
                action_proba=ab,
                action_cost=ac,
            )

        elif name == "3x4":
            ab = 0.8 if action_proba is None else action_proba
            ac = 1 / 25 if action_cost is None else action_cost
            n_rows = 3
            n_columns = 4
            discount_factor = 1
            obstacles = np.zeros((n_rows, n_columns))
            obstacles[1, 1] = 1
            grid_rewards = np.zeros((n_rows, n_columns))
            grid_rewards[0, n_columns - 1] = 1
            grid_rewards[1, n_columns - 1] = -1

            terminals = np.zeros((n_rows, n_columns))
            terminals[0, n_columns - 1] = 1
            terminals[1, n_columns - 1] = 1

            return GridWorld(
                n_rows,
                n_columns,
                obstacles,
                terminals,
                grid_rewards,
                discount_factor=discount_factor,
                action_proba=ab,
                action_cost=ac,
            )

        elif name == "4x4":
            ab = 0.5 if action_proba is None else action_proba
            ac = 2 / 25 if action_cost is None else action_cost
            n_rows = 4
            n_columns = 4
            discount_factor = 0.8
            obstacles = np.zeros((n_rows, n_columns))
            obstacles[1, 1] = 1
            obstacles[2, 2] = 1
            grid_rewards = np.zeros((n_rows, n_columns))
            grid_rewards[0, n_columns - 1] = 1
            grid_rewards[1, n_columns - 1] = -1

            terminals = np.zeros((n_rows, n_columns))
            terminals[0, n_columns - 1] = 1
            terminals[1, n_columns - 1] = 1

            return GridWorld(
                n_rows,
                n_columns,
                obstacles,
                terminals,
                grid_rewards,
                discount_factor=discount_factor,
                action_proba=ab,
                action_cost=ac,
            )

        elif name == "5x5":
            ab = 0.95 if action_proba is None else action_proba
            ac = 1 / 50 if action_cost is None else action_cost
            n_rows = 5
            n_columns = 5
            discount_factor = 0.99
            obstacles = np.zeros((n_rows, n_columns))
            obstacles[1, 0] = 1
            obstacles[1, 1] = 1
            obstacles[2, 2] = 1

            grid_rewards = np.zeros((n_rows, n_columns))
            grid_rewards[0, 0] = 2
            grid_rewards[1, 2] = -1
            grid_rewards[0, n_columns - 1] = 1
            grid_rewards[1, n_columns - 1] = -1
            grid_rewards[4, 2] = -1

            terminals = np.zeros((n_rows, n_columns))
            terminals[0, 0] = 1
            terminals[1, 2] = 1
            terminals[0, n_columns - 1] = 1
            terminals[1, n_columns - 1] = 1
            terminals[4, 2] = 1

            return GridWorld(
                n_rows,
                n_columns,
                obstacles,
                terminals,
                grid_rewards,
                discount_factor=discount_factor,
                action_proba=ab,
                action_cost=ac,
            )

        elif name == "6x8":
            ab = 0.95 if action_proba is None else action_proba
            ac = 1 / 50 if action_cost is None else action_cost
            n_rows = 6
            n_columns = 8
            discount_factor = 0.99
            obstacles = np.zeros((n_rows, n_columns))
            obstacles[1:5, 1] = 1
            obstacles[1, 1:6] = 1
            obstacles[1:5, 5] = 1
            obstacles[4, 3] = 1
            obstacles[1, 6] = 1
            obstacles[3, 7] = 1

            grid_rewards = np.zeros((n_rows, n_columns))
            grid_rewards[3, 3] = 2
            grid_rewards[4, 4] = -1

            terminals = np.zeros((n_rows, n_columns))
            terminals[3, 3] = 1
            terminals[4, 4] = 1

            return GridWorld(
                n_rows,
                n_columns,
                obstacles,
                terminals,
                grid_rewards,
                discount_factor=discount_factor,
                action_proba=ab,
                action_cost=ac,
            )

        elif name == "6x12":
            ab = 0.7 if action_proba is None else action_proba
            ac = 0.01 if action_cost is None else action_cost
            n_rows = 6
            n_columns = 12
            discount_factor = 0.99
            obstacles = np.zeros((n_rows, n_columns))
            obstacles[4, 1:10] = 1
            obstacles[1:4, 1] = 1
            obstacles[0:3, 3] = 1
            obstacles[1:4, 5] = 1
            obstacles[0:3, 7] = 1
            obstacles[1:4, 9] = 1

            grid_rewards = np.zeros((n_rows, n_columns))
            grid_rewards[5, 8] = 2
            grid_rewards[1:5, 11] = -1

            terminals = np.zeros((n_rows, n_columns))
            terminals[5, 8] = 1
            terminals[1:5, 11] = 1

            return GridWorld(
                n_rows,
                n_columns,
                obstacles,
                terminals,
                grid_rewards,
                discount_factor=discount_factor,
                action_proba=ab,
                action_cost=ac,
            )

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        obstacles: np.ndarray,
        terminals: np.ndarray,
        grid_rewards: np.ndarray,
        discount_factor: float = 1,
        action_proba: float = 0.8,
        action_cost: float = 0.04,
    ):
        """ Grid world __init__ method.
    
            Initialize GridWorld instance describing MDP problem on
            rectangular grid with non deterministic actions and
            obstacles.
    
            Parameters
            ----------
            n_rows : int
                The number of rows of the grid.
            
            n_columns : int
                The number of columns of the grid.
            
            obstacles : ndarray of `int`
                The `n_rows`x`n_columns` numpy array describing the obstacles on the grid.
                An obstacle is at position `(i,j)` iff `obstacles[i,j] == 1`.
                
            terminals : ndarray of `int`
                The `n_rows`x`n_columns` numpy array describing the terminal states on the grid.
                A terminal state is at position `(i,j)` iff `terminals[i,j] == 1`.
                
            grid_rewards : ndarray of `float` or `int`
                The `n_rows`x`n_columns` numpy array describing the rewards for reaching the state on the grid.
                The reward for reaching the state does not include the action reward.
                Overrides the action cost for reaching given state.
                
            discount_factor : float, optional
                The discount factor :math:`\\in [0,1]` representing the
                difference in importance between future rewards and present rewards.
                
            action_proba : float, optional
                The probability of executing the intended action :math:`\\in [0,1]`.
                The other actions actions are executed with probability `(1 - action_proba)/n`
                where `n` is the number of actions that might be executed instead
                of the intended action (which is usually 2).
                
            action_cost : float, optional
                The cost for making any action, usually negative and representing cost of making. 
                Can be locally overridden by setting nonzero `grid_rewards`.
                     
            Returns
            -------
            GridWorld
                A predefined GridWorld instance.
    
        """
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.obstacles = obstacles
        self.terminals = terminals
        self.grid_rewards = grid_rewards

        self.discount_factor = discount_factor
        self.action_proba = action_proba
        self.action_cost = action_cost
        self.n_states = n_rows * n_columns  # number of states without terminal sink state
        self.transition_proba = None
        self._get_transition_proba()

        self.rewards = np.ones(self.n_states + 1) * -self.action_cost
        for ind_reward in np.flatnonzero(self.grid_rewards):
            self.rewards[ind_reward] = self.grid_rewards.flat[ind_reward]
        self.rewards[self.n_states] = 0

    def _state2coord(self, state: int) -> Tuple[int, int]:
        """ Convert state number to grid coordinate.

            Converts state number to grid coordinate (if applicable - the artificial
            sink state has no grid coordinate and such conversion results in error).
        
            Parameters
            ----------
            state : int
                The state number.
        
            Returns
            -------
            (int, int)
                A coordinate on the grid.

        """
        return np.unravel_index(state, (self.n_rows, self.n_columns))

    def _coord2state(self, coord: Tuple[int, int]) -> int:
        """ Convert grid coordinate to state number.

            Converts state number to grid coordinate (if applicable - the artificial
            sink state has no grid coordinate and such conversion results in error).

            Parameters
            ----------
            coord : (int, int)
                A coordinate on the grid.

            Returns
            -------
            int
                A state number.

        """
        return np.ravel_multi_index(coord, (self.n_rows, self.n_columns))

    def _is_on_grid(self, coord: Tuple[int, int]) -> bool:
        """ Checks whether the coordinate is on the grid.

            Checks whether the state given by the coordinate is whithin the boundary of the grid.
            Does not check for obstacles.

            Parameters
            ----------
            coord : (int, int)
                A coordinate on the grid.

            Returns
            -------
            bool
                True if is on the grid.

        """
        r, c = coord
        return 0 <= r < self.n_rows and 0 <= c < self.n_columns

    def _is_obstacle(self, coord: Tuple[int, int]) -> bool:
        """ Checks whether there is an obstacle on given coordinate.

            Checks whether there is an obstacle on given coordinate.
            The method fails if the coordinate is not on the grid.

            Parameters
            ----------
            coord : (int, int)
                A coordinate on the grid.

            Returns
            -------
            bool
                True if there is obstacle.

        """
        r, c = coord
        return self.obstacles[r, c] == 1

    def _is_terminal(self, coord: Tuple[int, int]) -> bool:
        """ Checks whether there is a terminal state on given coordinate.

            Checks whether there is a terminal state on given coordinate.
            The method fails if the coordinate is not on the grid.

            Parameters
            ----------
            coord : (int, int)
                A coordinate on the grid.

            Returns
            -------
            bool
                True if there is a terminal state.

        """
        return self.terminals[coord] == 1

    def _get_transition_proba(self):
        """ Set the transition probabilities.

            Set `transition_proba` to an :obj:`ndarray` of shape `(self.n_states + 1, self.n_actions, self.n_states + 1)` of 
            transition probabilities `transition_proba[s,a,s'] = P(s'|a,s)`. 
        """
        self.transition_proba = np.zeros(
            (self.n_states + 1, self.n_actions, self.n_states + 1)
        )  # terminal sink state added
        for state in range(self.n_states):
            state_coord = self._state2coord(state)
            if self._is_terminal(state_coord):
                # if the state is terminal, go to the single terminal sink with p = 1
                self.transition_proba[state, :, self.n_states] = 1
            else:
                for action in self.actions:
                    # compute the transition probabilities of intended actions
                    next_state_coord = state_coord + self.actions[action]
                    if not self._is_on_grid(next_state_coord) or self._is_obstacle(next_state_coord):
                        self.transition_proba[state, action, state] += self.action_proba
                    else:
                        next_state = self._coord2state(next_state_coord)
                        self.transition_proba[state, action, next_state] += self.action_proba

                    # compute the transition probabilities of mistaken actions
                    # the probability of mistaken action is the same for all possible mistakes for given action
                    for mismatched_action in self.mismatched_actions[action]:
                        mistake_proba = (1 - self.action_proba) / len(self.mismatched_actions[action])
                        next_state_coord = state_coord + self.actions[mismatched_action]
                        if not self._is_on_grid(next_state_coord) or self._is_obstacle(next_state_coord):
                            self.transition_proba[state, action, state] += mistake_proba
                        else:
                            next_state = self._coord2state(next_state_coord)
                            self.transition_proba[state, action, next_state] += mistake_proba
        # the terminal state cannot be left -> with any action, an agent stays in the state with p = 1
        self.transition_proba[self.n_states, :, self.n_states] = 1

    def plot(self, V: np.ndarray = None, policy: np.ndarray = None):
        """ Visualize the GridWorld.

            Visualize the GridWorld rewards (without the `action_cost`).
            If `V` is set, then it visualize the computed values of individual states.
            If `policy` is set then the policy (arrows) are shown instead of the state values.
            
            Parameters
            ----------
            V : :obj:`ndarray`, optional
                The value of states. Either with the sink state (i.e as returned by e.g. `Q2V`) or without.
                
            policy : :obj:`ndarray`, optional
                The optimal policy for individual states (including the sink state).
        """
        if V is None:
            data = self.grid_rewards
        else:
            if len(V) == self.n_rows * self.n_columns + 1:
                V = V[:-1]
            data = V.reshape(self.n_rows, self.n_columns)
        policy_symbols = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        mask = np.zeros_like(data, dtype=bool)
        mask[self.obstacles == 1] = True
        # to center the heatmap around zero
        maxval = max(np.abs(np.min(data)), np.abs(np.max(data)))
        ax = sns.heatmap(
            data, annot=False, mask=mask, fmt=".3f", square=1, linewidth=1.0, cmap="coolwarm", vmin=-maxval, vmax=maxval
        )
        for i, j in zip(*np.where(self.terminals == 1)):
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=3))
        if policy is not None:
            for t, pol in zip(ax.texts, policy[:-1][(~mask).flat]):
                t.set_text(policy_symbols[pol])
                t.set_size("xx-large")
        plt.show()


if __name__ == "__main__":
    # You can experiment here or you can experiment in a separate script where you import the GridWorld.
    name = "3x4"
    # algorithm_rtol = 1e-4  # relative tolerance for comparing two floats
    # algorithm_atol = 1e-08  # absolute tolerance for comparing two floats
    #
    # valuations = []
    # changes = []
    # old_policy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # p = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # for i in p:
    #     gw = GridWorld.get_world(name, action_proba=i)
    #     Q = gw.value_iteration(rtol=algorithm_rtol, atol=algorithm_atol)
    #     policy = gw.Q2policy(Q)
    #     V = gw.Q2V(Q)
    #
    #     valuations.append(V.tolist())
    #
    #     # compare old and current policy
    #     policy_compare = policy == old_policy
    #     if not policy_compare.all():
    #         changes.append(i)
    #     old_policy = policy
    #
    # valuations = np.transpose(valuations)
    #
    # # p for which there occurs a change in the optimal policy
    # for ch in changes:
    #     plt.axvline(ch, 0, 1, c='r', ls='--', lw=0.7)
    #
    # # plot graph for each state
    # states = [0, 3, 6, 8, 9, 10, 11]
    # for state in states:
    #     plt.plot(p, valuations[state], label=str(state))
    # plt.legend()
    # plt.show()
