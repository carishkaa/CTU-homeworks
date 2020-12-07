import math
import random
from game import Game
from typing import Callable


#
# Data types which might be useful for your implementation
# For more details follow the seminar slides
#
class Node(object):
    def __init__(self):
        self.visit_count = 0
        self.to_play = 0
        self.value_sum = 0
        self.children = {}
        self.unexplored_actions = None
        self.is_terminal = False

    def expanded(self) -> bool:
        return len(self.children) > 0 or self.is_terminal

    def fully_expanded(self) -> bool:
        return self.expanded() and \
               self.unexplored_actions is not None and \
               len(self.unexplored_actions) == 0 and \
               not self.is_terminal

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class DeterministicAgent:
    def __init__(self, evaluate_fn: Callable[[Game, int], float], maximizing_player: int, depth: int):
        self.maximizing_player = maximizing_player
        self.evaluate_fn = evaluate_fn
        self.estimate_depth = depth
        self.cache = {}

    def compute_estimate(self, game: Game) -> float:
        """
        Computes the game estimate
        """
        assert game.to_play == self.maximizing_player
        return self.alphabeta(game, -float("inf"), float("inf"), self.estimate_depth, 1)

    def act(self, game: Game) -> int:
        """
        Selects the best action according to its estimate
        """

        def _step_new(g, action):
            g = g.clone()
            g.apply(action)
            return g

        assert game.to_play == self.maximizing_player
        _, action = max(((self.compute_estimate(_step_new(game, action)), action) for action in game.legal_actions()))
        return action

    def alphabeta(self, game: Game, alpha: float, beta: float, depth: int, color: int) -> float:
        alpha_tmp = alpha
        if (game, depth) in self.cache:
            alpha_cache, beta_cache = self.cache[(game, depth)]
            if beta_cache <= alpha:
                return beta_cache
            if beta <= alpha_cache:
                return alpha_cache
            alpha = max(alpha, alpha_cache)
            beta = min(beta, beta_cache)
        else:
            alpha_cache = -float("inf")
            beta_cache = float("inf")

        # return heuristic value
        if depth == 0 or game.terminal():
            return color * self.evaluate_fn(game, self.maximizing_player)

        v = -float("inf")

        # create children list
        children = []
        for action in game.legal_actions():
            child = apply_new(game, action)
            children.append(child)

        # sort children
        children.sort(key=lambda x: -color * self.evaluate_fn(x, self.maximizing_player))

        # negamax
        for child in children:
            v = max(v, -1 * self.alphabeta(child, -1*beta, -1*alpha, depth - 1,  -1 * color))
            alpha = max(alpha, v)
            if beta <= alpha:
                break

        if v <= alpha_tmp:
            self.cache[(game, depth)] = (alpha_cache, v)
        elif alpha_tmp < v < beta:
            self.cache[(game, depth)] = (v, v)
        else:
            self.cache[(game, depth)] = (v, beta_cache)

        return alpha


class MonteCarloAgent:
    def __init__(self, num_simulations: int, uct_constant: float = math.sqrt(2)):
        self.num_simulations = num_simulations
        self.uct_constant = uct_constant

    def compute_estimate(self, game: Game) -> float:
        """
        Computes the game estimate
        """
        root = self.mcts(game)
        return root.value_sum / root.visit_count

    def act(self, game: Game) -> int:
        root = self.mcts(game)
        return self._select_action(root)

    def mcts(self, game: Game) -> Node:
        """
        Runs MCTS search on current game state
        """
        root = Node()
        root.unexplored_actions = game.legal_actions()
        for _ in range(self.num_simulations):
            node = root
            scratch_game = game.clone()
            search_path = [node]  # stack

            while node.fully_expanded():  # node is fully expanded
                action, node = self._uct_select(node)
                scratch_game = apply_new(scratch_game, action)
                search_path.append(node)

            node, search_path = self._expand(search_path, node, scratch_game)
            value = self._simulate(node, scratch_game)
            self._backpropagate(search_path, value)
        return root

    def _select_action(self, node):
        _, action = max((c.visit_count, a) for a, c in node.items())
        return action

    def _expand(self, search_path, node, game):
        """
        If the node is terminal, does nothing and returns the node.
        Otherwise, expands the node, selects a child, adds the child on the search path and returns it.
        """

        if node.is_terminal:
            return node, search_path

        # apply random unexplored action in game
        random_idx = random.randrange(len(node.unexplored_actions))
        action = node.unexplored_actions.pop(random_idx)
        game.apply(action)

        # create new child
        child = Node()
        child.to_play = game.to_play
        child.is_terminal = game.terminal()
        child.unexplored_actions = game.legal_actions()

        # append child to node and put to search path
        node.children[action] = child
        search_path.append(child)
        return child, search_path

    def _backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the
        tree to the root. We also increase the visit_count.
        :param float value: The game score relative to the last state on the search path. It could be 1 for winning player and -1 for losing player.
        """
        player = search_path[-1].to_play
        while len(search_path) > 0:
            child = search_path.pop()
            child.visit_count = child.visit_count + 1
            if child.to_play == player:
                child.value_sum = child.value_sum + value
            else:
                child.value_sum = child.value_sum - value

    def _simulate(self, node, game):
        """
        We simulate one rollout starting at node :node with current game state.
        :return: terminal_value relative to the node.to_play player.
        """
        while not game.terminal():
            actions = game.legal_actions()
            game = apply_new(game, actions[random.randrange(len(actions))])
        return game.terminal_value(node.to_play)

    def _uct_select(self, node):
        _, action, child = max(((self._uct_score(node, child), action, child) \
                                for action, child in node.children.items()), key=lambda x: x[0])
        return action, child

    def _uct_score(self, parent, child):
        """
        Computes the uct score of a given node
        """
        N = child.visit_count
        Np = parent.visit_count
        prior = (-1 * child.value_sum/N) if N > 0 else 0
        c = self.uct_constant
        return prior + c * math.sqrt(math.log(Np + 1)/(N + 1))


#
# Functions which might be useful for your implementation
#
def apply_new(game: Game, action: int) -> Game:
    game = game.clone()
    game.apply(action)
    return game


def less_then_or_inf(x, y):
    return x < y or (x == y and (x == -float("inf") or x == float("inf")))
