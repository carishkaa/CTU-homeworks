from random import Random
import math
import numpy as np  # type: ignore


def extend_game_with_counter(Game, estimate_fn=None):
    class ExtendedGame(Game):
        _estimated_nodes_count = 0
        _expanded_nodes_count = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._is_expanded = False

        def legal_actions(self, *args, **kwargs):
            if not self._is_expanded:
                ExtendedGame._expanded_nodes_count += 1
                self._is_expanded = True
            return super().legal_actions(*args, **kwargs)

        def apply(self, *args, **kwargs):
            self._is_expanded = False
            return super().apply(*args, **kwargs)

        def clone(self):
            b = super().clone()
            b.__class__ = ExtendedGame
            b._is_expanded = False
            return b

        @property
        def expanded_nodes_count(self):
            return ExtendedGame._expanded_nodes_count

        @property
        def estimated_nodes_count(self):
            return ExtendedGame._estimated_nodes_count

    def _estimate_fn(*args, **kwargs):
        ExtendedGame._estimated_nodes_count += 1
        return estimate_fn(*args, **kwargs)

    if estimate_fn is None:
        return ExtendedGame
    return ExtendedGame, _estimate_fn


def start_random(game_fn, steps, seed, assert_not_terminal=False):
    random = Random()
    random.seed(seed)
    game = game_fn()
    for _ in range(steps):
        if game.terminal():
            if assert_not_terminal:
                assert False, "the game terminated"
            game = game_fn()

        action = random.choice(game.legal_actions())
        if hasattr(game, "_apply_internal"):
            game._apply_internal(action)
        else:
            game.apply(action)
    return game


def play_tournament(game, player1, player2):
    while not game.terminal():
        action = player1.act(game)
        game.apply(action)
        player2, player1 = player1, player2


def connect4_estimate(agent, seed, depth, random_moves):
    from connect4 import Connect4, connect4_score
    game_fn, score_fn = extend_game_with_counter(Connect4, connect4_score)
    agent = agent(score_fn, 0, depth)
    game = start_random(game_fn, random_moves, seed, assert_not_terminal=True)
    estimate = agent.compute_estimate(game)
    return estimate, game.expanded_nodes_count, game.estimated_nodes_count


def tree_game_estimate(agent, seed, depth):
    branching_factor = 4
    from testing_games import TreeGame, create_deep_tree_game_score
    score_fn = create_deep_tree_game_score(depth, branching_factor, seed)
    game_fn_o, score_fn = extend_game_with_counter(TreeGame, score_fn)
    game = game_fn_o(depth + 1, branching_factor)
    agent = agent(score_fn, 0, depth)
    estimate = agent.compute_estimate(game)
    return estimate, game.expanded_nodes_count, game.estimated_nodes_count


def connect4_mean_and_std(Agent, seed, start_steps, simulations=1000):
    from connect4 import Connect4
    estimates = []
    game = start_random(Connect4, start_steps, seed, assert_not_terminal=True)

    for _ in range(200):
        agent = Agent(simulations)
        estimates.append(agent.compute_estimate(game.clone()))
    return np.mean(estimates), np.std(estimates)


ZTEST_LEVEL = 1.96


def ztest(observed, expected):
    (m1, s1) = observed
    (m2, s2) = expected
    zstat = math.sqrt(200) * (m1 - m2) / math.sqrt(s1 ** 2 + s2 ** 2)
    print("mean: %.4f, std: %.4f" % observed)
    print(f"Z-statistic: {zstat}")
    return zstat
