import math
import numpy as np  # type: ignore
from game import Game


class TreeGame(Game):
    def __init__(self, height=3, branching_factor=3, terminal_values=None):
        super().__init__()
        self.height = height
        self._number = 1
        self._step = 0
        self.branching_factor = branching_factor
        self.terminal_values = terminal_values

        if self.terminal_values:
            assert len(self.terminal_values) == self.branching_factor ** self.height

    def terminal(self):
        return self._step == self.height

    def terminal_value(self, player):
        if not self.terminal():
            return None
        if self._terminal_values:
            return self._terminal_values[self._number]
        else:
            raise Exception('Cannot call terminal_value without providing terminal_values')

    def legal_actions(self):
        return list(range(self.branching_factor))

    def apply(self, action):
        assert 0 <= action < self.branching_factor, "invalid action"
        assert not self.terminal(), "already terminated"

        self._number = (self._number - 1) * self.branching_factor + action + 2
        self._step += 1

    @property
    def to_play(self):
        return self._step % 2

    def clone(self) -> 'TreeGame':
        b = TreeGame(self.height, self.branching_factor, self.terminal_values)
        b._number = self._number
        b._step = self._step
        return b

    def __eq__(self, other):
        if not isinstance(other, TreeGame):
            return False
        return self._number == other._number

    def __hash__(self):
        return self._number % 1000


def create_tree_game_score(depth, values):
    def game_score(game, player=0):
        assert isinstance(game, TreeGame), "game is not TreeGame"
        if game._step != depth:
            # Called before the enf of the game
            return 0
        offset = int((1 - game.branching_factor ** depth) / (1 - game.branching_factor))
        val = values[game._number - offset - 1]
        if player != 0:
            val = -val
        return val

    return game_score


def create_deep_tree_game_score(depth, branching_factor, seed=0):
    np.random.seed(seed)
    values_count = (1 - branching_factor ** (depth + 1)) // (1 - branching_factor)
    values = np.ndarray((values_count,), dtype=np.int32)
    range_gen = branching_factor * 3
    values[0] = np.random.randint(-100, 101)
    for l in range(0, depth + 1):
        offset = int((1 - branching_factor ** l) / (1 - branching_factor))
        for i in range(branching_factor ** l):
            values[offset + i] += np.random.randint(-range_gen, range_gen + 1)
            if l < depth:
                for j in range(branching_factor):
                    child_index = (offset + i) * branching_factor + j + 2
                    values[child_index - 1] = values[offset + i]

    def game_score(game, player=0):
        assert isinstance(game, TreeGame), "game is not TreeGame"
        assert game._step <= depth, "cannot call the heuristic function on deeper games"
        val = values[game._number - 1]
        if player != 0:
            val = -val
        return val

    return game_score


def generate_bandits(arms, simulations, seed, scale=1.0):
    np.random.seed(seed)
    means = np.random.normal(size=(10,))
    return [np.random.normal(loc=x, scale=scale, size=(simulations,)) for x in means], means


class MultiArmedBanditGame(Game):
    def __init__(self, length, simulations, seed, arms=10, store=None, scale=1.0):
        super().__init__()
        self._to_play = 0
        self._step = 0
        self._last_action = None
        self.length = length
        if store is None:
            store = generate_bandits(arms, simulations, seed, scale=scale)
        self._bandits, self._means = store
        self._bandit = None
        assert length >= 2

    @property
    def to_play(self):
        return self._to_play

    def apply(self, action):
        assert self._step < self.length
        if self._step == self.length - 2:
            self._last_action = action
        if self._step == self.length - 1:
            self._bandit = action

        self._step += 1
        self._to_play = 1 - self._to_play

    def terminal(self):
        return self._step == self.length

    def terminal_value(self, player):
        if not self.terminal(): return None
        multiplier = 1 if player == 0 else -1
        return multiplier * self._bandits[self._last_action][self._bandit]


    def clone(self) -> 'MultiArmedBanditGame':
        b = MultiArmedBanditGame(self.length, -1, 0, len(self._bandits), (self._bandits, self._means))
        b._to_play = self._to_play
        b._step = self._step
        b._last_action = self._last_action
        b._bandit = self._bandit
        return b

    def legal_actions(self):
        if self._step < self.length - 2:
            return [0]
        if self._step == self.length - 2:
            return list(range(len(self._bandits)))
        if self._step == self.length - 1:
            return list(range(len(self._bandits[self._last_action])))
