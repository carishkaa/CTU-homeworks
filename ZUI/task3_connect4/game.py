from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional


class Game(metaclass=ABCMeta):
    """
    Game interface
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def to_play(self) -> int:
        pass

    @abstractmethod
    def apply(self, action: int) -> None:
        """
        Applies selected action. Note, that this changes the state of the game as well as the hash of this object.
        :param int action: The column number where a token is requested to be placed.
        """
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """
        Returns True if the game has ended
        """
        return self.terminal_value(0) is not None

    @abstractmethod
    def terminal_value(self, player: int) -> Optional[int]:
        """
        This function returns 1 if :player won the game, -1 if it lost the game
        0 is returned in the case of a draw and None is returned otherwise
        :param int player: 0 for first player and 1 for second player
        """
        pass

    @abstractmethod
    def clone(self) -> 'Game':
        """
        Clones the instance of this game
        """
        pass

    @abstractmethod
    def legal_actions(self) -> List[int]:
        """
        Returns the list of available actions
        """
        pass
