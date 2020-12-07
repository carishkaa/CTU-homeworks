from random import Random

BOARDWIDTH = 7
BOARDHEIGHT = 6


def apply_new(game, action):
    game = game.clone()
    game.apply(action)
    return game


class Connect4:
    def __init__(self):
        self._to_play = 0
        self._bitboards = [0, 0]
        self._board = [0 for _ in range(BOARDWIDTH)]

    @property
    def to_play(self):
        return self._to_play

    def apply(self, action):
        """
        Applies selected action. Adds a token to column specified by :action \in [0,6]
        :param int action: The column number where a token is requested to be placed.
        """
        assert 0 <= action < BOARDWIDTH
        assert self._board[action] < BOARDHEIGHT

        self._flip_bit((action, self._board[action]), self._to_play)
        self._board[action] += 1
        self._to_play = 1 - self._to_play

    def _flip_bit(self, position, player):
        x, y = position
        self._bitboards[player] |= (1 << (x * 7 + y))

    def _has_won(self, bitboard):
        # taken from http://stackoverflow.com/q/7033165/1524592
        y = bitboard & (bitboard >> 6)
        if (y & (y >> 2 * 6)):  # check \ diagonal
            return True
        y = bitboard & (bitboard >> 7)
        if (y & (y >> 2 * 7)):  # check horizontal
            return True
        y = bitboard & (bitboard >> 8)
        if (y & (y >> 2 * 8)):  # check / diagonal
            return True
        y = bitboard & (bitboard >> 1)
        if (y & (y >> 2)):  # check vertical
            return True
        return False

    def _is_draw(self, overall_bitboard):
        """
        If the board has all of its valid slots filled, then it is a draw.
        We mask the board to a bitboard with all positions filled
        (0xFDFBF7EFDFBF) and if all the bits are active, it is a draw.
        """
        return (overall_bitboard & 0xFDFBF7EFDFBF) == 0xFDFBF7EFDFBF

    def terminal(self):
        return self.terminal_value(0) is not None

    def terminal_value(self, player):
        """
        This function returns 1 if :player won the game, -1 if it lost the game
        0 is returned in the case of a draw and None is returned otherwise
        :param int player: 0 for first player and 1 for second player
        """

        if self._has_won(self._bitboards[player]):
            return 1

        if self._has_won(self._bitboards[1 - player]):
            return -1

        if self._is_draw(self._bitboards[0] | self._bitboards[1]):
            return 0
        return None

    def clone(self):
        """
        Clones the instance of this board
        """
        b = Connect4()
        b._to_play = self._to_play
        b._bitboards = list(self._bitboards)
        b._board = list(self._board)
        return b

    def legal_actions(self):
        """
        Returns the list of available actions
        """
        return [i for i, v in enumerate(self._board) if v < BOARDHEIGHT]

    def __eq__(self, other):
        if not isinstance(other, Connect4):
            return False
        return all((x == y for (x, y) in zip(self._bitboards, other._bitboards)))

    def __hash__(self):
        return hash(tuple(self._bitboards))

    @property
    def board(self):
        return self._bitboards


def _evaluate3(oppBoard, myBoard):
    """
    Returns the number of possible 3 in a rows in bitboard format.
    Running time: O(1)
    http://www.gamedev.net/topic/596955-trying-bit-boards-for-connect-4/
    """
    inverseBoard = ~(myBoard | oppBoard)
    rShift7MyBoard = myBoard >> 7
    lShift7MyBoard = myBoard << 7
    rShift14MyBoard = myBoard >> 14
    lShit14MyBoard = myBoard << 14
    rShift16MyBoard = myBoard >> 16
    lShift16MyBoard = myBoard << 16
    rShift8MyBoard = myBoard >> 8
    lShift8MyBoard = myBoard << 8
    rShift6MyBoard = myBoard >> 6
    lShift6MyBoard = myBoard << 6
    rShift12MyBoard = myBoard >> 12
    lShift12MyBoard = myBoard << 12
    # check _XXX and XXX_ horizontal
    result = inverseBoard & rShift7MyBoard & rShift14MyBoard \
             & (myBoard >> 21)
    result |= inverseBoard & rShift7MyBoard & rShift14MyBoard \
              & lShift7MyBoard
    result |= inverseBoard & rShift7MyBoard & lShift7MyBoard \
              & lShit14MyBoard
    result |= inverseBoard & lShift7MyBoard & lShit14MyBoard \
              & (myBoard << 21)
    # check XXX_ diagonal /
    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard \
              & (myBoard >> 24)
    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard \
              & lShift8MyBoard
    result |= inverseBoard & rShift8MyBoard & lShift8MyBoard \
              & lShift16MyBoard
    result |= inverseBoard & lShift8MyBoard & lShift16MyBoard \
              & (myBoard << 24)
    # check _XXX diagonal \
    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard \
              & (myBoard >> 18)
    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard \
              & lShift6MyBoard
    result |= inverseBoard & rShift6MyBoard & lShift6MyBoard \
              & lShift12MyBoard
    result |= inverseBoard & lShift6MyBoard & lShift12MyBoard \
              & (myBoard << 18)
    # check for _XXX vertical
    result |= inverseBoard & (myBoard << 1) & (myBoard << 2) \
              & (myBoard << 3)
    return result


def _evaluate2(oppBoard, myBoard):
    """
    Returns the number of possible 2 in a rows in bitboard format.
    Running time: O(1)
    """
    inverseBoard = ~(myBoard | oppBoard)
    rShift7MyBoard = myBoard >> 7
    rShift14MyBoard = myBoard >> 14
    lShift7MyBoard = myBoard << 7
    lShift14MyBoard = myBoard << 14
    rShift8MyBoard = myBoard >> 8
    lShift8MyBoard = myBoard << 8
    lShift16MyBoard = myBoard << 16
    rShift16MyBoard = myBoard >> 16
    rShift6MyBoard = myBoard >> 6
    lShift6MyBoard = myBoard << 6
    rShift12MyBoard = myBoard >> 12
    lShift12MyBoard = myBoard << 12

    # check for _XX
    result = inverseBoard & rShift7MyBoard & rShift14MyBoard
    result |= inverseBoard & rShift7MyBoard & rShift14MyBoard
    result |= inverseBoard & rShift7MyBoard & lShift7MyBoard
    # check for XX_
    result |= inverseBoard & lShift7MyBoard & lShift14MyBoard
    # check for XX / diagonal
    result |= inverseBoard & lShift8MyBoard & lShift16MyBoard
    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard
    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard
    result |= inverseBoard & rShift8MyBoard & lShift8MyBoard
    # check for XX \ diagonal
    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard
    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard
    result |= inverseBoard & rShift6MyBoard & lShift6MyBoard
    result |= inverseBoard & lShift6MyBoard & lShift12MyBoard
    # check for _XX vertical
    result |= inverseBoard & (myBoard << 1) & (myBoard << 2) \
              & (myBoard << 2)
    return result


def _evaluate1(oppBoard, myBoard):
    """
    Returns the number of possible 1 in a rows in bitboard format.
    Running time: O(1)
    Diagonals are skipped since they are worthless.
    """
    inverseBoard = ~(myBoard | oppBoard)
    # check for _X
    result = inverseBoard & (myBoard >> 7)
    # check for X_
    result |= inverseBoard & (myBoard << 7)
    # check for _X vertical
    result |= inverseBoard & (myBoard << 1)
    return result


def _bitboard_bits(i):
    """"
    Returns the number of bits in a bitboard (7x6).
    Running time: O(1)
    Help from: http://stackoverflow.com/q/9829578/1524592
    """
    i = i & 0xFDFBF7EFDFBF  # magic number to mask to only legal bitboard
    # positions (bits 0-5, 7-12, 14-19, 21-26, 28-33, 35-40, 42-47)
    i = (i & 0x5555555555555555) + ((i & 0xAAAAAAAAAAAAAAAA) >> 1)
    i = (i & 0x3333333333333333) + ((i & 0xCCCCCCCCCCCCCCCC) >> 2)
    i = (i & 0x0F0F0F0F0F0F0F0F) + ((i & 0xF0F0F0F0F0F0F0F0) >> 4)
    i = (i & 0x00FF00FF00FF00FF) + ((i & 0xFF00FF00FF00FF00) >> 8)
    i = (i & 0x0000FFFF0000FFFF) + ((i & 0xFFFF0000FFFF0000) >> 16)
    i = (i & 0x00000000FFFFFFFF) + ((i & 0xFFFFFFFF00000000) >> 32)
    return i


def _eval_cost(myBoard, oppBoard):
    """
    Returns cost of each board configuration.
    winning is a winning move
    blocking is a blocking move
    Running time: O(7n)
    """
    OppCost3Row = 1000
    MyCost3Row = 3000
    OppCost2Row = 500
    MyCost2Row = 500
    OppCost1Row = 100
    MyCost1Row = 100

    get3Win = _evaluate3(oppBoard, myBoard)
    winning3 = _bitboard_bits(get3Win) * MyCost3Row

    get3Block = _evaluate3(myBoard, oppBoard)
    blocking3 = _bitboard_bits(get3Block) * -OppCost3Row

    get2Win = _evaluate2(oppBoard, myBoard)
    winning2 = _bitboard_bits(get2Win) * MyCost2Row

    get2Block = _evaluate2(myBoard, oppBoard)
    blocking2 = _bitboard_bits(get2Block) * -OppCost2Row

    get1Win = _evaluate1(oppBoard, myBoard)
    winning1 = _bitboard_bits(get1Win) * MyCost1Row

    get1Block = _evaluate1(myBoard, oppBoard)
    blocking1 = _bitboard_bits(get1Block) * -OppCost1Row
    return winning3 + blocking3 + winning2 + blocking2 \
           + winning1 + blocking1


def connect4_score(board, player=0):
    terminal_value = board.terminal_value(player)
    if terminal_value is not None:
        if terminal_value == 0:
            return 0
        else:
            return terminal_value * float("inf")

    boards = tuple(board.board)
    return _eval_cost(*boards) * (1 if player == 0 else -1)
