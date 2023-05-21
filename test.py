from OmegaGomoku import *

board = Board()

log = [
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
    [3, 10, Player.BLACK],
    [3, 10, Player.WHITE],
]

print(board)
print(Board.beautify_board(board.get_suggested_moves(0) * board.get_valid_moves(), number_mode=True))
