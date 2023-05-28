import time

from OmegaGomoku import *
import re
import numba as nb
import numpy as np

board_status = """
    0 1 2 3 4 5 6
0   . . . . . . .
1   . . . . . . O
2   . . . . . . .
3   . . . . X . .
4   . . . X . . .
5   . . X . . . .
6   . X . . . . .


"""

board_logs = [
    [2, 5, Player.BLACK],
    [4, 1, Player.WHITE],
    [4, 2, Player.BLACK],
    [3, 2, Player.WHITE],
    [3, 5, Player.BLACK],
    [2, 3, Player.WHITE],
    [5, 0, Player.BLACK],
    [1, 4, Player.WHITE],
    [0, 4, Player.BLACK],
]


# ['.', 'O', 'X']
def set_board_to_status(board: Board, status: str):
    status = re.sub(r'(\d| )', '', status).strip().split('\n')
    board_size = len(status)
    for y in range(board_size):
        for x in range(board_size):
            if status[y][x] != '.':
                board.put(x, y, ['.', 'O', 'X'].index(status[y][x]))


def set_board_to_log(board: Board, logs):
    for log in logs:
        x, y, player = log
        board.board[player - 1, x, y] = 1


board = Board(board_size=7)
set_board_to_status(board, board_status)
# set_board_to_log(board, board_logs)
# print(board.put(0, 5, Player.WHITE))

# print(Board.beautify_board(b))
print(board.put(5, 2, Player.WHITE))
print(board)


def get_lines_to_check(board: np.ndarray, x, y, pattern_len) -> np.ndarray:
    board_size = board.shape[0]
    lines = []
    # 横向
    start_x = max(0, x - pattern_len + 1)
    end_x = min(board_size - pattern_len + 1, x + 1)
    for i in range(start_x, end_x):
        lines.append(board[i:i + pattern_len, y])
    # 纵向
    start_y = max(0, y - pattern_len + 1)
    end_y = min(board_size - pattern_len + 1, y + 1)
    for i in range(start_y, end_y):
        lines.append(board[x, i:i + pattern_len])
    # 对角线
    print(x, y)
    diag_k = y - x
    # diag_s = min(x, y)
    diag_s = x if diag_k >= 0 else y
    diag = board.diagonal(diag_k)
    print(diag)
    start_i = max(0, diag_s - pattern_len + 1)
    end_i = min(len(diag) - pattern_len + 1, diag_s + 1)
    for i in range(start_i, end_i):
        lines.append(diag[i:i + pattern_len])
    # 斜对角线
    anti_diag = np.diag(np.fliplr(board), board_size - y - x - 1)
    start_i = max(0, diag_s - pattern_len + 1)
    end_i = min(len(anti_diag) - pattern_len + 1, diag_s + 1)
    print(anti_diag, start_i, end_i)
    for i in range(start_i, end_i):
        lines.append(anti_diag[i:i + pattern_len])
    lines = np.unique(np.asarray(lines), axis=0)
    return lines


# b = board.get_state()[1]
b = np.rot90(np.fliplr(np.arange(7 ** 2).reshape((7, 7))))
print(Board.beautify_board(b, number_mode=True, space='\t'))
print(get_lines_to_check(b, 5, 2, 5))
