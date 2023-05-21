import numpy as np


def get_atk_value_offset(p):
    if len(p) == 5:
        # xx_xx, x_xxx, _xxxx
        if p.sum() == 4 and p.max() == 1:
            return 5, [p.argmin()]
    elif len(p) == 6:
        if p.sum() == 3 and p.max() == 1:
            # _xxx__
            if p[1] and p[2] and p[3]:
                return 4, [4]
            # __xxx_
            if p[2] and p[3] and p[4]:
                return 4, [1]
            # _x_xx_
            if p[1] and p[3] and p[4]:
                return 4, [2]
            # _xx_x_
            if p[1] and p[2] and p[4]:
                return 4, [3]
        # o_xxx_, oxxx__, o__xxx, oxx_x_, ox_xx_, o_x_xx, o_xx_x, r
        elif (p[1:6].sum() == 3 and p[1:6].max() == 1) \
                or (p[0:5].sum() == 3 and p[0:5].max() == 1):
            return 3, np.where(p == 0)[0].tolist()
        elif p.sum() == 2 and p.max() == 1:
            # _xx___
            if p[1] and p[2]:
                return 2, [3, 4]
            # __xx__
            elif p[2] and p[3]:
                return 2, [1, 4]
            # ___xx_
            elif p[3] and p[4]:
                return 2, [1, 2]
            # _x_x__
            elif p[1] and p[3]:
                return 2, [2, 4]
            # __x_x_
            elif p[2] and p[4]:
                return 2, [1, 3]
    return 0, []


def get_dfs_value_offset(p):
    if len(p) == 5:
        if p.sum() == 4 and p.max() == 1:
            # xx_xx, x_xxx
            if p[0] and p[4]:
                return 4, [p.argmin()]
    elif len(p) == 6:
        if p.sum() == 3 and p.max() == 1:
            # _xxx__
            if p[1] and p[2] and p[3]:
                return 3, [0, 4]
            # __xxx_
            elif p[2] and p[3] and p[4]:
                return 3, [1, 5]
            # _x_xx_
            elif p[1] and p[3] and p[4]:
                return 3, [0, 2, 5]
            # _xx_x_
            elif p[1] and p[2] and p[4]:
                return 3, [0, 3, 5]
        # _xxxx_
        elif p.sum() == 4 and p.max() == 1 and not p[0] and not p[5]:
            return 5, []
        # oxxxx_, |xxxx_, r
        elif p[1:5].sum() == 4 and p[1:5].max() == 1 and (p[0] == 0 or p[5] == 0):
            return 4, [p.argmin()]
    return 0, []


def index_by_diagonal(offset, i, position):
    if offset >= 0:
        return i + position - 1, offset + i + position - 1
    if offset < 0:
        return 0 - offset + i + position - 1, i + position - 1


def gen_dfs_atk_moves(board, atk):
    board = (board[0] + board[1] * 2).copy()
    board_size = board.shape[0]
    # add an edge of 3 around the board
    t_board = np.full((board_size + 2, board_size + 2), 3, np.int32)
    t_board[1:board_size + 1, 1:board_size + 1] = board
    board = t_board

    # when dfs, swap (1, 2)
    if not atk:
        board = np.where(board == 1, 2, np.where(board == 2, 1, board))

    level2 = np.zeros((board_size, board_size), np.int32)
    level3 = np.zeros((board_size, board_size), np.int32)
    level4 = np.zeros((board_size, board_size), np.int32)
    level5 = np.zeros((board_size, board_size), np.int32)

    # horizonal and vertical
    for i in range(board_size + 2):
        for j in range(board_size + 2):
            p = board[i, j:j + 5]
            if atk:
                v, positions = get_atk_value_offset(p)
            else:
                v, positions = get_dfs_value_offset(p)
            if v == 2:
                for position in positions:
                    level2[i - 1, j + position - 1] = 1
            if v == 3:
                for position in positions:
                    level3[i - 1, j + position - 1] = 1
            if v == 4:
                for position in positions:
                    level4[i - 1, j + position - 1] = 1
            if v == 5:
                for position in positions:
                    level5[i - 1, j + position - 1] = 1

            p = board[i:i + 5, j]
            if atk:
                v, positions = get_atk_value_offset(p)
            else:
                v, positions = get_dfs_value_offset(p)
            if v == 2:
                for position in positions:
                    level2[i + position - 1, j - 1] = 1
            if v == 3:
                for position in positions:
                    level3[i + position - 1, j - 1] = 1
            if v == 4:
                for position in positions:
                    level4[i + position - 1, j - 1] = 1
            if v == 5:
                for position in positions:
                    level5[i + position - 1, j - 1] = 1

            p = board[i, j:j + 6]
            if atk:
                v, positions = get_atk_value_offset(p)
            else:
                v, positions = get_dfs_value_offset(p)
            if v == 2:
                for position in positions:
                    level2[i - 1, j + position - 1] = 1
            if v == 3:
                for position in positions:
                    level3[i - 1, j + position - 1] = 1
            if v == 4:
                for position in positions:
                    level4[i - 1, j + position - 1] = 1
            if v == 5:
                for position in positions:
                    level5[i - 1, j + position - 1] = 1

            p = board[i:i + 6, j]
            if atk:
                v, positions = get_atk_value_offset(p)
            else:
                v, positions = get_dfs_value_offset(p)
            if v == 2:
                for position in positions:
                    level2[i + position - 1, j - 1] = 1
            if v == 3:
                for position in positions:
                    level3[i + position - 1, j - 1] = 1
            if v == 4:
                for position in positions:
                    level4[i + position - 1, j - 1] = 1
            if v == 5:
                for position in positions:
                    level5[i + position - 1, j - 1] = 1

    # diagonal
    for offset in range(-(board_size - 5), (board_size - 5) + 1):
        p = board.diagonal(offset)
        for i in range(board_size + 2 - offset - 5 + 1):
            if atk:
                v, positions = get_atk_value_offset(p[i:i + 5])
            else:
                v, positions = get_dfs_value_offset(p[i:i + 5])
            # v, positions = self.get_value_and_offset(p[i:i+5])
            if v == 2:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level2[x, y] = 1
            if v == 3:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level3[x, y] = 1
            if v == 4:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level4[x, y] = 1
            if v == 5:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level5[x, y] = 1
            if atk:
                v, positions = get_atk_value_offset(p[i:i + 6])
            else:
                v, positions = get_dfs_value_offset(p[i:i + 6])
            # v, positions = self.get_value_and_offset(p[i:i+6])
            if v == 2:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level2[x, y] = 1
            if v == 3:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level3[x, y] = 1
            if v == 4:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level4[x, y] = 1
            if v == 5:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level5[x, y] = 1

    # swap rows to compute another diagonal
    m_board = board.copy()
    for i in range(board_size + 2):
        m_board[i] = board[board_size + 1 - i]
    # another diagonal
    for offset in range(-(board_size - 5), (board_size - 5) + 1):
        p = m_board.diagonal(offset)
        for i in range(board_size + 2 - offset - 5 + 1):
            if atk:
                v, positions = get_atk_value_offset(p[i:i + 5])
            else:
                v, positions = get_dfs_value_offset(p[i:i + 5])
            # v, positions = self.get_value_and_offset(p[i:i+5])
            if v == 2:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level2[board_size - 1 - x, y] = 1
            if v == 3:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level3[board_size - 1 - x, y] = 1
            if v == 4:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level4[board_size - 1 - x, y] = 1
            if v == 5:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level5[board_size - 1 - x, y] = 1
            if atk:
                v, positions = get_atk_value_offset(p[i:i + 6])
            else:
                v, positions = get_dfs_value_offset(p[i:i + 6])
            # v, positions = self.get_value_and_offset(p[i:i+6])
            if v == 2:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level2[board_size - 1 - x, y] = 1
            if v == 3:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level3[board_size - 1 - x, y] = 1
            if v == 4:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level4[board_size - 1 - x, y] = 1
            if v == 5:
                for position in positions:
                    x, y = index_by_diagonal(offset, i, position)
                    level5[board_size - 1 - x, y] = 1
    if atk:
        return level2, level3, level4, level5
    else:
        return level3, level4
