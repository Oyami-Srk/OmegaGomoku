"""
五子棋棋盘
"""
import numpy as np
from ..Enums import Player, TermColor
from colorama import Fore, Back, Style

"""
   def get_pieces_connected(self, player, x, y):
        ""
        检查着子产生的连线情况
        :param player:
        :param x:
        :param y:
        :return:
        ""
        board = self.board[player - 1]
        for t in range(self.win_size - 1, 1, -1):
            # 检查纵列是否成型
            for i in range(max(0, y - t), min(self.board_size - t, y + 1)):
                if np.all(board[x, i:i + self.win_size] == 1):
                    return t + 1
            # 检查横行是否成型
            for i in range(max(0, x - t), min(self.board_size - t, x + 1)):
                if np.all(board[i:i + self.win_size, y] == 1):
                    return t + 1
            # 检查对角线
            for i in range(max(0, x - t), min(x + 1, self.board_size - t)):
                for j in range(max(0, y - t), min(y + 1, self.board_size - t)):
                    if np.all(np.diag(board[i:i + self.win_size, j:j + self.win_size]) == 1):
                        return t + 1
            # 检查斜对角线
            for i in range(max(0, x - t), min(x + 1, self.board_size - t)):
                for j in range(max(t, y), min(y + self.win_size, self.board_size)):
                    if np.all(np.diag(board[i:i + self.win_size, j - t:j + 1][::-1]) == 1):
                        return t + 1
        return 0
"""

"""
         for i in range(max(0, x - t), min(x + 1, self.board_size - t)):
                for j in range(max(0, y - t), min(y + 1, self.board_size - t)):
                    if np.all(np.diag(board[i:i + self.win_size, j:j + self.win_size]) == 1):
                        return t + 1
            # 检查斜对角线
            for i in range(max(0, x - t), min(x + 1, self.board_size - t)):
                for j in range(max(t, y), min(y + self.win_size, self.board_size)):
                    if np.all(np.diag(board[i:i + self.win_size, j - t:j + 1][::-1]) == 1):
                        return t + 1
"""


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
    diag_k = y - x
    # diag_s = min(x, y)
    # diag_s = x
    diag_s = x if diag_k >= 0 else y
    diag = board.diagonal(diag_k)
    start_i = max(0, diag_s - pattern_len + 1)
    end_i = min(len(diag) - pattern_len + 1, diag_s + 1)
    for i in range(start_i, end_i):
        lines.append(diag[i:i + pattern_len])
    # 斜对角线
    anti_diag = np.diag(np.fliplr(board), board_size - y - x - 1)
    start_i = max(0, diag_s - pattern_len + 1)
    end_i = min(len(anti_diag) - pattern_len + 1, diag_s + 1)
    for i in range(start_i, end_i):
        lines.append(anti_diag[i:i + pattern_len])
    lines = np.unique(np.asarray(lines), axis=0)
    return lines


class Board:
    PATTERNS = {
        'L5': [[1, 1, 1, 1, 1]],  # 连五，即获胜局面
        'H4': [[0, 1, 1, 1, 1, 0]],  # 活四，即两个点可以成连五
        'C4': [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1]],  # 冲四，仅有一个点可以连五
        'H3': [[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0]],  # 活三，能形成活四的三
        'M3': [[1, 1, 1, 0, 0], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1]],  # 眠三，只能形成冲四的三
        'H2': [[0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]],  # 活二，能形成活三的二
        'M2': [[0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 1, 0, 0, 1]]  # 眠二，能形成眠三的二
    }
    PATTERNS_FLATTEN = [item for sub in PATTERNS.values() for item in sub]
    PATTERN_LENS = set(map(len, PATTERNS_FLATTEN))
    PATTERN_MAX_LEN = max(PATTERN_LENS)
    PATTERN_MIN_LEN = min(PATTERN_LENS)

    def __init__(self, board_size=8, win_size=5):
        self.board = np.zeros((2, board_size, board_size), dtype=int)
        self.board_size = board_size
        self.win_size = win_size
        self.steps = 0

    def get_state(self) -> np.ndarray:
        return self.board

    def reset(self):
        self.board = np.zeros(self.board.shape, dtype=int)
        self.steps = 0

    def put(self, x, y, player) -> str | None:
        """
        :return: 返回当前着子产生的棋形
        """
        if (self.board[0, x, y] + self.board[1, x, y]) != 0:
            raise Exception(f"Player {player} 试图下在有子的位置 {x},{y}")
        self.board[player - 1, x, y] = 1
        pattern = self.find_pattern(x, y, player)
        self.steps += 1
        # is_draw = self.steps == self.board_size ** 2 and pattern != 'L5'
        # return (self.PATTERNS_REWARD[pattern] if pattern is not None else -150 if is_draw else 0,
        #         is_draw or pattern == 'L5')
        return pattern

    def is_done(self):
        return self.steps == self.board_size ** 2

    def is_empty(self):
        return self.steps == 0

    def get_valid_moves(self) -> np.ndarray:
        """
        获取当前有效的落子区域，一个Mask，-inf代表不可落子，1代表可以落子
        :return: 返回一个-inf和1组成的二维数组
        """
        valid_moves = (self.board == 0).astype(int)
        valid_moves[self.board != 0] = -1
        valid_moves = valid_moves * np.inf
        valid_moves[valid_moves == np.inf] = 1
        return valid_moves[0] * valid_moves[1]

    def get_suggested_moves(self, player, extend=1) -> np.ndarray:
        """
        获取当前建议的落子区域，一个Mask，-inf代表不建议落子，1代表建议落子。
        建议的落子区域为当前棋盘上有子存在的矩形外围一格
        :return: 返回一个-inf和1组成的二维数组
        """
        if self.steps == 0:
            return np.full((self.board_size, self.board_size), 1)
        pieces = np.argwhere(self.get_valid_moves() == -np.inf)
        min_x, min_y = pieces.min(axis=0)
        max_x, max_y = pieces.max(axis=0)
        min_x = max(0, min_x - extend)
        min_y = max(0, min_y - extend)
        max_x = min(self.board_size, max_x + extend + 1)
        max_y = min(self.board_size, max_y + extend + 1)
        suggested_moves = np.zeros((self.board_size, self.board_size))
        suggested_moves[min_x:max_x, min_y:max_y] = 1
        suggested_moves[suggested_moves == 0] = -np.inf
        return suggested_moves

    def find_pattern(self, x, y, player) -> str | None:
        """
        落子在x, y时产生的最佳棋形
        """
        # print(f"Find pattern when put {x},{y} as {player}")
        board = self.board[player - 1]
        is_a_attempt = board[x, y] == 0
        if is_a_attempt:
            board[x, y] = 1
        to_check_lines = {}
        for pattern_len in self.PATTERN_LENS:
            lines = get_lines_to_check(board, x, y, pattern_len)
            to_check_lines[pattern_len] = lines

        # 优化一下，根据line里面1的数量决定判断哪一个pattern
        for pattern_key, pattern_list in self.PATTERNS.items():
            for pattern in pattern_list:
                pattern_len = len(pattern)
                lines = to_check_lines[pattern_len]
                for line in lines:
                    if (line == pattern).all() or (line == list(reversed(pattern))).all():
                        if is_a_attempt:
                            board[x, y] = 0
                        return pattern_key
        if is_a_attempt:
            board[x, y] = 0
        return None

    @staticmethod
    def beautify_board(board: np.ndarray, mask_mode=False, number_mode=False, space=' ') -> str:
        result = '    ' + Fore.LIGHTBLUE_EX + space.join(map(str, range(board.shape[0]))) + Style.RESET_ALL + '\n'
        i = 0

        def get_symbol(i):
            non = Fore.WHITE + Style.BRIGHT + '.' + Style.RESET_ALL
            cross = TermColor.BOLD + Fore.RED + Style.BRIGHT + 'X' + Style.RESET_ALL
            circle = TermColor.BOLD + Fore.GREEN + Style.BRIGHT + 'O' + Style.RESET_ALL
            if mask_mode:
                return cross if i == -np.inf else circle if y == 1 else non
            elif number_mode:
                return str(i)
            else:
                return {'.': non, 'X': cross, 'O': circle}[Player.PlayerSymbol(i)]

        for x in board.transpose((1, 0)):
            result += Fore.LIGHTBLUE_EX + '{:<4d}'.format(i) + Style.RESET_ALL
            i += 1
            for y in x:
                result += get_symbol(y) + space
            result += "\n"
        return result

    def __str__(self):
        return Board.beautify_board(self.board[0] + self.board[1] * 2)


Board.PATTERNS_BY_LEN = {
    pattern_len: list(filter(lambda p: len(p) == pattern_len, Board.PATTERNS_FLATTEN))
    for pattern_len in Board.PATTERN_LENS
}
