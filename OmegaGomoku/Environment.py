import numpy as np
from .Enums import *


# Gomoku Environment
# 五子棋环境
class GomokuEnv:
    def __init__(self, board_size=8, win_size=5):
        self.board_size = board_size
        self.win_size = win_size
        assert win_size <= board_size
        # self.board = np.zeros((board_size, board_size), dtype=int)
        # 如果使用一层棋盘，0代表空，1和2代表两种棋子，会无法收敛
        # 因为会变成回归问题
        self.board = np.zeros((2, board_size, board_size), dtype=int)
        self.current_player = Player.BLACK
        self.winner = None
        self.done = False
        self.steps = 0

    def reset(self) -> np.ndarray:
        """
        重置环境状态
        :return:
        """
        # self.board = np.zeros((self.board_size, self.board_size))
        self.board = np.zeros((2, self.board_size, self.board_size))
        self.current_player = Player.BLACK
        self.winner = None
        self.done = False
        return self.board

    def get_pieces_connected(self, player, x, y):
        """
        检查着子产生的连线情况
        :param player:
        :param x:
        :param y:
        :return:
        """
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

    def check_piece_around(self, x, y):
        """
        for i in range(max(0, x - 1), min(self.board_size, x + 2)):
            for j in range(max(0, y - 1), min(self.board_size, y + 2)):
                for p in (0, 1):
                    if self.board[p, j, i] == 1:
                        return True
        return False
        """
        surrounding = self.board[:,
                      max(0, x - 1):min(self.board_size, x + 2),
                      max(0, y - 1):min(self.board_size, y + 2)]
        # 判断是否存在着子
        # return np.any(surrounding == 1)
        return np.sum(surrounding) - 1 > 0

    def step(self, action: tuple[int, int] | int) -> tuple[np.ndarray, int, bool]:
        """
        :param action: 一个由着子位置X和Y组成的元组或int:X*size + Y
        :returns:
            board: 当前棋盘状态,
            reward: 奖励分数,
            ended: 棋局是否结束
        """
        # 如果游戏结束，返回当前棋盘，0分，True
        if self.done:
            return self.board, 0, True
        x, y = action if isinstance(action, tuple) else divmod(action, self.board_size)
        # 如果这个位置已经有棋子，返回当前棋盘，-10分，False
        # if self.board[x][y] != 0:
        #     return self.board, -10, False
        if self.board[0, x, y] != 0 or self.board[1, x, y] != 0:
            raise Exception(f'试图下在有子的位置：{x},{y}')
        self.steps += 1
        # 在这个位置放置当前玩家的棋子
        # self.board[x][y] = self.current_player
        self.board[self.current_player - 1, x, y] = 1
        if True:
            # 如果当前玩家获胜，设置胜者为当前玩家，游戏结束，返回当前棋盘，100分，True
            if self.check_win(self.current_player, x, y):
                self.winner = self.current_player
                self.done = True
                self.current_player = Player.SwitchPlayer(self.current_player)
                return self.board, 1, True
            self.current_player = Player.SwitchPlayer(self.current_player)
            if self.check_draw():
                self.done = True
            return self.board, 0, self.done
        else:
            # 获取当前着子产生的连线长度
            t = self.get_pieces_connected(self.current_player, x, y)
            if t == self.win_size:
                self.winner = self.current_player
                self.done = True
                return self.board, 100, True
            # 如果平局，游戏结束，返回当前棋盘，0分，True
            if self.check_draw():
                self.done = True
                return self.board, -50, True
            if t > 0:
                base = 50
                for i in range(self.win_size - 1, 2, -1):
                    if t == i:
                        return self.board, base - 10 * (self.win_size - i - 1), False
            # 切换玩家
            self.current_player = Player.SwitchPlayer(self.current_player)
            have_surroundings = self.check_piece_around(x, y)
            if have_surroundings:
                return self.board, 10, False
            else:
                if self.steps == 1:
                    return self.board, 0, False
                return self.board, -30, False

    def check_win(self, player, x, y) -> bool:
        """
        检查着子是否会让棋局产生胜负
        :param player: 当前玩家
        :param x: 行座标
        :param y: 列座标
        :return: 是否产生胜利
        """
        # player = self.board[x][y]
        t = self.win_size - 1
        board = self.board[player - 1]
        # 检查行是否成型
        for i in range(max(0, y - t), min(self.board_size - t, y + 1)):
            if np.all(board[x, i:i + self.win_size] == 1):
                return True
        # 检查列是否成型
        for i in range(max(0, x - t), min(self.board_size - t, x + 1)):
            if np.all(board[i:i + self.win_size, y] == 1):
                return True
        # 检查对角线
        for i in range(max(0, x - t), min(x + 1, self.board_size - t)):
            for j in range(max(0, y - t), min(y + 1, self.board_size - t)):
                if np.all(np.diag(board[i:i + self.win_size, j:j + self.win_size]) == 1):
                    return True
        # 检查斜对角线
        for i in range(max(0, x - t), min(x + 1, self.board_size - t)):
            for j in range(max(t, y), min(y + self.win_size, self.board_size)):
                if np.all(np.diag(board[i:i + self.win_size, j - t:j + 1][::-1]) == 1):
                    return True
        return False

    def check_draw(self) -> bool:
        """
        检查是否平局
        :return: 是否平局
        """
        # return np.all(self.board != 0)
        return np.count_nonzero(self.board) == self.board_size ** 2

    """
    def get_valid_moves(self):
        valid_moves = np.full((self.board_size, self.board_size), -np.inf)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    valid_moves[i][j] = 1
        return valid_moves
    """

    def get_valid_moves(self):
        valid_moves = (self.board == 0).astype(int)
        valid_moves[self.board != 0] = -1
        valid_moves = valid_moves * np.inf
        valid_moves[valid_moves == np.inf] = 1
        return valid_moves[0] * valid_moves[1]


def beautify_board(board: np.ndarray) -> str:
    result = ""
    for x in board:
        for y in x:
            result += f"{Player.PlayerSymbol(y)} "
        result += "\n"
    return result


def beautify_step_result(board: np.ndarray, reward: int, ended: bool) -> str:
    result = beautify_board(board[0] + board[1] * 2)
    result += f"\n该步着子的奖励：{reward}"
    if ended:
        result += "\n棋局结束"
    return result


if __name__ == '__main__':
    # 测试Check win
    env = GomokuEnv(5, 4)
    print(beautify_step_result(*env.step((3, 3))))
    print(beautify_step_result(*env.step((3, 2))))
    print(beautify_step_result(*env.step((0, 1))))
    print(beautify_step_result(*env.step((3, 4))))
    print(beautify_step_result(*env.step((0, 2))))
    print(beautify_step_result(*env.step((2, 4))))
    print(beautify_step_result(*env.step((0, 3))))
    print(beautify_step_result(*env.step((4, 2))))
    print(beautify_step_result(*env.step((0, 4))))
