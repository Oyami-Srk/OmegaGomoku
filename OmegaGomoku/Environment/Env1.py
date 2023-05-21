import numpy as np
from ..Enums import *
from . import BaseGomokuEnv
from .Board import Board


# Gomoku Environment
# 五子棋环境
class GomokuEnv(BaseGomokuEnv):
    def __init__(self, board_size=8, win_size=5):
        super().__init__(board_size, win_size)
        # self.board = np.zeros((board_size, board_size), dtype=int)
        # 如果使用一层棋盘，0代表空，1和2代表两种棋子，会无法收敛
        # 因为会变成回归问题
        # self.board = np.zeros((2, board_size, board_size), dtype=int)
        self.board = Board(board_size=board_size, win_size=win_size)
        self.current_player = Player.BLACK
        self.winner = None
        self.done = False

    def reset(self) -> Board:
        """
        重置环境状态
        :return:
        """
        # self.board = np.zeros((self.board_size, self.board_size))
        # self.board = np.zeros((2, self.board_size, self.board_size))
        self.board.reset()
        self.current_player = Player.BLACK
        self.winner = None
        self.done = False
        return self.board

    def step(self, action: tuple[int, int] | int) -> tuple[int, bool]:
        """
        :param action: 一个由着子位置X和Y组成的元组或int:X*size + Y
        :returns:
            board: 当前棋盘状态,
            reward: 奖励分数,
            ended: 棋局是否结束
        """
        x, y = action if isinstance(action, tuple) else divmod(action, self.board_size)
        result = self.board.put(x, y, self.current_player)

        if result is None:
            raise Exception(f'试图下在有子的位置：{x},{y}')
        reward, is_done = result

        # 切换玩家
        self.current_player = Player.SwitchPlayer(self.current_player)

        return reward, is_done


"""
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
"""
