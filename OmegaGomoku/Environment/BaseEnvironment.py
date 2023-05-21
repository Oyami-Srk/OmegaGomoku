from abc import ABC, abstractmethod
import numpy as np
from . import Board


class BaseGomokuEnv:
    @abstractmethod
    def __init__(self, board_size=8, win_size=5):
        self.board_size = board_size
        self.win_size = win_size
        assert win_size <= board_size

    @abstractmethod
    def reset(self) -> Board:
        """
        重置环境状态
        :return: 状态数组，两通道的棋盘
        """
        pass

    @abstractmethod
    def step(self, action: tuple[int, int] | int) -> tuple[Board, int, bool]:
        """
        :param action: 一个由着子位置X和Y组成的元组或int:X*size + Y
        :returns:
            board: 着子之后的棋盘状态,
            reward: 奖励分数,
            ended: 棋局是否结束
        """
        pass
