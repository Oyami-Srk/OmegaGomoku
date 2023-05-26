from abc import ABC, abstractmethod
from . import Board
from ..Agents import BaseAgent
from ..GUIBoard import GUIBoard


class BaseGomokuEnv(ABC):
    @abstractmethod
    def __init__(self, rival_agent: BaseAgent, board_size=8, win_size=5, gui_board: GUIBoard = None):
        self.board_size = board_size
        self.win_size = win_size
        self.rival_agent = rival_agent
        self.gui_board = gui_board
        assert win_size <= board_size

    @abstractmethod
    def reset(self) -> Board:
        """
        重置环境状态
        :return: 状态数组，两通道的棋盘
        """
        pass

    @abstractmethod
    def step(self, action: tuple[int, int] | int) -> tuple[Board, float, int | None]:
        """
        :param action: 一个由着子位置X和Y组成的元组或int:X*size + Y
        :returns:
            - board: 着子之后的棋盘状态
            - reward: 奖励分数
            - terminal_status: 棋局结束状态, None代表未结束，1代表胜利，-1代表失败，0代表平局
        """
        pass

    @abstractmethod
    def create_eval(self) -> 'BaseGomokuEnv':
        pass
