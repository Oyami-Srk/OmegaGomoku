import numpy as np
from ..Enums import *
from . import BaseGomokuEnv, Reward
from .Board import Board
from ..Agents import BaseAgent
from ..GUIBoard import GUIBoard


# Gomoku Environment
# 五子棋环境
class GomokuEnv(BaseGomokuEnv):
    def __init__(self, rival_agent: BaseAgent, board_size=8, win_size=5, gui_board: GUIBoard = None):
        super().__init__(rival_agent, board_size, win_size, gui_board)
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
        if self.gui_board is not None:
            self.gui_board.reset()
        return self.board

    def step(self, action: tuple[int, int] | int) -> tuple[Board, float, int | None]:
        """
        :param action: 一个由着子位置X和Y组成的元组或int:X*size + Y
        :returns:
            - board: 着子之后的棋盘状态
            - reward: 奖励分数
            - terminal_status: 棋局结束状态, None代表未结束，1代表胜利，-1代表失败，0代表平局
        """

        # 先手 action
        x, y = action if isinstance(action, tuple) else divmod(action, self.board_size)
        pattern = self.board.put(x, y, self.current_player)
        is_done = self.board.is_done()
        if self.gui_board is not None:
            self.gui_board.draw_piece(x, y, self.current_player)
        if is_done or pattern == 'L5':
            # 如果棋局结束，判断是否此着形成连五，否则平局
            terminal_status = 1 if pattern == 'L5' else 0
            reward = Reward.calculate_reward(pattern, None, None, terminal_status)
            if self.gui_board is not None:
                self.gui_board.info(f"AI落子于({x},{y})，奖励：{reward}")
            return self.board, reward, terminal_status
        # 切换玩家
        self.current_player = Player.SwitchPlayer(self.current_player)
        # 先手的action可以打断的后手玩家的棋形
        break_pattern = self.board.find_pattern(x, y, self.current_player)
        # 后手 action
        rival_action = self.rival_agent.act(self.board, self.current_player)
        rx, ry = rival_action if isinstance(rival_action, tuple) else divmod(rival_action, self.board_size)
        rival_pattern = self.board.put(rx, ry, self.current_player)
        is_done = self.board.is_done()
        terminal_status = None
        if is_done or rival_pattern == 'L5':
            terminal_status = -1 if rival_pattern == 'L5' else 0
        reward = Reward.calculate_reward(
            pattern, break_pattern, rival_pattern, terminal_status)
        if self.gui_board is not None:
            self.gui_board.draw_piece(rx, ry, self.current_player)
            self.gui_board.info(f"AI落子于({x},{y})，奖励：{reward}")
        # 切换玩家
        self.current_player = Player.SwitchPlayer(self.current_player)

        return self.board, reward, terminal_status

    def create_eval(self) -> 'GomokuEnv':
        """
        :return: 以当前环境参数创建的评估环境
        """
        return GomokuEnv(
            self.rival_agent.create_eval(),
            self.board_size,
            self.win_size,
            self.gui_board
        )
