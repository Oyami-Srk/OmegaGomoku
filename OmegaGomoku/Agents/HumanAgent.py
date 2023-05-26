from . import BaseAgent
from ..Environment import Board
from ..GUIBoard import GUIBoard


class HumanAgent(BaseAgent):
    def __init__(self, gui_board: GUIBoard):
        self.gui_board = gui_board

    def act(self, board: Board, player):
        x, y, _ = self.gui_board.wait_place()
        return x, y

    def remember(self, state: Board, next_state: Board, action, reward, is_done):
        """
        Agent do not need to remember.
        """
        pass

    def learn(self, **kwargs):
        """
        Agent do not need to learn.
        """
        pass

    def finish(self, **kwargs):
        """
        Agent do not need to finish.
        """
        pass

    def save(self, **kwargs):
        """
        Agent do not need to save.
        """
        pass

    def create_eval(self) -> 'HumanAgent':
        pass
