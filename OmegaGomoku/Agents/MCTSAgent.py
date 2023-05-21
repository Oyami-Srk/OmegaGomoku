import json
import math

import numpy as np

from . import BaseAgent
from ..DQN import BaseDQN
from tensorboardX import SummaryWriter
from ..Environment import Board


class MCTSAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        pass

    def act(self, board: Board, player):
        pass

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
