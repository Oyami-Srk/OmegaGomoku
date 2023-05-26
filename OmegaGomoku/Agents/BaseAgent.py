from abc import ABC, abstractmethod
import numpy as np
from ..Environment import Board


class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def act(self, board: Board, player):
        pass

    @abstractmethod
    def remember(self, state: Board, next_state: Board, action, reward, is_done):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass

    @abstractmethod
    def finish(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def create_eval(self) -> 'BaseAgent':
        pass
