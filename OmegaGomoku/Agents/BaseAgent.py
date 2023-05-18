from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def act(self, state: np.ndarray, valid_moves: np.ndarray):
        pass

    @abstractmethod
    def remember(self, state, next_state, action, reward, is_done):
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
