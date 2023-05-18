"""
Deep Q Network基类
用于统一实验中使用的不同的DQN实现方法的接口

Author: 韩昊轩
"""
from ..Hyperparameters import Hyperparameters
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
import os
import re


class BaseDQN(ABC):
    def __init__(self, board_size=8, win_size=5, hyperparameters=Hyperparameters(), cuda=False):
        """
        :param board_size: 棋盘大小
        :param win_size: 取胜所需的棋子连线大小
        :param hyperparameters: 超参数
        :param cuda: 是否使用CUDA
        """
        self.board_size = board_size
        self.win_size = win_size
        self.hyperparameters = hyperparameters
        self.cuda = cuda
        self.device = 'cuda' if cuda else 'cpu'

    @abstractmethod
    def act(self, state, valid_moves: np.ndarray):
        pass

    @abstractmethod
    def remember(self, state, next_state, action, reward, is_done):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def make_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def get_network_name(self):
        pass

    def decay_epsilon(self, decay_function=None):
        if self.hyperparameters.epsilon > self.hyperparameters.epsilon_min:
            if decay_function is not None:
                self.hyperparameters.epsilon = decay_function(
                    self.hyperparameters.epsilon, self.hyperparameters.epsilon_decay_rate)
            else:
                self.hyperparameters.epsilon *= self.hyperparameters.epsilon_decay_rate

    def make_savepath(self, model_dir=None, no_make=False):
        save_dir = Path(".")
        if model_dir is not None:
            save_dir = save_dir.joinpath(model_dir)
        save_dir = save_dir.joinpath(self.get_network_name() + f"_{self.board_size}x{self.board_size}-{self.win_size}")
        if no_make and not os.path.exists(save_dir):
            return None
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def last_saved_episode(self, model_dir) -> int:
        save_dir = self.make_savepath(model_dir, True)
        if save_dir is None:
            return 0
        episodes = os.listdir(save_dir)
        if len(episodes) == 0:
            return 0

        def get_episode(filename):
            return int(re.findall(r"^(\d+)\.", filename)[0])

        latest = sorted(episodes, key=get_episode)[-1]
        return get_episode(latest)

    def save(self, episode, model_dir=None):
        checkpoint = self.make_checkpoint()
        # torch.save(self.model.state_dict(), filename)
        savepath = self.make_savepath(model_dir)
        savepath = savepath.joinpath(f"{episode}.pt")
        torch.save(checkpoint, savepath)

    def load(self, episode, model_dir=None):
        # self.model.load_state_dict(torch.load(filename))
        savepath = self.make_savepath(model_dir)
        savepath = savepath.joinpath(f"{episode}.pt")
        checkpoint = torch.load(savepath)
        self.load_checkpoint(checkpoint)
