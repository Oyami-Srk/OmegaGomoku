"""
初版的DQN学习方法

Author: 韩昊轩
"""

from . import BaseDQN
from ..Hyperparameters import Hyperparameters
from ..Models import GomokuAI

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class DQN1(BaseDQN):
    def __init__(self, board_size=8, win_size=5, hyperparameters=Hyperparameters(), cuda=False):
        self.hyperparameters = None  # Make pycharm happy =v=
        super().__init__(board_size, win_size, hyperparameters, cuda)
        self.action_size = board_size ** 2
        self.memory = []
        self.model = GomokuAI(board_size).to('cuda' if cuda else 'cpu')
        self.optimizer = optim.Adam(self.model.parameters(), lr=hyperparameters.learning_rate)
        self.criterion = nn.MSELoss()

    def get_network_name(self):
        return "DQN1-" + self.model.model_name

    def act(self, state, valid_moves: np.ndarray):
        valid_moves = valid_moves.reshape(self.action_size)
        if np.random.rand() <= self.hyperparameters.epsilon:
            action_values = np.random.uniform(-1, 1, size=self.action_size)
            valid_values = np.where(valid_moves == 1, action_values, -np.inf)
            return np.argmax(valid_values)

        state = torch.from_numpy(state).float().unsqueeze(0).to('cuda' if self.cuda else 'cpu')
        action_values = self.model(state)
        valid_values = np.where(valid_moves == 1, action_values.cpu().numpy()[0], -np.inf)
        # return np.argmax(action_values.cpu().numpy()[0] * valid_moves)
        return np.argmax(valid_values)

    def remember(self, state, next_state, action, reward, is_done):
        self.memory.append((state, next_state, action, reward, is_done))

    def learn(self):
        batch_size = self.hyperparameters.batch_size
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory[-self.hyperparameters.memory_size:], batch_size)
        loss = None
        for state, action, reward, next_state, done in batch:
            state = torch.from_numpy(state).float().unsqueeze(0).to('cuda' if self.cuda else 'cpu')
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to('cuda' if self.cuda else 'cpu')
            target = reward
            if not done:
                target = reward + self.hyperparameters.gamma * torch.max(self.model(next_state)[0])
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        return loss

    def make_checkpoint(self):
        return {
            'model': self.model.state_dict(),
            'memory': self.memory,
            'hyperparameters': self.hyperparameters
        }

    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])
        self.memory = checkpoint['memory']
        self.hyperparameters = checkpoint['hyperparameters']
