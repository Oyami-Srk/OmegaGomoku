"""
第二版的DQN学习方法

Author: 韩昊轩
"""

from . import BaseDQN
from ..Hyperparameters import Hyperparameters
from ..Models import GomokuAI

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN2(BaseDQN):
    def __init__(self, board_size=8, win_size=5, hyperparameters=Hyperparameters(), cuda=False):
        self.hyperparameters = None  # Make pycharm happy =v=
        super().__init__(board_size, win_size, hyperparameters, cuda)
        self.action_size = board_size ** 2
        self.memory = np.zeros((hyperparameters.memory_size, self.action_size * 4 + 2))
        # self.memory = [None] * self.hyperparameters.memory_size
        self.eval_model = GomokuAI(board_size).to('cuda' if cuda else 'cpu')
        self.target_model = GomokuAI(board_size).to('cuda' if cuda else 'cpu')

        self.memory_counter = 0
        self.learn_count = 0
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.eval_model.parameters(), lr=hyperparameters.learning_rate)

    def act(self, state, valid_moves: np.ndarray):
        valid_moves = valid_moves.reshape(self.action_size)
        if np.random.rand() <= self.hyperparameters.epsilon:
            # Do Random
            action_values = np.random.uniform(-1, 1, size=self.action_size)
            valid_values = np.where(valid_moves == 1, action_values, -np.inf)
            return np.argmax(valid_values)

        # Do Think
        state = state.reshape([2, self.board_size, self.board_size])
        state = state[np.newaxis, :, :]
        # state = torch.from_numpy(state).float().unsqueeze(0).to('cuda' if self.cuda else 'cpu')
        state = torch.from_numpy(state).float().to('cuda' if self.cuda else 'cpu')
        # print(state, state.size())
        action_values = self.eval_model(state)
        valid_values = np.where(valid_moves == 1, action_values.cpu().detach().numpy()[0], -np.inf)
        # return np.argmax(action_values.cpu().numpy()[0] * valid_moves)
        return np.argmax(valid_values)

    def remember(self, state, next_state, action, reward, _is_done):
        memory = np.hstack((
            state.reshape(self.action_size * 2),
            next_state.reshape(self.action_size * 2),
            [action, reward]
        ))
        index = self.memory_counter % self.hyperparameters.memory_size
        self.memory[index, :] = memory
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter <= 8:
            return None
        if self.learn_count % self.hyperparameters.swap_model_each_iter == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        if self.memory_counter > self.hyperparameters.memory_size:
            sample_index = np.random.choice(self.hyperparameters.memory_size, size=self.hyperparameters.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.hyperparameters.batch_size)
        batch_memory = self.memory[sample_index, :]
        s = torch.from_numpy(
            batch_memory[:, 0:self.action_size * 2]
            .reshape([-1, 2, self.board_size, self.board_size])
            .astype(np.float32)
        ).to('cuda' if self.cuda else 'cpu')
        s_ = torch.from_numpy(
            batch_memory[:, self.action_size * 2:self.action_size * 4]
            .reshape([-1, 2, self.board_size, self.board_size])
            .astype(np.float32)
        ).to('cuda' if self.cuda else 'cpu')
        a = torch.from_numpy(batch_memory[:, self.action_size * 4]).long().to('cuda' if self.cuda else 'cpu')
        r = torch.from_numpy(batch_memory[:, self.action_size * 4 + 1]).to('cuda' if self.cuda else 'cpu')

        q_eval = self.eval_model(s)
        q_target = q_eval.clone()

        batch_index = np.arange(self.hyperparameters.batch_size, dtype=np.int32)
        q_target[batch_index, a] = torch.where(
            r != 0, r, self.hyperparameters.gamma * torch.max(self.target_model(s_), dim=1)[0]
        ).float()

        # fit
        q_target = q_target.detach()
        loss = 0
        for _ in range(self.hyperparameters.train_epochs):
            pred = self.eval_model(s)
            pred_loss = self.loss(pred, q_target)
            self.optimizer.zero_grad()
            pred_loss.backward()
            self.optimizer.step()
            loss += pred_loss
        self.learn_count += 1
        avg_loss = loss / self.hyperparameters.train_epochs
        return avg_loss

    def make_checkpoint(self):
        return {
            'model': self.eval_model.state_dict(),
            'memory': self.memory,
            'memory_counter': self.memory_counter,
            'hyperparameters': self.hyperparameters
        }

    def load_checkpoint(self, checkpoint):
        self.eval_model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        self.memory = checkpoint['memory']
        self.memory_counter = checkpoint['memory_counter']
        self.hyperparameters = checkpoint['hyperparameters']

    def get_network_name(self):
        return "DQN2-" + self.eval_model.model_name