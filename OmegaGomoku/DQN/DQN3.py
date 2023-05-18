"""
第三版的DQN学习方法

Author: 韩昊轩
"""
from collections import deque, namedtuple

from . import BaseDQN
from ..Hyperparameters import Hyperparameters
from ..Models import GomokuAI

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class ReplayMemory(object):
    def __init__(self, capacity, cuda=False):
        self.memory = deque([], maxlen=capacity)
        self.device = 'cuda' if cuda else 'cpu'

    def push(self, state, next_state, action, reward, is_done):
        self.memory.append((
            torch.from_numpy(state).to(self.device),
            torch.from_numpy(next_state).to(self.device),
            torch.tensor([action], device=self.device),
            torch.tensor([reward], device=self.device),
            is_done
        ))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN3(BaseDQN):
    def __init__(self, board_size=8, win_size=5, hyperparameters=Hyperparameters(), cuda=False):
        self.hyperparameters = None  # Make pycharm happy =v=
        super().__init__(board_size, win_size, hyperparameters, cuda)
        self.action_size = board_size ** 2
        self.memory = ReplayMemory(hyperparameters.memory_size)
        self.policy = GomokuAI(board_size).to(self.device)
        self.target = GomokuAI(board_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        # self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=hyperparameters.learning_rate)

    def act(self, state, valid_moves: np.ndarray):
        valid_moves = valid_moves.reshape(self.action_size)
        if np.random.rand() <= self.hyperparameters.epsilon:
            # Do Random
            action_values = np.random.uniform(-1, 1, size=self.action_size)
            valid_values = np.where(valid_moves == 1, action_values, -np.inf)
            return np.argmax(valid_values)

        # Do Think
        with torch.no_grad():
            state = state.reshape([2, self.board_size, self.board_size])
            state = state[np.newaxis, :, :]
            state = torch.from_numpy(state).float().to(self.device)
            action_values = self.policy(state)
            valid_values = np.where(valid_moves == 1, action_values.cpu().detach().numpy()[0], -np.inf)
            return np.argmax(valid_values)

    def remember(self, state, next_state, action, reward, is_done):
        self.memory.push(state, next_state, action, reward, is_done)

    def learn(self):
        if len(self.memory) < self.hyperparameters.batch_size:
            return None
        transition = self.memory.sample(self.hyperparameters.batch_size)
        batch = tuple(zip(*transition))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: not s, batch[-1])),
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s[1] for s in transition if not s[-1]]).reshape(
            (-1, 2, self.board_size, self.board_size)).float().to(self.device)
        state_batch = torch.cat(batch[0]).reshape((-1, 2, self.board_size, self.board_size)).float().to(self.device)
        action_batch = torch.cat(batch[2]).to(self.device)
        reward_batch = torch.cat(batch[3]).to(self.device)

        state_action_values = self.policy(state_batch)  # .gather(1, action_batch)
        next_state_values = torch.zeros(self.hyperparameters.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.hyperparameters.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = \
                policy_net_state_dict[key] * self.hyperparameters.tau \
                + target_net_state_dict[key] * (1 - self.hyperparameters.tau)
        self.target.load_state_dict(target_net_state_dict)

        return loss

    def make_checkpoint(self):
        return {
            'model': self.policy.state_dict(),
            'memory': self.memory,
            'hyperparameters': self.hyperparameters
        }

    def load_checkpoint(self, checkpoint):
        self.policy.load_state_dict(checkpoint['model'])
        self.target.load_state_dict(checkpoint['model'])
        self.memory = checkpoint['memory']
        self.hyperparameters = checkpoint['hyperparameters']

    def get_network_name(self):
        return "DQN3-" + self.policy.model_name
