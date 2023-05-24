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
from ..Environment import Board, Utils


class DQN3(BaseDQN):
    def __init__(self, board_size=8, win_size=5, hyperparameters=Hyperparameters(), cuda=False, training=True):
        self.hyperparameters = None  # Make pycharm happy =v=
        super().__init__(board_size, win_size, hyperparameters, cuda)
        self.action_size = board_size ** 2
        self.policy = GomokuAI(board_size).to(self.device)
        self.target = GomokuAI(board_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        # self.loss = nn.MSELoss()
        # self.optimizer = optim.AdamW(self.policy.parameters(), lr=hyperparameters.learning_rate)
        self.training = training
        loss_class = getattr(nn, hyperparameters.loss)
        optimizer_class = getattr(optim, hyperparameters.optimizer)
        self.loss = loss_class()
        self.optimizer = optimizer_class(self.policy.parameters(), lr=hyperparameters.learning_rate)

    def act(self, board: Board, player):
        valid_moves = board.get_valid_moves().reshape(self.action_size)
        suggested_moves = board.get_suggested_moves(player).reshape(self.action_size)
        if self.training:
            valid_moves *= suggested_moves  # Let valid moves only in suggested moves
        state = board.get_state()
        if self.training and np.random.rand() <= self.hyperparameters.epsilon:
            # Do Random
            """
            action_values = np.random.uniform(-1, 1, size=self.action_size)
            valid_values = np.where(valid_moves == 1, action_values, -np.inf)
            return np.argmax(valid_values)
            """
            # Do fast chose
            atk2, atk3, atk4, atk5 = Utils.gen_dfs_atk_moves(state, True)
            def3, def4 = Utils.gen_dfs_atk_moves(state, False)
            pool = None
            if atk5.any():
                pool = atk5
            elif def4.any():
                pool = def4
            elif atk4.any():
                pool = atk4
            elif def3.any():
                pool = def3
            elif atk3.any():
                pool = atk3
            elif atk2.any():
                pool = atk2
            else:
                # pool = valid_moves * suggested_moves
                pool = valid_moves
            pool = pool.reshape(self.action_size)
            action_values = np.random.uniform(-1, 1, size=self.action_size)
            valid_values = np.where(pool == 1, action_values, -np.inf)
            return np.argmax(valid_values)

        # Do Think
        state = state.reshape([2, self.board_size, self.board_size])
        state = state[np.newaxis, :, :]
        # state = torch.from_numpy(state).float().unsqueeze(0).to('cuda' if self.cuda else 'cpu')
        state = torch.from_numpy(state).float().to('cuda' if self.cuda else 'cpu')
        # print(state, state.size())
        with torch.no_grad():
            action_values = self.policy(state)
        valid_values = np.where(valid_moves == 1, action_values.cpu().detach().numpy()[0], -np.inf)
        # return np.argmax(action_values.cpu().numpy()[0] * valid_moves)
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

        state_action_values = self.policy(state_batch).gather(1, action_batch.reshape(-1, 1))
        next_state_values = torch.zeros(self.hyperparameters.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.hyperparameters.gamma) + reward_batch

        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))
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
