"""
第二版的DQN学习方法

Author: 韩昊轩
"""

from . import BaseDQN, Memory, MemoryEntry
from ..Hyperparameters import Hyperparameters
from ..Models import GomokuAI
from ..Environment import Board, Utils

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(BaseDQN):
    def __init__(self, board_size=8, win_size=5, hyperparameters=Hyperparameters(), cuda=False, training=True):
        super().__init__(board_size, win_size, hyperparameters, cuda)
        self.memory: Memory = self.memory  # For auto-complete
        self.action_size = board_size ** 2
        self.eval_model = GomokuAI(board_size).train(training).to(self.device)
        self.target_model = GomokuAI(board_size).eval().to(self.device)

        self.learn_count = 0
        self.training = training
        loss_class = getattr(nn, hyperparameters.loss)
        optimizer_class = getattr(optim, hyperparameters.optimizer)
        self.loss = loss_class()
        self.optimizer = optimizer_class(self.eval_model.parameters(), lr=hyperparameters.learning_rate)

    def act(self, board: Board, player):
        valid_moves = board.get_valid_moves().reshape(self.action_size)
        suggested_moves = board.get_suggested_moves(player, extend=3).reshape(self.action_size)
        if self.training:
            valid_moves *= suggested_moves  # Let valid moves only in suggested moves
        if board.is_empty():
            # 如果棋盘为空，在棋盘中心的5x5区域随机选取一个点位下棋
            round_size = 5
            board_size = board.board_size
            round_arr = np.full((board_size, board_size), -np.inf)
            s = (board_size - round_size) // 2
            round_arr[s:s + round_size, s:s + round_size] = 1
            round_arr = round_arr.reshape(self.action_size)
            valid_moves *= round_arr
        state = board.get_state()
        if board.is_empty() or self.training and np.random.rand() <= self.hyperparameters.epsilon:
            # Do epsilon choose
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
            # 不需要梯度
            action_values = self.eval_model(state)
        valid_values = np.where(valid_moves == 1, action_values.cpu().detach().numpy()[0], -np.inf)
        # return np.argmax(action_values.cpu().numpy()[0] * valid_moves)
        return np.argmax(valid_values)

    def learn(self):
        # if self.memory_counter < self.hyperparameters.batch_size:
        if len(self.memory) < 16:
            return None
        if self.learn_count % self.hyperparameters.update_target_model_each_iter == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        batch_size = min(self.hyperparameters.batch_size, len(self.memory))
        batch = self.memory.sample(batch_size)
        # print(f"Sample: {batch[0]}")
        batch = MemoryEntry(*zip(*batch))
        states = torch.from_numpy(np.array(batch.state, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(batch.action, dtype=np.int64)).to(self.device)
        rewards = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(self.device)

        non_final_mask = torch.tensor(tuple(1 if not i else 0 for i in batch.is_done)).to(self.device)
        non_final_next_states = torch.from_numpy(
            np.array(tuple(i for i in batch.next_state if i is not None), dtype=np.float32)
        ).to(self.device)

        q_eval = self.eval_model(states)
        q_target = q_eval.clone()

        batch_index = np.arange(batch_size, dtype=np.int32)
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask == 1] = self.target_model(non_final_next_states).max(1)[0]
        q_target[batch_index, actions] = rewards + self.hyperparameters.gamma * next_state_values * non_final_mask

        # fit
        q_target = q_target.detach()
        loss = 0
        for _ in range(self.hyperparameters.train_epochs):
            pred = self.eval_model(states)
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
            'hyperparameters': self.hyperparameters,
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint):
        self.eval_model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.memory = checkpoint['memory']
        self.hyperparameters = checkpoint['hyperparameters']

    def get_network_name(self):
        return "DQN2-" + self.eval_model.model_name
