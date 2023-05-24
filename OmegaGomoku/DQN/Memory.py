from collections import deque, namedtuple
from ..Environment import Board

import numpy as np
import torch
import random


class MemoryEntry(namedtuple('MemoryEntry', ('state', 'next_state', 'action', 'reward', 'is_done'))):
    def __str__(self):
        board_size = self.state.shape[-1]
        state = Board.beautify_board(self.state[0] + self.state[1] * 2)
        boards = "\n"
        if self.next_state is None:
            boards += state
        else:
            next_state = Board.beautify_board(self.next_state[0] + self.next_state[1] * 2)
            state = state.split('\n')
            if next_state is not None:
                next_state = next_state.split('\n')
            center = len(state) // 2
            arrow = '==>'
            for i in range(len(state)):
                boards += f"{state[i]}\t{arrow if i == center else ''}\t{next_state[i]}\n"
        boards += f"Action: {divmod(self.action, board_size)}; Reward: {self.reward}; Terminal: {self.is_done}"
        return boards


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self,
             state: np.ndarray,
             next_state: np.ndarray | None,
             action: int | tuple[int, int],
             reward: int | float,
             is_done: bool):
        entry = MemoryEntry(state, next_state, action, reward, is_done)
        self.memory.append(entry)

    def sample(self, batch_size, replace=False) -> list[MemoryEntry]:
        # random.sample is slow when batch size too big
        return random.sample(self.memory, batch_size)
        # return np.random.choice(self.memory, batch_size, replace)

    def __len__(self):
        return len(self.memory)
