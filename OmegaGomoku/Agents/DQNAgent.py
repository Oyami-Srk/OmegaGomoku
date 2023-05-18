import math

import numpy as np

from . import BaseAgent
from ..DQN import DQN
from tensorboardX import SummaryWriter


class DQNAgent(BaseAgent):

    def __init__(self, deep_q_network, writer: SummaryWriter | None = None, model_dir=None):
        self.dqn = deep_q_network
        self.writer = writer
        self.model_dir = model_dir

    def act(self, state: np.ndarray, valid_moves: np.ndarray):
        return self.dqn.act(state, valid_moves)

    def remember(self, state, next_state, action, reward, is_done):
        self.dqn.remember(state, next_state, action, reward, is_done)

    def learn(self, episode=None):
        assert episode is not None
        avg_loss = self.dqn.learn()
        if self.writer and avg_loss is not None:
            self.writer.add_scalar("Train/Loss", avg_loss, episode)

    def finish(self, episode=None, max_episode=None):
        assert episode is not None and max_episode is not None
        if self.writer:
            self.writer.add_scalar("Train/Epsilon", self.dqn.hyperparameters.epsilon, episode)

        e_min = self.dqn.hyperparameters.epsilon_min
        e_max = self.dqn.hyperparameters.epsilon_max
        e_rate_exp = self.dqn.hyperparameters.epsilon_decay_rate_exp

        # Decay Epsilon
        def f(_e, _er):
            # return 1 - (e_max * (1 - 0.7 ** (epi / 800 + 1)))
            # return 0.995 ** (episode / (max_episode / 700))
            # return 0.995 ** (episode / 18)
            return e_min + (e_max - e_min) * math.exp(-1. * episode / e_rate_exp)

        self.dqn.decay_epsilon(f)

    def save(self, episode=None):
        if episode:
            self.dqn.save(episode, self.model_dir)
