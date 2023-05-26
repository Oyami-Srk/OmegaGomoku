import os
import time

from .Environment import GomokuEnv, Board
from .DQN import MemoryEntry
from .GUIBoard import GUIBoard
from .Enums import *
from .Agents import BaseAgent, DQNAgent
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import deque


def evaluate_win_rate(env: GomokuEnv,
                      agent: BaseAgent,
                      rounds=50) -> tuple[float, float, float, float]:
    """
    return win_rate, draw_rate, avg_steps, avg_reward of agent against agent_rival
    """
    wins = 0
    draws = 0
    steps = 0
    rewards = 0
    for _ in range(rounds):
        board = env.reset()
        done = False
        while not done:
            steps += 1
            # 先手
            action = agent.act(board, env.current_player)
            _, reward, terminal_status = env.step(action)
            rewards += reward
            if terminal_status is not None:
                done = True
                if terminal_status == 0:
                    draws += 1
                elif terminal_status == 1:
                    wins += 1

    return wins / rounds, draws / rounds, steps / rounds, rewards / rounds


class SelfPlayTrainer:
    def __init__(self,
                 env: GomokuEnv,
                 agent: DQNAgent,
                 writer: SummaryWriter | None = None,
                 max_save_checkpoints=5):
        if not isinstance(agent, DQNAgent):
            raise Exception("Only DQNAgent Could be Trained")
        self.env = env
        self.agent = agent
        self.checkpoint_savepaths = deque()
        self.max_save_checkpoints = max_save_checkpoints
        self.writer = writer
        if self.writer is None:
            w = self.agent.__getattribute__('writer')
            if w is not None:
                self.writer: SummaryWriter = w

    def train(self, episodes, start_episode=0):
        wins = 0
        round_ = 0
        for e in tqdm(range(start_episode, episodes + 1)):
            round_ += 1
            try:
                board = self.env.reset()
                done = False
                steps = 0
                total_reward = 0
                while not done:
                    steps += 1
                    # 先手，待训练智能体
                    state = board.get_state().copy()
                    player = self.env.current_player
                    action = self.agent.act(board, player)
                    board, reward, terminal_status = self.env.step(action)
                    done = terminal_status is not None
                    next_state = None if done else board.get_state().copy()

                    total_reward += reward
                    memory = MemoryEntry(state, next_state, action, reward, done)
                    if False and done and terminal_status != 0:
                        tqdm.write(
                            f"Final states: {memory}\nTerminal Status: {['draw', 'win', 'loss'][terminal_status]}"
                        )
                    # noinspection all
                    self.agent.remember(*memory)

                    # 学习
                    self.agent.learn(episode=e)

                if e % 500 == 0 and e != 0:
                    savepath = self.agent.save(episode=e)
                    self.checkpoint_savepaths.append(savepath)
                    if len(self.checkpoint_savepaths) > self.max_save_checkpoints:
                        to_delete = self.checkpoint_savepaths.popleft()
                        try:
                            os.remove(to_delete)
                        except Exception as _:
                            tqdm.write(f"Failed to delete old checkpoint \"{to_delete}\"")
                if e % 200 == 0 and e != 0:
                    if self.writer is not None:
                        start_time = time.time()
                        win_rate, draw_rate, avg_steps, avg_rewards = evaluate_win_rate(
                            self.env.create_eval(),
                            self.agent.create_eval(),
                            rounds=50
                        )
                        end_time = time.time()
                        # self.writer.add_scalar('Train/WinRate', win_rate, e)
                        self.writer.add_scalars('Train/Evaluations/Rate', {
                            'Win Rate': win_rate,
                            'Draw Rate': draw_rate,
                            'Loss Rate': 1 - draw_rate - win_rate,
                        }, e)
                        self.writer.add_scalar('Train/Evaluations/Avg Steps', avg_steps, e)
                        self.writer.add_scalar('Train/Evaluations/Avg Rewards', avg_rewards, e)
                        # tqdm.write(f"Evaluation at Episode {e} spent {end_time - start_time} seconds.")

                self.agent.finish(episode=e, max_episode=episodes)
                avg_reward = total_reward / steps
                if self.writer is not None:
                    self.writer.add_scalar('Train/Reward', avg_reward, e)
                    self.writer.add_scalar('Train/Steps', steps, e)
            except KeyboardInterrupt:
                # self.agent.save(f"models/gomoku_ai_{e + 1}.pt")
                break
