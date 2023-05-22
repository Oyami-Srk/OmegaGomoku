import os

from .Environment import GomokuEnv
from .GUIBoard import GUIBoard
from .Enums import *
from .Agents import BaseAgent
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import deque


class SelfPlayTrainer:
    def __init__(self,
                 env: GomokuEnv,
                 agent: BaseAgent,
                 agent_rival: BaseAgent | None,
                 gui_enabled: bool,
                 writer: SummaryWriter | None = None,
                 max_save_checkpoints=5):
        self.env = env
        self.agent = agent
        self.agent_rival = agent_rival if agent_rival is not None else agent
        self.checkpoint_savepaths = deque()
        self.max_save_checkpoints = max_save_checkpoints
        if gui_enabled:
            self.gui_board = GUIBoard(env.board_size, GameType.AI_VS_AI)
            self.gui_board.wait_setup()
        else:
            self.gui_board = None
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
                if self.gui_board:
                    self.gui_board.reset()
                    self.gui_board.title(f"第{e}轮")
                board = self.env.reset()
                done = False
                steps = 0
                reward1 = 0
                reward2 = 0
                while not done:
                    steps += 1
                    # 先手
                    state = board.get_state()
                    action = self.agent.act(board, self.env.current_player)
                    reward, done = self.env.step(action)
                    next_state = board.get_state()
                    # 推迟记忆
                    memory1 = [state, next_state, action, reward, done]
                    # self.agent.remember(state, next_state, action, reward, done)
                    memory2 = None
                    if self.gui_board:
                        x, y = divmod(action, self.env.board_size)
                        self.gui_board.draw_piece(x, y, self.env.current_player)
                        self.gui_board.info(f"AI下{x},{y} 奖励:{reward}")
                    if not done:
                        # 后手
                        # steps += 1
                        action = self.agent_rival.act(board, self.env.current_player)
                        reward, done = self.env.step(action)
                        state = next_state
                        next_state = board.get_state()
                        memory2 = [state, next_state, action, reward, done]
                        # 如果后手胜利，则给先手惩罚
                        # if done and reward > 0:
                        #     memory1[3] = -reward
                        if reward > 0:
                            if reward <= 50:
                                memory1[3] -= reward * 0.5
                            else:
                                memory1[3] -= reward
                        if self.gui_board:
                            x, y = divmod(action, self.env.board_size)
                            self.gui_board.draw_piece(x, y, self.env.current_player)
                            self.gui_board.info(f"AI下{x},{y} 奖励:{reward}")
                    else:
                        wins += 1
                    self.agent.remember(*memory1)
                    if memory2 is not None:
                        self.agent.remember(*memory2)
                        reward2 += memory2[3]
                    reward1 += memory1[3]

                    # 学习
                    # if steps % 2 == 0:
                    self.agent.learn(episode=e)

                if e % 500 == 0 and e != 0:
                    savepath = self.agent.save(episode=e)
                    self.checkpoint_savepaths.append(savepath)
                    if len(self.checkpoint_savepaths) > self.max_save_checkpoints:
                        to_delete = self.checkpoint_savepaths.popleft()
                        try:
                            os.remove(to_delete)
                        except Exception as e:
                            tqdm.write(f"Failed to delete old checkpoint \"{to_delete}\"")
                if e % 50 == 0 and e != 0:
                    win_rate = wins / round_
                    wins = 0
                    round_ = 0
                    if self.writer is not None:
                        self.writer.add_scalar('Train/WinRate', win_rate, e)
                self.agent.finish(episode=e, max_episode=episodes)
                avg_reward = (reward1 / steps), (reward2 / steps)
                if self.writer is not None:
                    self.writer.add_scalars('Train/Reward', {
                        'self': avg_reward[0],
                        'rival': avg_reward[1]
                    }, e)
                    self.writer.add_scalar('Train/Steps', steps, e)
            except KeyboardInterrupt:
                # self.agent.save(f"models/gomoku_ai_{e + 1}.pt")
                break
