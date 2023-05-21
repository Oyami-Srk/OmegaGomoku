from .Environment import GomokuEnv
from .GUIBoard import GUIBoard
from .Enums import *
from .Agents import BaseAgent
from tqdm import tqdm
from tensorboardX import SummaryWriter


class SelfPlayTrainer:
    def __init__(self,
                 env: GomokuEnv,
                 agent: BaseAgent,
                 gui_enabled: bool,
                 writer: SummaryWriter | None = None):
        self.env = env
        self.agent = agent
        if gui_enabled:
            self.gui_board = GUIBoard(env.board_size, GameType.AI_VS_AI)
            self.gui_board.wait_setup()
        else:
            self.gui_board = None
        self.writer = writer
        if self.writer is None:
            w = self.agent.__getattribute__('writer')
            if w is not None:
                self.writer = w

    def train(self, episodes, start_episode=0):
        for e in tqdm(range(start_episode, episodes + 1)):
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
                        action = self.agent.act(board, self.env.current_player)
                        reward, done = self.env.step(action)
                        state = next_state
                        next_state = board.get_state()
                        memory2 = [state, next_state, action, reward, done]
                        # 如果后手胜利，则给先手惩罚
                        # if done and reward > 0:
                        #     memory1[3] = -reward
                        if reward > 0:
                            memory1[3] = -reward
                        if self.gui_board:
                            x, y = divmod(action, self.env.board_size)
                            self.gui_board.draw_piece(x, y, self.env.current_player)
                            self.gui_board.info(f"AI下{x},{y} 奖励:{reward}")
                    self.agent.remember(*memory1)
                    if memory2 is not None:
                        self.agent.remember(*memory2)
                        reward2 += 3 * memory2[3]
                    reward1 += memory1[3]

                    # 学习
                    # if steps % 2 == 0:
                    self.agent.learn(episode=e)

                if e % 100 == 0 and e != 0:
                    self.agent.save(episode=e)
                self.agent.finish(episode=e, max_episode=episodes)
                avg_reward = (reward1 / steps), (reward2 / steps)
                if self.writer is not None:
                    self.writer.add_scalar('Train/Avg_Reward1', avg_reward[0], e)
                    self.writer.add_scalar('Train/Avg_Reward2', avg_reward[1], e)
            except KeyboardInterrupt:
                # self.agent.save(f"models/gomoku_ai_{e + 1}.pt")
                break
