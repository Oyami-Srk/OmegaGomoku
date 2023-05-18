from .Environment import GomokuEnv
from .GUIBoard import GUIBoard
from .Enums import *
from .Agents import BaseAgent
from tqdm import tqdm


class SelfPlayTrainer:
    def __init__(self,
                 env: GomokuEnv,
                 agent: BaseAgent,
                 gui_enabled: bool):
        self.env = env
        self.agent = agent
        if gui_enabled:
            self.gui_board = GUIBoard(env.board_size, GameType.AI_VS_AI)
            self.gui_board.wait_setup()
        else:
            self.gui_board = None

    def train(self, episodes, start_episode=0):
        for e in tqdm(range(start_episode, episodes + 1)):
            try:
                if self.gui_board:
                    self.gui_board.reset()
                    self.gui_board.title(f"第{e}轮")
                state = self.env.reset()
                done = False
                steps = 0
                while not done:
                    steps += 1
                    # 先手
                    action = self.agent.act(state, self.env.get_valid_moves())
                    next_state, reward, done = self.env.step(action)
                    # 推迟记忆
                    memory1 = [state, next_state, action, reward, done]
                    # self.agent.remember(state, next_state, action, reward, done)
                    state = next_state
                    memory2 = None
                    if not done:
                        # 后手
                        # steps += 1
                        action = self.agent.act(state, self.env.get_valid_moves())
                        next_state, reward, done = self.env.step(action)
                        memory2 = [state, next_state, action, reward, done]
                        state = next_state
                        # 如果后手胜利，则给先手惩罚
                        if done:
                            memory1[3] = -1
                    self.agent.remember(*memory1)
                    if memory2 is not None:
                        self.agent.remember(*memory2)

                    # 学习
                    # if steps % 2 == 0:
                    self.agent.learn(episode=e)

                    if self.gui_board:
                        x, y = divmod(action, self.env.board_size)
                        self.gui_board.draw_piece(x, y, self.env.current_player)
                        self.gui_board.info(f"AI下{x},{y} 奖励:{reward}")

                if e % 500 == 0 and e != 0:
                    self.agent.save(episode=e)
                self.agent.finish(episode=e, max_episode=episodes)
            except KeyboardInterrupt:
                # self.agent.save(f"models/gomoku_ai_{e + 1}.pt")
                break
