from .Environment import GomokuEnv
from .GUIBoard import GUIBoard
from .Enums import *
from .Agents import BaseAgent


class HumanPlay:
    def __init__(self,
                 env: GomokuEnv,
                 agent: BaseAgent):
        self.env = env
        self.agent = agent
        self.gui_board = GUIBoard(env.board_size, GameType.PLAYER_VS_AI)
        self.gui_board.wait_setup()

    def play(self, on_done_callback):
        state = self.env.reset()
        self.gui_board.reset()
        self.gui_board.on_close(on_done_callback)

        done = False
        steps = 0
        reward = 0
        while not done:
            steps += 1
            action = self.agent.act(state, self.env.get_valid_moves())
            current_player = self.env.current_player
            next_state, reward, done = self.env.step(action)
            state = next_state
            # print("Player {} takes action: {}".format(self.env.current_player, action))
            x, y = divmod(action, self.env.board_size)
            self.gui_board.draw_piece(x, y, current_player)
            self.gui_board.info(f"AI下{x},{y} 奖励:{reward}")
            print(f"AI下{x},{y} 奖励:{reward}")
            if not done:
                # Player move
                action = self.gui_board.wait_place()
                if action is None:
                    return "Quit"
                # x, y, player = self.gui_board.wait_place()
                x, y, player = action
                current_player = self.env.current_player
                steps += 1
                next_state, reward, done = self.env.step((x, y))
                self.gui_board.draw_piece(x, y, current_player)
                print(f"Player下{x}, {y} 奖励:{reward}")
                state = next_state
            else:
                return "AI win" if reward > 0 else "Draw"
            print("\n")
        return "You win" if reward > 0 else "Draw"

    def __del__(self):
        self.gui_board.wait()
