import sys

from OmegaGomoku import *
import tkinter.messagebox as msgbox

from train import board_size, win_size, model_dir


def make_dqn_agent():
    dqn = DQN(
        board_size,
        win_size,
        hyperparameters=Hyperparameters(epsilon=0),
        cuda=True,
        training=False
    )

    last_episode = dqn.last_saved_episode(model_dir)
    if last_episode == 0:
        msgbox.showerror("错误", "没有训练好的模型")
        exit(0)

    if len(sys.argv) > 1:
        last_episode = int(sys.argv[1])

    print(f"Loading Episode: {last_episode}")
    dqn.load(last_episode, model_dir)
    agent = DQNAgent(deep_q_network=dqn, writer=None, model_dir=model_dir)
    return dqn, agent


gui_board = GUIBoard(board_size, mode=GameType.PLAYER_VS_AI)
dqn, agent = make_dqn_agent()
human_agent = HumanAgent(gui_board=gui_board)
env = GomokuEnv(
    rival_agent=human_agent,
    board_size=board_size,
    win_size=win_size,
    gui_board=gui_board
)

quit = False

while not quit:
    board = env.reset()
    done = False
    steps = 0
    terminal_status = 0
    try:
        while not done:
            steps += 1
            # 先手，AI智能体
            player = env.current_player
            action = agent.act(board, player)
            board, reward, terminal_status = env.step(action)
            done = terminal_status is not None

        try:
            msgbox.showinfo("游戏结束",
                            "AI获胜" if terminal_status == 1 else "人类获胜" if terminal_status == -1 else "平局")
        except Exception as e:
            print(e)
    except Exception as e:
        if str(e) == "User Exit":
            quit = True
            print("Bye.")
        else:
            print(e)
