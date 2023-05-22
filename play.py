import sys

from OmegaGomoku import *
import tkinter.messagebox as msgbox
from OmegaGomoku.Play import HumanPlay

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


dqn, agent = make_dqn_agent()
# agent = MiniMaxAgent(board_size)

p = HumanPlay(
    GomokuEnv(board_size=board_size, win_size=win_size),
    agent
)

done = False


def on_done():
    global done
    done = True


while not done:
    result = p.play(on_done)
    try:
        msgbox.showinfo("游戏结束", result)
    except:
        pass
