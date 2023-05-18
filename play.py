import sys

from OmegaGomoku import *
import tkinter.messagebox as msgbox
from OmegaGomoku.Play import HumanPlay

from train import board_size, win_size, model_dir

dqn = DQN(
    board_size,
    win_size,
    hyperparameters=Hyperparameters(epsilon=0),
    cuda=True
)

last_episode = dqn.last_saved_episode(model_dir)
if last_episode == 0:
    msgbox.showerror("错误", "没有训练好的模型")
    exit(0)

if len(sys.argv) > 1:
    last_episode = int(sys.argv[1])

dqn.load(last_episode, model_dir)
agent = DQNAgent(
    deep_q_network=dqn,
    writer=None,
    model_dir=model_dir
)

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