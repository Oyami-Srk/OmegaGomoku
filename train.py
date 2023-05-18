import shutil
import sys
import time

from OmegaGomoku import SelfPlayTrainer, GomokuEnv
from OmegaGomoku.Agents import DQNAgent
from OmegaGomoku.DQN import DQN
from OmegaGomoku.Hyperparameters import Hyperparameters

from tensorboardX import SummaryWriter

board_size = 4
win_size = 3
target_episode = 1000
model_dir = "models"
log_dir = f"log/{time.strftime('%Y-%m-%d %H-%M-%S')}"
writer = SummaryWriter(log_dir)
print(f"Tensorboard Log write to {log_dir}")

hyperparameters = Hyperparameters(
    batch_size=256,
    memory_size=20000,  # 记忆空间大小
    learning_rate=1e-8,  # 学习率
    gamma=0.95,  # 奖励折扣因子
    epsilon=1.0,  # 探索率，探索率越高随机探索的可能性越大
    epsilon_decay_rate=0.995,  # 探索率衰减率
    epsilon_min=0.05,  # 最小探索率
    epsilon_max=1.00,  # 最大探索率
    epsilon_decay_rate_exp=1000,  # 探索率指数衰减参数，e = e_min + (e_max - e_min) * exp(-1.0 * episode / rate)
    swap_model_each_iter=300,  # 每学习N次交换Target和Eval模型
    train_epochs=64,
    tau=0.005
)

if __name__ == '__main__':
    deep_q_network = DQN(
        board_size=board_size,
        win_size=win_size,
        hyperparameters=hyperparameters,
        cuda=True
    )

    if sys.argv[-1] == "continue":
        last_episode = deep_q_network.last_saved_episode(model_dir)
        if last_episode != 0:
            print(f"Resume from {last_episode}")
            deep_q_network.load(last_episode, model_dir)
    else:
        # os.removedirs(model_dir)
        # os.removedirs(log_dir)
        save_dir = deep_q_network.make_savepath(model_dir, no_make=True)
        if save_dir is not None:
            shutil.rmtree(save_dir, True)
        # shutil.rmtree(log_dir, True)
        last_episode = 0

    trainer = SelfPlayTrainer(
        env=GomokuEnv(board_size, win_size),
        agent=DQNAgent(
            deep_q_network=deep_q_network,
            writer=writer,
            model_dir=model_dir
        ),
        gui_enabled=False
    )

    trainer.train(target_episode, start_episode=last_episode)
