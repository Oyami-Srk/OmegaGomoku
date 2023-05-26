import shutil
import sys
import time

from OmegaGomoku import SelfPlayTrainer, GomokuEnv
from OmegaGomoku.Agents import DQNAgent, MiniMaxAgent
from OmegaGomoku.DQN import DQN
from OmegaGomoku.Hyperparameters import Hyperparameters

from tensorboardX import SummaryWriter

board_size = 7
win_size = 5
target_episode = 50000
model_dir = "models"

"""
note:
batch_size=256, memory_size=20000, learning_rate=1e-06, gamma=0.65, epsilon=1.0, epsilon_decay_rate=0.995, epsilon_min=0.2, epsilon_max=1.0, epsilon_decay_rate_exp=2000, swap_model_each_iter=100, train_epochs=20, tau=0.005, loss=MSELoss, optimizer=Adam, 
"""

hyperparameters = Hyperparameters(
    batch_size=256,
    memory_size=20000,  # 记忆空间大小
    learning_rate=1e-5,  # 学习率
    gamma=0.50,  # 奖励折扣因子，越高的话智能体会倾向于长期价值
    epsilon=1.0,  # 探索率，探索率越高随机探索的可能性越大
    epsilon_decay_rate=0.995,  # 探索率衰减率
    epsilon_min=0.2,  # 最小探索率
    epsilon_max=1.00,  # 最大探索率
    epsilon_decay_rate_exp=1000,  # 探索率指数衰减参数，越高越慢，e = e_min + (e_max - e_min) * exp(-1.0 * episode / rate)
    update_target_model_each_iter=200,  # 每学习N次更新Target模型
    # train_epochs=20,
    train_epochs=1,
    tau=0.005,
    loss='SmoothL1Loss',
    optimizer='AdamW'
)

if __name__ == '__main__':
    log_dir = f"log/{time.strftime('%Y-%m-%d %H-%M-%S')}"
    writer = SummaryWriter(log_dir)
    print(f"Tensorboard Log write to {log_dir}")

    deep_q_network = DQN(
        board_size=board_size,
        win_size=win_size,
        hyperparameters=hyperparameters,
        cuda=True,
        training=True
    )

    if sys.argv[-1] == "continue":
        last_episode = deep_q_network.last_saved_episode(model_dir)
        if last_episode != 0:
            print(f"Resume from {last_episode}")
            deep_q_network.load(last_episode, model_dir)
            target_episode += last_episode
    else:
        # os.removedirs(model_dir)
        # os.removedirs(log_dir)
        save_dir = deep_q_network.make_savepath(model_dir, no_make=True)
        if save_dir is not None:
            shutil.rmtree(save_dir, True)
        # shutil.rmtree(log_dir, True)
        last_episode = 0

    trainer = SelfPlayTrainer(
        env=GomokuEnv(
            rival_agent=MiniMaxAgent(board_size, depth=1),
            board_size=board_size,
            win_size=win_size
        ),
        agent=DQNAgent(
            deep_q_network=deep_q_network,
            writer=writer,
            model_dir=model_dir
        )
    )

    trainer.train(target_episode, start_episode=last_episode)
