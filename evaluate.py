import sys

from OmegaGomoku import *
from train import board_size, win_size, model_dir
from tqdm import tqdm


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
        raise Exception("没有训练好的模型")

    if len(sys.argv) > 1:
        last_episode = int(sys.argv[1])

    print(f"Loading Episode: {last_episode}")
    dqn.load(last_episode, model_dir)
    agent = DQNAgent(deep_q_network=dqn, writer=None, model_dir=model_dir)
    return dqn, agent


dqn, agent = make_dqn_agent()
rival_agent = MiniMaxAgent(board_size, depth=2)
env = GomokuEnv(
    rival_agent=rival_agent,
    board_size=board_size,
    win_size=win_size,
)

evaluate_rounds = 100
win = 0
loss = 0
steps = 0
rewards = 0

for _ in tqdm(range(evaluate_rounds)):
    board = env.reset()
    done = False
    terminal_status = None
    while not done:
        steps += 1
        # 先手，AI智能体
        player = env.current_player
        action = agent.act(board, player)
        board, reward, terminal_status = env.step(action)
        done = terminal_status is not None
        rewards += reward
        if done:
            if terminal_status == 1:
                win += 1
            elif terminal_status == -1:
                loss += 1

avg_step = steps / evaluate_rounds
avg_reward = rewards / steps
win_rate = win / evaluate_rounds
loss_rate = loss / evaluate_rounds
draw_rate = 1.0 - win_rate - loss_rate

print(f"""
===============Evaluation Result================
Average Steps per Round: {avg_step}
Average Reward per Step: {avg_reward}
Win Rate: {win_rate}
Loss Rate: {loss_rate}
Draw Rate: {draw_rate}
Non-Loss Rate: {1.0 - loss_rate}
""")
