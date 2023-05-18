class Hyperparameters:
    batch_size = 128
    memory_size = 500  # 记忆空间大小
    learning_rate = 1e-8  # 学习率
    gamma = 0.95  # 奖励折扣因子
    epsilon = 1.0,  # 探索率，探索率越高随机探索的可能性越大
    epsilon_decay_rate = 0.995,  # 探索率衰减率
    epsilon_min = 0.05,  # 最小探索率
    epsilon_max = 1.00,  # 最大探索率
    epsilon_decay_rate_exp = 1000,  # 探索率指数衰减参数，e = e_min + (e_max - e_min) * exp(-1.0 * episode / rate)
    swap_model_each_iter = 300  # 每学习N次交换Target和Eval模型
    train_epochs = 20
    tau = 0.005

    def __init__(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
