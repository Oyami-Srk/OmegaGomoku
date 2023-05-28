import math

PATTERNS_REWARD = {
    'L5': 200,
    'H4': 100,
    'C4': 60,
    'H3': 30,
    'M3': 20,
    'H2': 10,
    'M2': 5
}

PATTERNS_GAMMA = {
    'rival': [0.3, 0.5, 0.8],
    'break': [0.3, 0.5, 1]
}

FINAL_STATUS = {
    'win': 800,
    'loss': -200,
    'draw': -20
}


def calculate_reward(pattern: str | None,
                     break_pattern: str | None,
                     rival_pattern: str | None,
                     terminal_status: int | None) -> float:
    if pattern is None and break_pattern is None and rival_pattern is None:
        return 0
    if terminal_status is not None:
        if terminal_status == 0:
            return FINAL_STATUS['draw']
        elif terminal_status == 1:
            return FINAL_STATUS['win']
        else:
            return FINAL_STATUS['loss']
    reward = 0
    if pattern is not None:
        # 构成了特定的棋形
        reward += PATTERNS_REWARD[pattern]
    if rival_pattern is not None:
        # 对手形成了棋形
        rival_reward = PATTERNS_REWARD[rival_pattern]
        reward -= rival_reward * (
            PATTERNS_GAMMA['rival'][0] if rival_reward <= reward
            else PATTERNS_GAMMA['rival'][1] if rival_reward <= 60
            else PATTERNS_GAMMA['rival'][2])
    if break_pattern is not None:
        # 打破了对手的棋形
        assume_reward = PATTERNS_REWARD[break_pattern]
        reward += assume_reward * (
            PATTERNS_GAMMA['break'][0] if assume_reward <= 20
            else PATTERNS_GAMMA['break'][1] if assume_reward <= 60
            else PATTERNS_GAMMA['break'][2])
    return reward
