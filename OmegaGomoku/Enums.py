class Player:
    EMPTY = 0  # 空白
    BLACK = 1  # 黑方, O
    WHITE = 2  # 白方, X

    @staticmethod
    def SwitchPlayer(player: int) -> int:
        # return [0, -1, 1][player]
        return 3 - player

    @staticmethod
    def PlayerSymbol(player: int) -> str:
        return ['.', 'O', 'X'][player]


class GameType:
    PLAYER_VS_AI = 1  # 人机对弈模式
    AI_VS_AI = 2  # AI自对弈模式
    PLAYER_VS_PLAYER = 3


class TermColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
