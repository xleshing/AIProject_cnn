import random

# 蛇的大小
SIZE = 20

# 設定大小的lambda函數
set_size = lambda x: SIZE * x

# 默認遊戲結束帧数和速度
DEFAULT_KILL_FRAME = 100

# 神經網絡配置
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

GAMMA = 0.9

EPSILON = 80

EPS_RANGE = (0, 200)
is_random_move = lambda eps, eps_range: random.randint(eps_range[0], eps_range[1]) < eps

# 默認遊戲結束點
DEFAULT_END_GAME_POINT = 300
