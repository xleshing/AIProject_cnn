import torch
from collections import deque
from model import Linear_QNet, QTrainer
from settings import *


class Agent:
    """
    代理物件類別
    負責執行遊戲並控制蛇的動作
    """

    def __init__(self, game, pars=dict()):
        """
        (Agent, Snake, dict()) -> None
        初始化所有參數
        從 json 檔案中取得的參數，用於修改屬性並訓練模型
        """
        self.n_games = 0
        self.epsilon = pars.get('eps', EPSILON)  # 探索率
        self.eps = pars.get('eps', EPSILON)  # 探索率初始值
        self.gamma = pars.get('gamma', GAMMA)  # 折扣率
        self.eps_range = pars.get('eps_range', EPS_RANGE)  # 探索率遞減範圍
        print(self.epsilon, self.eps)
        self.memory = deque(maxlen=MAX_MEMORY)  # 存放經驗回放的 deque 物件
        self.model = Linear_QNet(len(game.get_state()), pars.get('hidden_size', HIDDEN_SIZE), OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=pars.get('lr', LR), gamma=self.gamma)

        self.game = game

    def remember(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        將當前的狀態、動作、獎勵、下一個狀態、是否結束的資訊加入記憶庫
        在每一個遊戲畫面更新時呼叫
        """
        state, action, reward, next_state, done = args
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        (Agent) -> None
        在每個遊戲結束後進行長期記憶訓練
        """
        # 取得記憶庫
        # 如果記憶庫大於某個 BATCH SIZE，則隨機取樣 BATCH SIZE 的記憶
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # 取得所有的狀態、動作、獎勵等資訊
        # 並透過 QTrainer 進行訓練
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        if self.n_games % 300 == 0:
            self.trainer.plot_losses()  # 顯示損失函數圖表

    def train_short_memory(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        在每個遊戲畫面更新時進行短期記憶訓練
        """
        state, action, reward, next_state, done = args
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        (Agent, float) -> np.array(dtype=int): (1, 3)
        根據策略或隨機選取動作
        """
        # 以 epsilon 和 eps_range 為基準進行探索/利用的權衡
        self.epsilon = self.eps - self.n_games
        final_move = [0, 0, 0]
        # 檢查是否應該隨機移動
        if is_random_move(self.epsilon, self.eps_range):
            # 如果是，則隨機選擇一個方向
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # 否則從神經網路中獲得最佳動作
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
