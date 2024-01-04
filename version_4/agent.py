import torch
from collections import deque
from model import DNN, QTrainer  # 修改为CNN_QNet
from settings import *


class Agent:
    """
    Agent类
    代理运行和蛇
    """

    def __init__(self, game, pars=dict()):
        """
        (Agent, Snake, dict()) -> None
        初始化所有内容
        从json文件中获取传递的所有内容以修改属性和训练模型
        """
        self.n_games = 0
        self.epsilon = pars.get('eps', EPSILON)
        self.eps = pars.get('eps', EPSILON)
        self.gamma = pars.get('gamma', GAMMA)  # 折扣率
        self.eps_range = pars.get('eps_range', EPS_RANGE)
        print(self.epsilon, self.eps_range)
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # 修改为CNN_QNet
        self.model_dnn = DNN(input_size=(len(game.get_state()))**2, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
        self.trainer = QTrainer(self.model_dnn, lr=pars.get('lr', LR), gamma=self.gamma)

        self.game = game

    def remember(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        state: 当前状态
        action: 当前动作
        reward: 当前即时奖励
        next_state: 获取下一个状态
        done: 终端状态
        将所有这些属性附加到队列memory中
        每一帧都要执行这个操作
        """
        state_dnn, action, reward, next_state_dnn, done = args
        self.memory.append((state_dnn, action, reward, next_state_dnn, done))

    def train_long_memory(self):
        """
        (Agent) -> None
        在每场游戏结束后进行训练
        """
        # 获取内存
        # 如果内存超过一定的BATCH SIZE，那么
        # 随机采样BATCH SIZE的内存
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # 元组列表
        else:
            mini_sample = self.memory

        # 获取所有的状态、动作、奖励等...
        # 并使用QTrainer训练步骤
        for sample in mini_sample:
            self.trainer.train_step(*sample)

        if self.n_games % 100 == 0:
            self.trainer.plot_losses()  # 显示损失图表
            self.trainer.plot_gradient_norms()  # 顯示梯度范数圖表

    def train_short_memory(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        state: 当前状态
        action: 当前动作
        reward: 当前即时奖励
        next_state: 获取下一个状态
        done: 终端状态
        在每个游戏帧上训练代理
        """
        state_dnn, action, reward, next_state_dnn, done = args
        self.trainer.train_step(state_dnn, action, reward, next_state_dnn, done)

    def get_action(self, state_dnn):
        """
        (Agent, float) -> np.array(dtype=int): (1, 3)
        从策略或随机选择动作
        """
        # 根据epsilon和eps_range进行探索/开发的权衡
        self.epsilon = self.eps - self.n_games
        final_move = [0, 0, 0]
        # 检查是否应该随机移动
        if is_random_move(self.epsilon, self.eps_range):
            # 如果是，随机选择其中一个位
            # 来向右、左或直走
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # 否则，从NN获取最佳移动
            # 通过取其argmax并设置
            # 其位
            state0 = torch.tensor(state_dnn, dtype=torch.float)
            prediction = self.model_dnn(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
