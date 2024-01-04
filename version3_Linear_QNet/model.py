import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Linear_QNet(nn.Module):
    """
    Linear_QNet nn.Module类
    用于表示Q值的线性神经网络模型
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        (Linear_QNet, int, int, int) -> None
        初始化模型的结构
        input_size: 游戏状态的大小
        hidden_size: 神经网络隐藏层的大小
        output_size: 输出层的大小，即蛇的动作数量
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        (Linear_QNet, *input) -> *output
        前向传播函数的重写
        添加ReLU激活函数
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        (Linear_QNet, str) -> None
        将模型状态保存到文件中
        file_name: 保存状态文件的路径
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    QTrainer类
    用于训练模型
    """

    def __init__(self, model, lr, gamma):
        """
        (QTrainer, Linear_QNet, float, float) -> None
        初始化所有模型参数
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.losses = []  # 用于记录损失

    def train_step(self, state, action, reward, next_state, done):
        """
        (QTrainer, float, long, float, float, bool) -> None
        state: 代理的当前状态
        action: 代理当前采取的动作
        reward: 当前的即时奖励
        next_state: 代理的下一个状态
        done: 终端状态的布尔值
        """
        # 转换为张量
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 使用当前状态预测的Q值
        pred = self.model(state)
        target = pred.clone()

        # 更新Q值
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        # 计算损失
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        self.losses.append(loss.item())  # 记录损失

    def plot_losses(self):
        plt.plot(self.losses, label='Loss')
        plt.title('Training Loss over Steps')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
