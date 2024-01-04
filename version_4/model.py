import torch
import torch.nn as nn
from torch_optimizer import Lookahead
import torch.nn.functional as F
import os
import matplotlib
from radam import RAdam
import torch.nn.init as init
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 使用He初始化
        init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model_dnn.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    QTrainer类
    训练模型
    """

    def __init__(self, model_dnn, lr, gamma):
        """
        (QTrainer, CNN_QNet, float, float) -> None
        初始化所有模型参数
        """
        self.lr = lr
        self.gamma = gamma
        self.model_dnn = model_dnn
        self.opt_RAdam = RAdam(model_dnn.parameters(), lr=self.lr)
        self.optimizer = Lookahead(self.opt_RAdam, k=5, alpha=0.5)
        self.criterion = nn.MSELoss()
        self.losses = []  # 用于记录损失
        self.gradient_norms = []  # 用于记录梯度范数

    def train_step(self, state_dnn, action, reward, next_state_dnn, done):
        """
        (QTrainer, float, long, float, float, bool) -> None
        state: 代理的当前状态
        action: 代理当前采取的动作
        reward: 当前即时奖励
        next_state: 代理的下一个状态
        done: 终端布尔值
        """
        # 转换为张量
        state_dnn = torch.tensor(state_dnn, dtype=torch.float)
        next_state_dnn = torch.tensor(next_state_dnn, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state_dnn.shape) == 1:
            state_dnn = torch.unsqueeze(state_dnn, 0)
            next_state_dnn = torch.unsqueeze(next_state_dnn, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 使用当前状态预测的Q值
        pred = self.model_dnn(state_dnn)
        target = pred.clone()

        # update Q values
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model_dnn(next_state_dnn[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        # 计算损失
        loss = self.criterion(target, pred)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model_dnn.parameters(), max_norm=1.0)

        # 记录梯度范数
        for param in self.model_dnn.parameters():
            if param.grad is not None:
                self.gradient_norms.append(param.grad.norm().item())

        self.optimizer.step()

        self.losses.append(loss.item())  # 记录损失

    def plot_losses(self):
        plt.plot(self.losses, label='Loss')
        plt.title('训练步数下的损失')
        plt.xlabel('训练步数')
        plt.ylabel('损失')
        plt.legend()
        plt.show()

    def plot_gradient_norms(self):
        plt.plot(self.gradient_norms, label='Gradient Norm')
        plt.title('梯度范数随训练步数变化')
        plt.xlabel('训练步数')
        plt.ylabel('梯度范数')
        plt.legend()
        plt.show()
