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


class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(4)  # 批标准化层
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(8)  # 批标准化层
        self.fc1 = nn.Linear(200, output_size)

        # 使用He初始化
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)  # 添加批标准化
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)  # 添加批标准化
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = (self.fc1(x))
        return x

    def save(self, file_name='model_cnn.pth'):
        """
        (CNN_QNet, str) -> None
        file_name: 保存状态文件的路径
        将模型状态保存到file_name
        """
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

    def __init__(self, model_cnn, lr, gamma):
        """
        (QTrainer, CNN_QNet, float, float) -> None
        初始化所有模型参数
        """
        self.lr = lr
        self.gamma = gamma
        self.model_cnn = model_cnn
        self.opt_RAdam = RAdam(model_cnn.parameters(), lr=self.lr)
        self.optimizer = Lookahead(self.opt_RAdam, k=5, alpha=0.5)
        self.criterion = nn.MSELoss()
        self.losses = []  # 用于记录损失
        self.gradient_norms = []  # 用于记录梯度范数

    def train_step(self, state_cnn, action, reward, next_state_cnn, done):
        """
        (QTrainer, float, long, float, float, bool) -> None
        state: 代理的当前状态
        action: 代理当前采取的动作
        reward: 当前即时奖励
        next_state: 代理的下一个状态
        done: 终端布尔值
        """
        # turn into tensor and add batch dimension
        state_cnn = torch.tensor(state_cnn, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        next_state_cnn = torch.tensor(next_state_cnn, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        # 使用当前状态预测的Q值
        pred = self.model_cnn(state_cnn)
        target = pred.clone()

        # update Q values
        Q_new = reward
        if not done:
            Q_new = reward + self.gamma * torch.max(
                self.model_cnn(next_state_cnn))

        target[0][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        # 计算损失
        loss = self.criterion(target, pred)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model_cnn.parameters(), max_norm=1.0)

        # 记录梯度范数
        for param in self.model_cnn.parameters():
            if param.grad is not None:
                self.gradient_norms.append(param.grad.norm().item())

        self.optimizer.step()

        self.losses.append(loss.item())  # 记录损失

    def plot_losses(self):
        plt.plot(self.losses, label='Loss')
        plt.title('Training Loss over Steps')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_gradient_norms(self):
        plt.plot(self.gradient_norms, label='Gradient Norm')
        plt.title('Gradient Norm over Steps')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.show()
