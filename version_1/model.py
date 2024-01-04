import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class MultiLayer_QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayer_QNet, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # 添加第一隐藏层，使用SELU激活函数
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.hidden_layers.append(nn.SELU())

        for i in range(1, len(hidden_sizes)):
            # 添加后续隐藏层，使用SELU激活函数
            self.hidden_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.hidden_layers.append(nn.SELU())

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.losses = []  # 用于记录损失
        self.gradient_norms = []  # 用于记录梯度范数

    def train_step(self, state, action, reward, next_state, done):
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

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # 记录梯度范数
        for param in self.model.parameters():
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
        plt.title('Gradient Norms over Training Steps')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.show()
