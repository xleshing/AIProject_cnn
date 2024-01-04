import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


class TwoBranch_QNet(nn.Module):
    def __init__(self, input_size_group1, input_size_group2, hidden_sizes, output_size):
        super(TwoBranch_QNet, self).__init__()
        self.branch1 = self._build_branch(input_size_group1, hidden_sizes)
        self.branch1 = self.branch1.to("cuda:0")
        self.branch2 = self._build_branch(input_size_group2, hidden_sizes)
        self.branch2 = self.branch2.to("cuda:0")
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self._initialize_weights()

    def _build_branch(self, input_size, hidden_sizes):
        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            self._initialize_weights(layers[-2])
        return nn.Sequential(*layers)

    def _initialize_weights(self, layer=None):
        if layer is None:
            layer = self.output_layer
            layer = layer.to("cuda:0")
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # 使用 He 初始化

    def forward(self, x1, x2):
        x1 = x1.to("cuda:0")
        x2 = x2.to("cuda:0")
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x = x1 + x2  # 逐元素相加
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
        self.losses = []  # 用于记录损失

    def train_step(self, state_group1, state_group2, action, reward, next_state_group1, next_state_group2, done):
        state_group1, state_group2, next_state_group1, next_state_group2, action, reward = (
            torch.tensor(state_group1, dtype=torch.float),
            torch.tensor(state_group2, dtype=torch.float),
            torch.tensor(next_state_group1, dtype=torch.float),
            torch.tensor(next_state_group2, dtype=torch.float),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float),
        )

        if len(state_group1.shape) == len(state_group2.shape) == 1:
            state_group1, state_group2, next_state_group1, next_state_group2, action, reward, done = (
                torch.unsqueeze(state_group1, 0),
                torch.unsqueeze(state_group2, 0),
                torch.unsqueeze(next_state_group1, 0),
                torch.unsqueeze(next_state_group2, 0),
                torch.unsqueeze(action, 0),
                torch.unsqueeze(reward, 0),
                (done,),
            )

        state_group1, state_group2, next_state_group1, next_state_group2, action, reward = (
            state_group1.to("cuda:0"),
            state_group2.to("cuda:0"),
            next_state_group1.to("cuda:0"),
            next_state_group2.to("cuda:0"),
            action.to("cuda:0"),
            reward.to("cuda:0"),
        )

        pred = self.model(state_group1, state_group2)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state_group1[idx], next_state_group2[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        """
        Q_new = reward + self.gamma * torch.max(self.model(next_state_group1, next_state_group2), dim=1)[0]
        target[range(len(done)), action] = Q_new"""

        self.optimizer.zero_grad()
        loss = F.mse_loss(target, pred)
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
