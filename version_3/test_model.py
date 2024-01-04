import torch
from model import Linear_QNet
from settings import *
import os


class test:
    def __init__(self):
        # 加载已经训练好的模型权重
        self.model = Linear_QNet(11, HIDDEN_SIZE, OUTPUT_SIZE)
        model_folder_path = './model'
        file_name = 'model.pth'
        model_path = os.path.join(model_folder_path, file_name)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()  # 设置模型为评估模式

    def get_action(self, state):
        final_move = [0, 0, 0]
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            return final_move

