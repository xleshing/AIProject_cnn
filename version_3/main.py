import json
import os
from enum import Enum
import cv2
from agent import Agent
from game import Snake
from opencv import ColorObjectDetector
from settings import *


class Windows(Enum):
    W1 = (20, 20, 1, 1)


class Game:
    """
    游戏主类
    读取 par_lev.json 文件
    使用.json运行所有处理器
    参数
    """

    def __init__(self, lv=1):
        """
        (Game, int) -> None
        初始化游戏并设置要在其中运行的世界
        lv: 通过pygame选择的关卡
            默认设置为第一个世界
        """
        self.lv = lv
        self.awake()

    def awake(self):
        """
        (Game) -> None
        读取json文件获取带有其参数的世界
        迭代所有世界，直到找到我们初始化的世界
        创建环境并使用多进程并行运行处理器
        """
        file = open('par_lev.json', 'r')
        json_pars = json.load(file)
        file.close()

        # 获取所有枚举窗口
        for window in Windows:
            # 从json文件获取特定窗口
            pars = json_pars.get(window.name, [{}])
            # 取出窗口并解包值
            n, m, k, l = window.value
            # 设置屏幕大小为nxm个瓷砖
            n, m = (set_size(n), set_size(m))
            pars = pars[0]
            os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
            self.train(n, m, pars)

    # 将运行统计信息保存到指定路径的txt文件中
    # txt文件由build_graphs.py用于构建图表
    def save_to_file(self, path, game_num, score, record):
        """
        (Game, str, int, int, int) -> None
        将文件保存为.txt文件
        将游戏、得分、记录保存为以下格式
        g s r  分别代表游戏、得分和记录
        path: txt文件的路径
        game_num: 总代数
        score: 从游戏中取得的当前得分
        record: 最高分数
        """
        file = open(path, "a+")
        file.write("%s %s %s\n" % (game_num, score, record))
        file.close()


    def train(self, n, m, pars):
        """
        (Game, int, int, dict()) -> None
        训练游戏并以帧序列的形式运行每一步
        n: 屏幕的行瓷砖
        m: 屏幕的列瓷砖
        pars: 传递给每个处理器的参数
        """
        # 初始化
        record = 0
        game = Snake(n, m)
        obj = ColorObjectDetector()
        agent = Agent(obj, pars)
        while True:

            # 获取游戏中移动前的方向
            direction = game.get_direction()

            # 获取移动前的状态
            state_old = obj.get_state(direction=direction)

            # 从状态判断行动
            final_move = agent.get_action(state_old)

            # 移动蛇
            reward, done, score = game.play_step(action=final_move, kwargs=pars)

            # 获取游戏中移动后的方向
            direction = game.get_direction()

            # 获取移动后的状态
            state_new = obj.get_state(direction=direction)

            # 训练短期记忆
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # 记住
            agent.remember(state_old, final_move, reward, state_new, done)

            # 如果达到pars或DEFAULT_END_GAME_POINT中的num_games，则结束游戏
            # 如果设置为-1，则永远运行
            if pars.get('num_games', DEFAULT_END_GAME_POINT) != -1:
                if agent.n_games > pars.get('num_games', DEFAULT_END_GAME_POINT):
                    quit()
                    break

            # 当游戏结束时
            if done:
                # 重置游戏属性
                # 增加游戏代数
                # 训练长期记忆
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # 新高分
                if score > record:
                    record = score
                    # 保存最佳模型状态
                    agent.model.save()

                # 将游戏信息打印到控制台
                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                # 将游戏信息附加到指定路径的txt文件
                self.save_to_file(f"path/{pars.get('graph', 'test')}.txt", agent.n_games, score, record)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    Game(1)
