import pygame
import sys
from settings import *


class DefaultImediateReward:
    """
    DefaultImediateReward 类
    这是蛇的默认即时奖励类
    """
    COLLISION_WALL = -10
    COLLISION_SELF = -10
    LOOP = -10
    SCORED = 10
    VERY_FAR_FROM_FOOD = 0

class Snake:
    def __init__(self, w, h):
        # 初始化 Pygame
        pygame.init()
        self.TITLE = "g1"
        self.point = False

        # 设置颜色
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)

        # 设置蛇的初始速度和大小
        self.snake_size = SIZE
        self.snake_speed = 10

        self.w = w
        self.h = h

        # 设置食物的初始位置和大小
        self.food_size = self.snake_size

        # 定义蛇的移动方向
        self.UP = 'UP'
        self.DOWN = 'DOWN'
        self.LEFT = 'LEFT'
        self.RIGHT = 'RIGHT'
        self.direction = self.RIGHT
        self.reset()

        # 创建游戏窗口
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(self.TITLE)

    def reset(self):
        self.direction = self.RIGHT
        self.play_num = 0

        # 定义食物的初始位置
        self.food_position = (random.randrange(1, (self.w // self.snake_size)) * self.snake_size,
        random.randrange(1, (self.h // self.snake_size)) * self.snake_size)

        # 定义蛇的初始位置
        self.snake = [(self.w // 2, self.h // 2)]

        self.score = 0

    def draw_snake_and_food(self):
        # 画出食物
        pygame.draw.rect(self.screen, self.red,
                         pygame.Rect(self.food_position[0], self.food_position[1], self.food_size, self.food_size))

        # 画出蛇的头部
        head_x, head_y = self.snake[0]
        pygame.draw.rect(self.screen, self.green, pygame.Rect(head_x, head_y, self.snake_size, self.snake_size))

        # 画出蛇的身体
        for pos in self.snake[1:]:
            pygame.draw.rect(self.screen, self.white, pygame.Rect(pos[0], pos[1], self.snake_size, self.snake_size))
            pygame.draw.rect(self.screen, self.black, pygame.Rect(pos[0], pos[1], self.snake_size, self.snake_size), 2)

    def check_collision(self):
        head_x, head_y = self.snake[0]

        # 检查是否碰到墙壁
        if head_x < 0 or head_x >= self.w or head_y < 0 or head_y >= self.h:
            return True

        # 检查是否碰到自己的身体
        if (head_x, head_y) in self.snake[1:]:
            return True

        return False

    def control(self, action):
        if action[2] and self.direction != self.DOWN and self.direction != self.UP:
            self.direction = self.UP
        elif action[3] and self.direction != self.UP and self.direction != self.DOWN:
            self.direction = self.DOWN
        elif action[0] and self.direction != self.RIGHT and self.direction != self.LEFT:
            self.direction = self.LEFT
        elif action[1] and self.direction != self.LEFT and self.direction != self.RIGHT:
            self.direction = self.RIGHT

    def change_control(self, action):
        if self.direction == self.LEFT:
            if action[0]:
                return [1, 0, 0, 0]
            elif action[1]:
                return [0, 0, 1, 0]
            elif action[2]:
                return [0, 0, 0, 1]
        elif self.direction == self.RIGHT:
            if action[0]:
                return [0, 1, 0, 0]
            elif action[1]:
                return [0, 0, 0, 1]
            elif action[2]:
                return [0, 0, 1, 0]
        elif self.direction == self.UP:
            if action[0]:
                return [0, 0, 1, 0]
            elif action[1]:
                return [0, 1, 0, 0]
            elif action[2]:
                return [1, 0, 0, 0]
        elif self.direction == self.DOWN:
            if action[0]:
                return [0, 0, 0, 1]
            elif action[1]:
                return [1, 0, 0, 0]
            elif action[2]:
                return [0, 1, 0, 0]

    def move(self):
        # 移动蛇的尾部（从尾巴开始）
        for i in range(len(self.snake) - 1, 0, -1):
            self.snake[i] = (self.snake[i - 1][0], self.snake[i - 1][1])

        # 移动蛇的头部
        if self.direction == self.UP:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] - self.snake_size)
        elif self.direction == self.DOWN:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] + self.snake_size)
        elif self.direction == self.LEFT:
            self.snake[0] = (self.snake[0][0] - self.snake_size, self.snake[0][1])
        elif self.direction == self.RIGHT:
            self.snake[0] = (self.snake[0][0] + self.snake_size, self.snake[0][1])

    def get_direction(self):
        if self.direction == self.LEFT:
            return [1, 0, 0, 0]
        if self.direction == self.RIGHT:
            return [0, 1, 0, 0]
        if self.direction == self.UP:
            return [0, 0, 1, 0]
        if self.direction == self.DOWN:
            return [0, 0, 0, 1]

    def draw_border(self):
        # 在边界画一条线
        pygame.draw.line(self.screen, self.blue, (0, 0), (self.w, 0), 1)
        pygame.draw.line(self.screen, self.blue, (0, 0), (0, self.h), 1)
        pygame.draw.line(self.screen, self.blue, (self.w - 1, 0), (self.w - 1, self.h), 1)
        pygame.draw.line(self.screen, self.blue, (0, self.h - 1), (self.w, self.h - 1), 1)

    def get_point(self):
        # 在检查是否吃到食物的部分
        if self.snake[0][0] == self.food_position[0] and self.snake[0][1] == self.food_position[1]:
            self.point = True
            while True:
                new_food_position = (random.randrange(1, (self.w // self.snake_size)) * self.snake_size,
                                     random.randrange(1, (self.h // self.snake_size)) * self.snake_size)

                # 检查新生成的食物位置是否与蛇的身体重叠
                if new_food_position not in self.snake:
                    break

            self.food_position = new_food_position

            # 在蛇的尾部新增一个段落
            self.snake.append(self.snake[-1])

    def play_step(self, action, kwargs=None):
        if kwargs is None:
            kwargs = {}

        self.play_num += 1
        if self.play_num < 3:
            self.snake.append(self.snake[-1])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        # 清空屏幕
        self.screen.fill(self.black)
        self.draw_border()
        action = self.change_control(action)
        self.control(action)
        self.move()

        reward = kwargs.get('very_far_range', DefaultImediateReward.VERY_FAR_FROM_FOOD)

        terminal = False

        # 检查是否撞到墙壁或自己
        if self.check_collision():
            terminal = True
            reward = kwargs.get('col_wall', DefaultImediateReward.COLLISION_WALL)
            return reward, terminal, self.score

        if self.play_num > kwargs.get('kill_frame', DEFAULT_KILL_FRAME)*(len(self.snake) - 1):
            terminal = True
            reward = kwargs.get('loop', DefaultImediateReward.LOOP)
            return reward, terminal, self.score

        self.get_point()

        if self.point:
            self.score += 1
            reward = kwargs.get('scored', DefaultImediateReward.SCORED)
            self.point = False

        # 绘制蛇和食物
        self.draw_snake_and_food()
        pygame.display.flip()

        return reward, terminal, self.score