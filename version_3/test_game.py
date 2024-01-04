import cv2
import pygame
import sys
import random
from opencv import ColorObjectDetector
import os


class SnakeGame:
    def __init__(self, width=400, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("test_snake")

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)

        self.snake_size = 20
        self.snake = [(width // 2, height // 2)]
        self.direction = [0, 1, 0, 0]  # 起始方向為向左

        self.food = self.generate_food()

        self.clock = pygame.time.Clock()
        self.can_move = False

    def generate_food(self):
        return (random.randint(0, (self.width - self.snake_size) // self.snake_size) * self.snake_size,
                random.randint(0, (self.height - self.snake_size) // self.snake_size) * self.snake_size)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                self.can_move = True
                if event.key == pygame.K_UP and self.direction != [0, 0, 0, 1]:  # 防止反向移动
                    self.direction = [0, 0, 1, 0]
                elif event.key == pygame.K_DOWN and self.direction != [0, 0, 1, 0]:
                    self.direction = [0, 0, 0, 1]
                elif event.key == pygame.K_LEFT and self.direction != [0, 1, 0, 0]:
                    self.direction = [1, 0, 0, 0]
                elif event.key == pygame.K_RIGHT and self.direction != [1, 0, 0, 0]:
                    self.direction = [0, 1, 0, 0]

    def get_current_direction(self):
        return self.direction

    def update_snake(self):
        if self.can_move:
            new_head = [0, 1, 0, 0]
            if self.direction == [1, 0, 0, 0]:
                new_head = (self.snake[0][0] - self.snake_size, self.snake[0][1])
            elif self.direction == [0, 1, 0, 0]:
                new_head = (self.snake[0][0] + self.snake_size, self.snake[0][1])
            elif self.direction == [0, 0, 1, 0]:
                new_head = (self.snake[0][0], self.snake[0][1] - self.snake_size)
            elif self.direction == [0, 0, 0, 1]:
                new_head = (self.snake[0][0], self.snake[0][1] + self.snake_size)

            self.snake.insert(0, new_head)
            self.can_move = False

            if self.snake[0] == self.food:
                self.food = self.generate_food()
            else:
                self.snake.pop()

            if (self.snake[0][0] < 0 or self.snake[0][0] >= self.width or
                    self.snake[0][1] < 0 or self.snake[0][1] >= self.height or
                    self.snake[0] in self.snake[1:]):
                pygame.quit()
                sys.exit()

    def draw_screen(self):
        self.screen.fill(self.black)
        for i, segment in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.screen, self.green, (segment[0], segment[1], self.snake_size, self.snake_size))
            else:
                pygame.draw.rect(self.screen, self.white, (segment[0], segment[1], self.snake_size, self.snake_size))
                pygame.draw.rect(self.screen, self.black, (segment[0], segment[1], self.snake_size, self.snake_size), 2)
        pygame.draw.rect(self.screen, self.red, (self.food[0], self.food[1], self.snake_size, self.snake_size))

        pygame.display.flip()

    def run(self):
        self.handle_events()
        self.update_snake()
        self.draw_screen()


if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
    game = SnakeGame()
    obj = ColorObjectDetector()
    direction = game.get_current_direction()
    while True:
        state_old = obj.get_state(title="old", direction=direction)
        game.run()
        state_new = obj.get_state(title="new", direction=direction)
        cv2.waitKey(1000)
