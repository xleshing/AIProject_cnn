import json
import os
from enum import Enum
from agent import Agent
from game import Snake
from opencv import ColorObjectDetector
from settings import *
import matplotlib.pyplot as plt


class Windows(Enum):
    W14 = (20, 20, 1, 1)


class Game:
    """
    Run the Game
    Read the json file par_lev.json
    run all the processors with .json
    parameters
    """

    def __init__(self, lv=1):
        """
        (Game, int) -> None
        initialize the game and what 
        world you want the game to run in
        lv: level selected in pygame
            by default it is set to world 1
        """
        self.lv = lv
        self.awake()

    def awake(self):
        """
        (Game) -> None
        read json file get the worlds
        with their parameters
        iterate through all the world until we found
        the world we initialized create the environment
        run processors parallel with each other using multiprocessing
        """
        file = open('par_lev.json', 'r')
        json_pars = json.load(file)
        file.close()

        # get all enum Windows
        for window in Windows:
            # get specific window from json file
            pars = json_pars.get(window.name, [{}])
            # take the window and unpack values
            n, m, k, l = window.value
            # set screen size with nxm tiles
            n, m = (set_size(n), set_size(m))
            pars = pars[0]
            os.environ['SDL_VIDEO_WINDOW_POS'] = '0, 0'
            self.train(n, m, pars)

    # save run stats to a txt file at a specified path
    # txt files are used by build_graphs.py to build graphs
    def save_to_file(self, path, game_num, score, record):
        """
        (Game, str, int, int, int) -> None
        save the file as .txt file
        save game, score, record as following format
        g s r  respectively
        path: path of the txt file
        game_num: total number of generations
        score: current score taken from the game
        record: highest score
        """
        file = open(path, "a+")
        file.write("%s %s %s\n" % (game_num, score, record))
        file.close()

    def train(self, n, m, pars):
        """
        (Game, int, int, dict()) -> None
        train game and run each step as
        sequence of frames
        n: row tiles of the screen
        m: col tiles of the screen
        pars" parameters passed in for each processors
        """
        # initialize
        record = 0
        game = Snake(n, m)
        obj = ColorObjectDetector()
        agent = Agent(obj, pars)
        while True:
            # 取用game中移動前的方向
            direction = game.get_direction()

            # 獲取移動前的狀態
            state_old = obj.get_state(direction=direction)

            # 從狀態判斷行動
            final_move = agent.get_action(state_old)

            # 移動蛇
            reward, done, score = game.play_step(action=final_move, kwargs=pars)

            """if reward != 0:
                print(reward)"""

            # 取用game中移動後的方向
            direction = game.get_direction()

            # 獲取移動後的狀態
            state_new = obj.get_state(direction=direction)

            """plt.imshow(state_new.T, cmap='viridis', interpolation='nearest')
            plt.title('Snake Game State')
            plt.colorbar()
            plt.show()"""

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            # end game if reached num_games from pars or DEFAULT_END_GAME_POINT
            # if set to -1 then run for ever
            if pars.get('num_games', DEFAULT_END_GAME_POINT) != -1:
                if agent.n_games > pars.get('num_games', DEFAULT_END_GAME_POINT):
                    quit()
                    break

            # when game is over
            if done:
                # reset game attributes
                # increase game generation
                # train the long memory
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # new highscore
                if score > record:
                    record = score
                    # save the best model_state
                    agent.model_cnn.save()

                """
                # takes away food depending on given probability, up until 1 food remains
                decrease_probability = pars.get('decrease_food_chance', DECREASE_FOOD_CHANCE)
                if (game.n_food > 1) and (random.random() < decrease_probability):
                    game.n_food -= 1"""

                # prints game information to console
                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                # appends game information to txt file at specified path
                self.save_to_file(f"D:/Users/Lue/Desktop/{pars.get('graph', 'test')}.txt", agent.n_games, score, record)


if __name__ == "__main__":
    # for i in range(2, 3):
    # Game(i)
    Game(14)
