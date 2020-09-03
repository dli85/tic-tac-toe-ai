from game import Game
import random
import numpy as np
import keras

class NeuralNetworkReinforcement:
    def __init__(self):
        self.modelP1 = keras.models.Sequential()

        self.modelP2 = keras.models.Sequential()

    def flatten(self, board):
        board1d = []
        for r in range(len(board)):
            for c in range(len(board)):
                board1d.append(board[r][c])

        return board1d

    def train(self):
        pass

    def one_hot(self, board):
        pass

if __name__ == '__main__':
    pass