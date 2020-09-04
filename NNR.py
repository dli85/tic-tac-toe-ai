from game import Game
import random
import numpy as np
import keras
from copy import deepcopy

class NeuralNetworkReinforcement:
    def __init__(self):
        self.modelP1 = keras.models.Sequential()

        self.modelP1.add(keras.layers.Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.add(keras.layers.Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.add(keras.layers.Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.add(keras.layers.Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.add(keras.layers.Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        self.modelP2 = keras.models.Sequential()

        self.modelP2.add(keras.layers.Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.add(keras.layers.Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.add(keras.layers.Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.add(keras.layers.Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.add(keras.layers.Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


        self.gamma = 0.7
        self.epsilon = 0.7

        self.allGameHistory = []

    def flatten(self, board):
        board1d = []
        for r in range(len(board)):
            for c in range(len(board)):
                board1d.append(board[r][c])

        return board1d

    def doLearning(self):

        player1Boards = []
        player1QVals = []
        player2Boards = []
        player2QVals = []

        for game in self.allGameHistory:
            maxGameIndex = len(game) - 1
            temp = Game()
            temp.board = deepcopy(game[maxGameIndex])
            winner = temp.checkWinner()

            if winner == 1:
                reward = 1
            elif winner == 2:
                reward = -1
            else:
                reward = 0

            for i in range(maxGameIndex):
                pass
                #TODO

        #TODO

    def convertToHot(self, board):
        hot = []
        encodings = [[1,0,0], [0,1,0], [0,0,1]] #Open square, player 1, player 2

        for r in range(len(board)):
            for c in range(len(board)):
                hot.extend(encodings[board[r][c]])

        return hot

    def getAvailMoves(self, board):
        moves = []
        for r in range(len(board)):
            for c in range(len(board)):
                if board[r][c] == 0:
                    moves.append(int(r*3 + c + 1))

        return moves

    def startTraining(self, numGames=3000):


        for i in range(numGames):
            currentPlayer = 1
            g = Game()
            currentGameHistory = []

            while g.checkWinner() == -1:
                if currentPlayer == 1:
                    if random.random() <= self.epsilon:
                        availMoves = self.getAvailMoves(g.board)
                        selectedMove = random.choice(availMoves)
                        g.playMove(selectedMove, 1)
                        currentGameHistory.append(deepcopy(g.board))
                    else:
                        qVals = self.modelP1.predict(np.asarray([self.convertToHot(g.board)]), batch_size=1)[0]
                        bestQ = -999
                        selectedMove = 0

                        for r in range(len(g.board)):
                            for c in range(len(g.board)):
                                pos = r*3 + c + 1
                                if g.board[r][c] == 0 and qVals[pos-1] > bestQ:
                                    bestQ = qVals[pos-1]
                                    selectedMove = pos

                        g.playMove(selectedMove, 1)
                        currentGameHistory.append(deepcopy(g.board))

                    currentPlayer = 2
                else:
                    if random.random() <= self.epsilon:
                        availMoves = self.getAvailMoves(g.board)
                        selectedMove = random.choice(availMoves)
                        g.playMove(selectedMove, 2)
                        currentGameHistory.append(deepcopy(g.board))
                    else:
                        qVals = self.modelP2.predict(np.asarray([self.convertToHot(g.board)]), batch_size=1)[0]
                        bestQ = -999
                        selectedMove = 0

                        for r in range(len(g.board)):
                            for c in range(len(g.board)):
                                pos = r*3 + c + 1
                                if g.board[r][c] == 0 and qVals[pos-1] > bestQ:
                                    bestQ = qVals[pos-1]
                                    selectedMove = pos

                        g.playMove(selectedMove, 2)
                        currentGameHistory.append(deepcopy(g.board))

                    currentPlayer = 1

            self.allGameHistory.append(currentGameHistory)

        self.doLearning()



if __name__ == '__main__':
    agent = NeuralNetworkReinforcement()
    agent.startTraining()
