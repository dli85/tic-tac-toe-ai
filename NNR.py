from game import Game
import random
import numpy as np
import keras
import math
from copy import deepcopy
from tqdm import tqdm

class NeuralNetworkReinforcement:
    def __init__(self, player):
        self.modelP1 = keras.models.Sequential()

        self.modelP1.add(keras.layers.Dense(128, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.add(keras.layers.Dense(256, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.add(keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.add(keras.layers.Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        self.modelP2 = keras.models.Sequential()

        self.modelP2.add(keras.layers.Dense(128, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.add(keras.layers.Dense(256, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.add(keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.add(keras.layers.Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.modelP2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        self.player = player

        self.train1 = True

        try:
            self.modelP1 = keras.models.load_model('models/NNRp1.h5')
        except:
            print('Model1 failed to load')

        try:
            self.modelP2 = keras.models.load_model('models/NNRp2.h5')
        except:
            print('Model2 failed to load')

        self.gamma = 0.7
        self.epsilon = 0.7

        self.allGameHistory = []

    def flatten(self, board):
        board1d = []
        for r in range(len(board)):
            for c in range(len(board)):
                board1d.append(board[r][c])

        return board1d

    def convertToHot(self, board):
        hot = []
        encodings = [[1,0,0], [0,1,0], [0,0,1]] #Open square, player 1, player 2

        for r in range(len(board)):
            for c in range(len(board)):
                hot.extend(encodings[board[r][c]])

        return hot

    def doFitting(self, player1Boards, player1QVals, player2Boards, player2QVals):
        if self.train1:
            new = []
            for i in player1Boards:
                new.append(self.convertToHot(i))
            self.modelP1.fit(np.asarray(new), np.asarray(player1QVals), epochs=6, batch_size=len(player1QVals), verbose=2)
            self.modelP1.save('models/NNRp1.h5')
        else:
            new = []
            for i in player2Boards:
                new.append(self.convertToHot(i))
            self.modelP2.fit(np.asarray(new), np.asarray(player2QVals), epochs=6, batch_size=len(player2QVals), verbose=2)
            self.modelP2.save('models/NNRp2.h5')

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
                if i % 2 == 0:
                    for r in range(3):
                        for c in range(3):
                            if game[i][r][c] != game[i+1][r][c]:
                                rewards = np.zeros(9)
                                rewards[r*3+c] = reward * (0.7**(math.floor((len(game)-i)/2)-1))
                                player1Boards.append(deepcopy(game[i]))
                                tempArr = rewards.copy()
                                player1QVals.append(tempArr)
                else:
                    for r in range(3):
                        for c in range(3):
                            if game[i][r][c] != game[i+1][r][c]:
                                rewards = np.zeros(9)
                                rewards[r*3+c] = reward * (0.7**(math.floor((len(game)-i)/2)-1))
                                player2Boards.append(deepcopy(game[i]))
                                tempArr = rewards.copy()
                                player2QVals.append(tempArr)

        self.doFitting(player1Boards, player1QVals, player2Boards, player2QVals)

        self.train1 = not self.train1


    def getAvailMoves(self, board):
        moves = []
        for r in range(len(board)):
            for c in range(len(board)):
                if board[r][c] == 0:
                    moves.append(int(r*3 + c + 1))

        return moves

    def startTraining(self, numGames=1000):
        if self.train1:
            print('training model 1')
        else:
            print('training model 2')
        for i in tqdm(range(numGames)):
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
                        bestQ = 999
                        selectedMove = 0

                        for r in range(len(g.board)):
                            for c in range(len(g.board)):
                                pos = r*3 + c + 1
                                if g.board[r][c] == 0 and qVals[pos-1] < bestQ:
                                    bestQ = qVals[pos-1]
                                    selectedMove = pos

                        g.playMove(selectedMove, 2)
                        currentGameHistory.append(deepcopy(g.board))

                    currentPlayer = 1

            self.allGameHistory.append(currentGameHistory)
        self.doLearning()
        self.allGameHistory = []

    def getMove(self, board):
        if self.player == 1:
            best = -999
            Qs = self.modelP2.predict(np.asarray([self.convertToHot(board)]), batch_size=1)[0]
            selectedMove = 0

            for r in range(3):
                for c in range(3):
                    pos = r*3 + c + 1
                    if(board[r][c] == 0 and Qs[pos-1] > best):
                        selectedMove = pos
                        best = Qs[pos-1]

            return selectedMove

        elif self.player == 2:
            best = -999
            Qs = self.modelP1.predict(np.asarray([self.convertToHot(board)]), batch_size=1)[0]
            selectedMove = 0

            for r in range(3):
                for c in range(3):
                    pos = r*3 + c + 1
                    if(board[r][c] == 0 and Qs[pos-1] > best):
                        selectedMove = pos
                        best = Qs[pos-1]

            return selectedMove

if __name__ == '__main__':
    agent = NeuralNetworkReinforcement(1)
    for i in range(80):
        agent.startTraining()
        print('Training iteration number ' + str(i+1) + ' has finished.')
        if agent.epsilon > .5:
            agent.epsilon -= 0.01
