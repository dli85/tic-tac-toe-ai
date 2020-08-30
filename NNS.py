from randomAi import RandomAI
from minimaxAi import minimaxAI
from game import Game
import numpy as np
from copy import deepcopy
import pickle
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

class DataCollector:
    def __init__(self):
        self.history = []
        self.game = Game()

    def reset(self):
        self.game = Game()

    def doTraining(self, numGames=10000):
        player1WinCount = 0
        player2WinCount = 0
        drawCount = 0

        for i in range(numGames):
            self.reset()
            winner = self.playGame()
            print('Completed ' + str(i+1) + ' out of ' + str(numGames))
            if winner == 1: player1WinCount += 1
            if winner == 2: player2WinCount += 1
            if winner == 0: drawCount += 1

        with open('training.pkl', 'wb') as file:
            pickle.dump(self.history, file)

    def openData(self):
        with open('training.pkl', 'rb')as file:
            newList = pickle.load(file)
        #print(newList)


    def playGame(self):
        player1 = RandomAI(1)
        player2 = minimaxAI(2)
        currentTurn = 1
        winner = -1 #1 if player 1 won, 2 if player 2 won. 0 for draw
        currentGame = []

        while self.game.checkWinner() == -1:
            if currentTurn == 1:
                pos = player1.selectMove(self.game.board)
                self.game.playMove(pos, 1)
                currentTurn = 2
            else:
                pos = player2.getMove(self.game.board)
                self.game.playMove(pos, 2)
                currentTurn = 1
            currentGame.append(deepcopy(self.game.board))

        winner = self.game.checkWinner()
        for gameState in currentGame:
            self.history.append((winner, deepcopy(gameState)))

        return winner


class NeuralNetworkSupervised:
    def __init__(self, inputDims, outputDims, epochs, batchSize = 32):
        pass

if __name__ == '__main__':
    d = DataCollector()
    #d.doTraining()
    #d.openData()
