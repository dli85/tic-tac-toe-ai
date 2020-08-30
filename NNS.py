from randomAi import RandomAI
from minimaxAi import minimaxAI
from game import Game
import numpy as np
from copy import deepcopy
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

class DataCollector:
    def __init__(self):
        self.history = []
        self.game = Game()

    def reset(self):
        self.game = Game()

    def doCollection(self, numGames=10000):
        self.history = []
        for i in range(numGames):
            self.reset()
            self.playGame()
            print('Completed ' + str(i+1) + ' out of ' + str(numGames))

        with open('training.pkl', 'wb') as file:
            pickle.dump(self.history, file)

    def doCollectionRand(self, numGames=10000):
        self.history = []
        for i in range(numGames):
            self.reset()
            self.playGameRand()
            print('Completed ' + str(i+1) + ' out of ' + str(numGames))

        with open('trainingRand.pkl', 'wb') as file:
            pickle.dump(self.history, file)

    def playGameRand(self):
        player1 = RandomAI(1)
        player2 = RandomAI(2)
        currentTurn = 1
        winner = -1
        currentGame = []

        while self.game.checkWinner() == -1:
            if currentTurn == 1:
                pos = player1.selectMove(self.game.board)
                self.game.playMove(pos, 1)
                currentTurn = 2
            else:
                pos = player2.selectMove(self.game.board)
                self.game.playMove(pos, 2)
                currentTurn = 1
            currentGame.append(deepcopy(self.game.board))

        winner = self.game.checkWinner()
        for gameState in currentGame:
            self.history.append((winner, deepcopy(gameState)))

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

class NeuralNetworkSupervised:
    def __init__(self, player, inputDims=9, outputDims=3, epochs=100, batchSize=32):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(inputDims, )))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(outputDims, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.inputDims = inputDims
        self.outputDims = outputDims
        self.epochs = epochs
        self.batchSize = batchSize

        self.player = player

        self.data, self.dataRand = self.loadData()

    def loadData(self, fileName1='training.pkl', fileName2='trainingRand.pkl'):
        with open('training.pkl', 'rb')as file:
            minimaxData = pickle.load(file)

        with open('trainingRand.pkl', 'rb')as file:
            randData = pickle.load(file)

        return minimaxData, randData

    def splitData(self, inputs, outputs):
        boundary = int(0.9 * len(inputs))
        feature_train = inputs[:boundary]
        label_train = outputs[:boundary]
        feature_test = inputs[boundary:]
        label_test = outputs[boundary:]

        return feature_train, label_train, feature_test, label_test

    def doTraining(self):
        features = []
        labels = []
        for gameState in self.data:
            features.append(gameState[1])
            labels.append(gameState[0])

        inputs = np.array(features).reshape((-1, self.inputDims))
        outputs = to_categorical(labels, num_classes=3)

        feature_train, label_train, feature_test, label_test = self.splitData(inputs, outputs)

        self.model.fit(feature_train, label_train, validation_data=(feature_test, label_test), epochs=self.epochs, batch_size=self.batchSize)
        self.saveModel('NNMini.h5')

    def doTrainingRand(self):
        features = []
        labels = []
        for gameState in self.dataRand:
            features.append(gameState[1])
            labels.append(gameState[0])

        inputs = np.array(features).reshape((-1, self.inputDims))
        outputs = to_categorical(labels, num_classes=3)

        feature_train, label_train, feature_test, label_test = self.splitData(inputs, outputs)

        self.model.fit(feature_train, label_train, validation_data=(feature_test, label_test), epochs=self.epochs, batch_size=self.batchSize)
        self.saveModel('NNRand.h5')

    def saveModel(self, name):
        self.model.save(name)

    def loadModel(self, name):
        self.model = load_model(name)

    def getQVal(self, board, desired):
        arr = np.array(board).reshape(-1, self.inputDims)
        return self.model.predict(arr)[0][desired]

    def getMove(self, board):

        possibleNewBoardStates = []

        for r in range(len(board)):
            for c in range(len(board)):
                if board[r][c] == 0:
                    newBoard = deepcopy(board)
                    newBoard[r][c] = self.player
                    possibleNewBoardStates.append(newBoard)

        highestQ = -999

        for boardState in possibleNewBoardStates:
            if self.player == 1:
                qVal = self.getQVal(boardState, 1)
            else:
                qVal = self.getQVal(boardState, 2)

            if qVal > highestQ:
                highestQ = qVal
                bestBoardState = deepcopy(boardState)

        pos = 0



        for r in range(len(board)):
            for c in range(len(board)):
                if board[r][c] != bestBoardState[r][c]:
                    pos = r*3 + c + 1

        return pos

if __name__ == '__main__':

    #d = DataCollector()
    #d.doCollection()
    #d.doCollectionRand()

    #agent = NeuralNetworkSupervised(1, 9, 3, 50, 32)
    #agent.doTraining()

    #agentRand = NeuralNetworkSupervised(1, 9, 3, 50, 32)
    #agentRand.doTrainingRand()

    pass
