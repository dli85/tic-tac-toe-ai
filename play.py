from randomAi import RandomAI
from game import Game
from minimaxAi import minimaxAI
from NNS import NeuralNetworkSupervised

def playerTurn():
    pos = int(input('Where would you like to play? '))
    return pos

aiNum = int(input('What ai would you like to play against? (1 = random, 2 = minimax 3 = Neural Network Supervised Learning): '))

if aiNum == 1:
    currentTurn = 1
    g = Game()
    agent = RandomAI(2)

    while g.checkWinner() == -1:
        g.printBoard()
        if currentTurn == 1:
            pos = playerTurn()
            while not g.legalMove(pos):
                print('Not a legal move, please choose again')
                pos = playerTurn()
            g.playMove(pos, 1)
            currentTurn = 2
        else:
            pos = agent.getMove(g.board)
            g.playMove(pos, 2)
            currentTurn = 1
            print('Ai has played at position', pos)

    g.printBoard()

    if g.checkWinner() == 1:
        print("You won!")
    elif g.checkWinner() == 2:
        print("computer won")
    else:
        print("its a tie")
elif aiNum == 2:
    currentTurn = 1
    g = Game()
    agent = minimaxAI(2)
    while g.checkWinner() == -1:
        g.printBoard()
        if currentTurn == 1:
            pos = playerTurn()
            while not g.legalMove(pos):
                print('Not a legal move, please choose again')
                pos = playerTurn()
            g.playMove(pos, 1)
            currentTurn = 2
        else:
            pos = agent.getMove(g.board)
            g.playMove(pos, 2)
            currentTurn = 1
            print('Ai has played at position', pos)

    g.printBoard()

    if g.checkWinner() == 1:
        print("You won!")
    elif g.checkWinner() == 2:
        print("computer won")
    else:
        print("its a tie")
elif aiNum == 3:
    currentTurn = 1
    g = Game()
    agent = NeuralNetworkSupervised(2)
    agent.loadModel('NNRand.h5')

    while g.checkWinner() == -1:
        g.printBoard()
        if currentTurn == 1:
            pos = playerTurn()
            while not g.legalMove(pos):
                print('Not a legal move, please choose again')
                pos = playerTurn()
            g.playMove(pos, 1)
            currentTurn = 2
        else:
            pos = agent.getMove(g.board)
            g.playMove(pos, 2)
            currentTurn = 1
            print('Ai has played at position', pos)

    g.printBoard()

    if g.checkWinner() == 1:
        print("You won!")
    elif g.checkWinner() == 2:
        print("computer won")
    else:
        print("its a tie")
