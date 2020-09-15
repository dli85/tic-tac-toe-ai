from randomAi import RandomAI
from game import Game
from minimaxAi import minimaxAI
from NNS import NeuralNetworkSupervised
from NNR import NeuralNetworkReinforcement
from tqdm import tqdm

def NNRp1RandP2():
    player1 = NeuralNetworkReinforcement(1)
    player2 = RandomAI(2)

    player1WinCount = 0
    player2WinCount = 0
    drawCount = 0

    for i in tqdm(range(100)):

        g = Game()
        currentPlayer = 1

        while g.checkWinner() == -1:
            if currentPlayer == 1:
                pos = player1.getMove(g.board)
                g.playMove(pos, 1)
                currentPlayer = 2
            else:
                pos = player2.getMove(g.board)
                g.playMove(pos, 2)
                currentPlayer = 1

        if g.checkWinner() == 1:
            player1WinCount += 1
        elif g.checkWinner() == 2:
            player2WinCount += 1
        else:
            drawCount += 1

    print('[P1: minimax, P2: rand] Player 1 won: ' + str(player1WinCount) + ' games, player 2 won ' + str(player2WinCount) + ' games, it was a draw ' + str(drawCount) + ' times')


def NNRp1MiniP2():
    player1 = NeuralNetworkReinforcement(1)
    player2 = minimaxAI(2)

    player1WinCount = 0
    player2WinCount = 0
    drawCount = 0

    for i in tqdm(range(100)):
        g = Game()
        currentPlayer = 1

        while g.checkWinner() == -1:
            if currentPlayer == 1:
                pos = player1.getMove(g.board)
                g.playMove(pos, 1)
                currentPlayer = 2
            else:
                pos = player2.getMove(g.board)
                g.playMove(pos, 2)
                currentPlayer = 1

        if g.checkWinner() == 1:
            player1WinCount += 1
        elif g.checkWinner() == 2:
            player2WinCount += 1
        else:
            drawCount += 1

    print('[P1: Neural Net Reinforcement, P2: Minimax] Player 1 won: ' + str(player1WinCount) + ' games, player 2 won ' + str(player2WinCount) + ' games, it was a draw ' + str(drawCount) + ' times')


def MiniP1RandP2():
    player1 = minimaxAI(1)
    player2 = RandomAI(2)

    player1WinCount = 0
    player2WinCount = 0
    drawCount = 0

    for i in tqdm(range(100)):

        g = Game()
        currentPlayer = 1

        while g.checkWinner() == -1:
            if currentPlayer == 1:
                pos = player1.getMove(g.board)
                g.playMove(pos, 1)
                currentPlayer = 2
            else:
                pos = player2.getMove(g.board)
                g.playMove(pos, 2)
                currentPlayer = 1

        if g.checkWinner() == 1:
            player1WinCount += 1
        elif g.checkWinner() == 2:
            player2WinCount += 1
        else:
            drawCount += 1

    print('[P1: minimax, P2: rand] Player 1 won: ' + str(player1WinCount) + ' games, player 2 won ' + str(player2WinCount) + ' games, it was a draw ' + str(drawCount) + ' times')

def RandP1RandP2():
    player1 = RandomAI(1)
    player2 = RandomAI(2)

    player1WinCount = 0
    player2WinCount = 0
    drawCount = 0

    for i in tqdm(range(100)):

        g = Game()
        currentPlayer = 1

        while g.checkWinner() == -1:
            if currentPlayer == 1:
                pos = player1.getMove(g.board)
                g.playMove(pos, 1)
                currentPlayer = 2
            else:
                pos = player2.getMove(g.board)
                g.playMove(pos, 2)
                currentPlayer = 1

        if g.checkWinner() == 1:
            player1WinCount += 1
        elif g.checkWinner() == 2:
            player2WinCount += 1
        else:
            drawCount += 1

        #print('Finished ' + str(i+1) + ' games')

    print('[P1: rand, P2: rand] Player 1 won: ' + str(player1WinCount) + ' games, player 2 won ' + str(player2WinCount) + ' games, it was a draw ' + str(drawCount) + ' times')

def RandP1NNSP2():
    player1 = RandomAI(1)
    player2 = NeuralNetworkSupervised(2)
    player2.loadModel('models/NNRand.h5')

    player1WinCount = 0
    player2WinCount = 0
    drawCount = 0

    for i in tqdm(range(100)):

        g = Game()
        currentPlayer = 1

        while g.checkWinner() == -1:
            if currentPlayer == 1:
                pos = player1.getMove(g.board)
                g.playMove(pos, 1)
                currentPlayer = 2
            else:
                pos = player2.getMove(g.board)
                g.playMove(pos, 2)
                currentPlayer = 1

        if g.checkWinner() == 1:
            player1WinCount += 1
        elif g.checkWinner() == 2:
            player2WinCount += 1
        else:
            drawCount += 1

    print('[P1: Rand, P2: Neural Net] Player 1 won: ' + str(player1WinCount) + ' games, player 2 won ' + str(player2WinCount) + ' games, it was a draw ' + str(drawCount) + ' times')

if __name__ == '__main__':
    NNRp1RandP2()
    NNRp1MiniP2()
    RandP1NNSP2()
    MiniP1RandP2()
    RandP1RandP2()
