import random

class RandomAI:
    def __init__(self, playerNum):
        self.playerNum = playerNum

    def getMove(self, board):
        states = []
        for r in board:
            for c in r:
                states.append(c)

        availableMoves = []
        for i in range(len(states)):
            if states[i] == 0:
                availableMoves.append(i + 1)

        return availableMoves[random.randint(0, len(availableMoves) - 1)]


