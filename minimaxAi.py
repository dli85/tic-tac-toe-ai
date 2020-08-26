from copy import deepcopy

class minimaxAI:
    def __init__(self, player):
        self.player = player


    def getMove(self, board):
        best = 999
        aiMove = -1
        for r in range(len(board)):
            for c in range(len(board)):
                if board[r][c] == 0:
                    tempBoard = deepcopy(board)
                    tempBoard[r][c] = 2
                    val = self.minimax(tempBoard, 1)
                    if val < best:
                        best = val
                        aiMove = r*3 + c + 1
        return aiMove

    def minimax(self, temp_board, playerMaximizing):
        if self.checkWinner(temp_board) != -1:
            if self.checkWinner(temp_board) == 1:
                return 10
            elif self.checkWinner(temp_board) == 2:
                return -10
            else:
                return 0

        if playerMaximizing == 1:
            best = -999
            for r in range(len(temp_board)):
                for c in range(len(temp_board)):
                    if(temp_board[r][c] == 0):
                        tempBoard = deepcopy(temp_board)
                        tempBoard[r][c] = 1
                        val = self.minimax(tempBoard, 2)
                        if val > best:
                            best = val
            return best
        else:
            best = 999
            for r in range(len(temp_board)):
                for c in range(len(temp_board)):
                    if(temp_board[r][c] == 0):
                        tempBoard = deepcopy(temp_board)
                        tempBoard[r][c] = 2
                        val = self.minimax(tempBoard, 1)
                        if val < best:
                            best = val
            return best


    def checkWinner(self, board):
        if (board[0][0] == board[0][1] == board[0][2] != 0):
            return board[0][0]
        elif (board[1][0] == board[1][1] == board[1][2] != 0):
            return board[1][0]
        elif (board[2][0] == board[2][1] == board[2][2] != 0):
            return board[2][0]
        elif (board[0][0] == board[1][0] == board[2][0] != 0):
            return board[0][0]
        elif (board[0][1] == board[1][1] == board[2][1] != 0):
            return board[0][1]
        elif (board[0][2] == board[1][2] == board[2][2] != 0):
            return board[0][2]
        elif (board[0][0] == board[1][1] == board[2][2] != 0):
            return board[1][1]
        elif (board[0][2] == board[1][1] == board[2][0] != 0):
            return board[1][1]

        # Given that nobody has won, this code checks if all spaces are full.
        # If yes returns 0 for tie.
        sumBoard = 0
        for i in board:
            for p in i:
                sumBoard += p
        if (sumBoard == 13): return 0

        # No player has won or tied, return -1 for "game is still going
        return -1


