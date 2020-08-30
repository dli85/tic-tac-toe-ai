class Game:
    def __init__(self):
        self.board = [[0,0,0],
                      [0,0,0],
                      [0,0,0]]

        self.current_player = 1

    def legalMove(self, pos):
        playR = int((pos - 1) / 3)
        playC = (pos - 1) - (playR * 3)
        if self.board[playR][playC] == 0:
            return True
        return False

    def printBoard(self):
        symb = [' ', 'X', 'O']  # Symbols for the board. Only needed to display the board

        print()
        print(symb[self.board[0][0]] + ' | ' + symb[self.board[0][1]] + ' | ' + symb[
            self.board[0][2]] + "                                               " +
              "Board Positions are as follows:  1  2  3")
        print('---------')
        print(symb[self.board[1][0]] + ' | ' + symb[self.board[1][1]] + ' | ' + symb[
            self.board[1][2]] + "                                                                                "
              + "4  5  6")
        print('---------')
        print(symb[self.board[2][0]] + ' | ' + symb[self.board[2][1]] + ' | ' + symb[
            self.board[2][2]] + "                                                                                "
              + "7  8  9")
        print()

    #Returns 1 or 2 if somebody won, 0 for a tie, and -1 if the game is still on-going
    def checkWinner(self):
        if (self.board[0][0] == self.board[0][1] == self.board[0][2] != 0):
            return self.board[0][0]
        elif (self.board[1][0] == self.board[1][1] == self.board[1][2] != 0):
            return self.board[1][0]
        elif (self.board[2][0] == self.board[2][1] == self.board[2][2] != 0):
            return self.board[2][0]
        elif (self.board[0][0] == self.board[1][0] == self.board[2][0] != 0):
            return self.board[0][0]
        elif (self.board[0][1] == self.board[1][1] == self.board[2][1] != 0):
            return self.board[0][1]
        elif (self.board[0][2] == self.board[1][2] == self.board[2][2] != 0):
            return self.board[0][2]
        elif (self.board[0][0] == self.board[1][1] == self.board[2][2] != 0):
            return self.board[1][1]
        elif (self.board[0][2] == self.board[1][1] == self.board[2][0] != 0):
            return self.board[1][1]

        # Given that nobody has won, this code checks if all spaces are full.
        # If yes returns 0 for tie.
        sumBoard = 0
        for i in self.board:
            for p in i:
                sumBoard += p
        if (sumBoard == 13): return 0

        # No player has won or tied, return -1 for "game is still going
        return -1

    #pos is number 1 to 9. Player is whose turn it currently is
    def playMove(self, pos, player):
        playR = int((pos - 1) / 3)
        playC = (pos - 1) - (playR * 3)

        if (self.board[playR][playC] == 0):
            self.board[playR][playC] = player
        else:
            return None

