# tic-tac-toe-ai

Several algorithms for the game tic tac toe.

## Random

The random algorithm randomly selects moves from the available ones. Easily beatable

## Minimax

Minimax is the "standard" tic-tac-toe algorithm. It is able to play a perfect game every time (it is impossible to beat it). The algorithm works by essentially doing a tree-search on the available moves. The algorithm looks into every game-ending scenario for each move. The algorithm assumes that the opponent will play perfectly so if a move leads to a game-ending scenario with a possible loss, the algorithm gives that move a negative score. Moves that lead to only wins or draws are given positive scores and 0 respectively. The algorithm will then choose the move with the highest score (it does not matter if multiple moves are tied for the highest score - they all lead to the same game-ending scenario).

Although minimax can play the game perfectly, it is very inefficient as minimax must go through all possibilities that the game can end. For a game like chess or go, a pure minimax algorithm wouldn't be feasible as chess and go have way too many possible outcomes and going through all of them would take way too much time (There are more configurations of the go board than there are atoms in the observable universe). Even with a game as small as tic-tac-toe, the minimax algorithm takes a non-negligible amount of time to calculate a move.

## Neural Network (Supervised learning) - unfinished

Another way to construct a tic-tac-toe algorithm is to use a neural-network with supervised learning. The idea is to feed the algorithm a dataset and teach it to predict whether certain moves would result in a favorable outcome or an unfavorable outcome. We first create a dataset of tic-tac-toe moves.

## Neural Network (Reinforcement learning) - unfinished
