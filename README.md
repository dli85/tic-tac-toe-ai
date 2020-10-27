# tic-tac-toe-ai

Several algorithms for the game tic tac toe:

- Random 
- Minimax 
- Neural Network Supervised Learning 
- Neural Network Reinforcement Learning

## Random

The random algorithm randomly selects moves from the available ones. 

## Minimax

Minimax is the "standard" tic-tac-toe algorithm. It is able to play a perfect game every time (it is impossible to beat it). The algorithm works by conducting a tree-search on the available moves. The algorithm looks into every game-ending scenario for each move and every possible way the game can end. The algorithm assumes that the opponent will play perfectly so if a move leads to a game-ending scenario with a possible loss, the algorithm gives that move a negative score. Moves that lead to wins or draws are given positive scores and 0 respectively. The algorithm will then choose the move with the highest score (it does not matter if multiple moves are tied for the highest score - they all lead to the same game-ending scenario).

Although minimax can play the game perfectly, it is very inefficient as minimax must go through all possibilities that the game can end. For a game like chess or go, a pure minimax algorithm wouldn't be feasible as chess and go have way too many possible outcomes and going through all of them would take way too much time (There are more configurations of the go board than there are atoms in the observable universe). Even with a game as small as tic-tac-toe, the minimax algorithm takes a non-negligible amount of time to calculate a move.

## Neural Network (Supervised learning)

Another way to construct a tic-tac-toe algorithm is to use a neural-network with supervised learning. The algorithm works by first constructing a dataset of many tic-tac-toe game histories. The features will consist of board states and the labels will be which player won. The data is collected by having two random bots play against each other and recording the outcomes. The model is then trained using the data to predict which player will win from a given board state.

In a game scenario, the ai will first make an list of all the possible states from the current board state. The model then predicts, for each new board state, which player will win. Using these predictions, the model will then chose the move that gives itself the highest probability of winning.

The neural network (supervised learning) is not very good at defeating humans. This is because of the quality of the dataset and the method of learning. The dataset only consists of  randomAi vs randomAi games. Thus, predicting a winner for a given board state would be based on how a randomai bot might move, not how a minimax or a human would play.

## Neural Network (Reinforcement learning)

The reinforcement learning algorithm works by training an agent/model through a reward system. At first, the agent does not know any game strategies, it only knows the rules. Through a process of exploration, the agent is able to determine the most rewarding path/strategy. The process for learning is the following:

1. A new game is started.
2. The two agents/models (one for player 1, the other for player 2) play against each other until the game is finished. Whenever an agent has to make a move, it uses the epsilon-greedy method to decided whether to make a random move or make the best possible move (exploration vs exploitation).
3. Once the game is finished, the game history is recorded. Once enough games are finished, the learning starts.
4. During learning, the game history is analyzed. Moves that result in a win are rewarded positively, moves resulting in a draw or a loss are rewarded similarly. Over time, the agents become better. 

# Usage

Use the play.py file to play tic-tac-toe. You will be prompted to select which AI you would like to play against.

The randomAi.py and the minimaxAi.py contain the random bot and the minimax bot respectively. The NNS.py and NNR.py are used to train the neural network supervised learning and the neural network reinforcement learning respectively.

The datacollection.py program plays different types of AI's against each other. The results are then recorded.

The game.py file contains the tic-tac-toe game class which is used to play tic-tac-toe.
