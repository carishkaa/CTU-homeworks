#Task 3 - Connect 4
Please read the following instructions very carefully as it can help you later when solving your assignment. If you have any problems with the assignment, please feel free to contact Jonáš Kulhánek (jonas.kulhanek@live.com).

In this task, we focus on two-player zero-sum games. We will build a framework applicable to any two-player zero-sum game. We will mainly work with the game [Connect 4](https://www.mathsisfun.com/games/connect4.html). The homework consists of two tasks. One is to implement an agent, which acts according to the minimax strategy. For this agent, a heuristic function is available, and the agent has to utilize it. We will call this agent the *DeterministicAgent* in the assignment. The second task is to implement the Monte Carlo Tree Search agent utilizing the UCT sampling strategy to solve the same problem but without the heuristic function. The second agent will be called *MonteCarloAgent*.

## Code
Before you start, you should have a Python3 interpreter ready (the minimal Python version is 3.6, but version 3.8 is recommended). A virtual environment (venv) is recommended. You have to ensure all packages from the file `requirements.txt` are installed in your interpreter. If you use pip, you can install them by running `pip install -r requirements.txt`. Also, if you want to use the visualization in the first part of the assignment (strongly recommended), please install also `PyQt5` package. Package `PyQt5` has to be installed outside a virtual environment. If you use venv, you need to create your virtual environment with allowed global packages, e.g., `python -m venv ~/envs/zui --system-site-packages`.

You are given a package containing several files. You need to implement and submit a single file named `agent.py`. In this file, there are two classes: `DeterministicAgent`, `MonteCarloAgent` with missing parts. All missing parts are marked with a `TODO` mark, and you have to implement them. 

There is another file called `test_agent.py` which contains tests that might be useful when implementing your assignment. You can extend them with your own tests. In order to run them, you need to run the following command `pytest --runslow`. We recommend you get familiar with pytest since you can filter which tests you want to be run and which not. If you run the command as follows: `pytest`, you will run only faster tests, which might be useful at the beginning. Generally, you should pass all tests before submitting your solution.
Examples:
You can filter tests for the first part of the assignment by running:
```
pytest --runslow -k Deterministic
```
or only faster tests by running:
```
pytest -k Deterministic
```
Similarly, to filter tests for the second part of the assignment, you can use:
```
pytest --runslow -k MonteCarlo
```
or only faster tests by running:
```
pytest -k MonteCarlo
```

In the file `playground.py`, there is a sample code that you might find useful in the first part of the assignment. You can use it to debug your code. Ultimately, it is strongly recommended to write your own tests and add them to `test_agent.py`.

Another very useful tool is `visualize_tree.py`. It visualizes your Alpha-beta algorithm as it walks over the tree, printing the values your function returns. It also shows which branches are not taken due to cut-offs. Even if you finished your assignment, we recommend you try to run it and look at its output.

## Game Interface
The interface of all games used in this assignment either for your testing or for evaluation is contained in file `game.py` and has the following interface:
```
class Game:
    @property
    def to_play(self):
        return the number of the player whose move it is - either 0 or 1

    def apply(self, action):
        apply :action (mutating the object) and returns nothing

    def terminal(self):
        return true if the game terminated

    def terminal_value(self, player):
        let t be the final reward (usually 1)
        return None if the game had not terminated
        return t if :player won the game -t if it lost and 0 otherwise

    def clone(self):
        creates and returns a deep-copy of this instance

    def legal_actions(self):
        returns the list of available actions

    def __eq__(self, other):
        returns True if the two games are in the same state
        applying a move may change this equality!!

    def __hash__(self):
        returns the hash of the current game state
        applying a move may change this number!!
```
The important thing to keep in mind is that by applying an action, the game changes its state and, therefore, may change its hash.

There are 3 games implemented at the moment. One of them is Connect4. There are two testing games which you might find useful when designing your tests. `TreeGame` allows you to design your own search tree and then run the algorithm on that tree. You can generate random heuristic function values and visualize your game tree using `visualize_tree.py`, or alternatively, you can navigate to this page: [http://inst.eecs.berkeley.edu/~cs61b/fa14/ta-materials/apps/ab\_tree\_practice/](http://inst.eecs.berkeley.edu/~cs61b/fa14/ta-materials/apps/ab_tree_practice/), generate your tree there and then use their values. The last game you might find useful is `MultiArmedBanditGame`, which is essentially a multi-armed bandit problem placed at a fixed depth of the tree. You can verify that your algorithm returns a value close to the optimal value for maximizing and minimizing players.


## DeterministicAgent
In this part of the assignment, you have to implement the missing code pieces in class `DeterministicAgent` in the file `agent.py`. The agent is given a heuristic function and the desired depth of the search tree. In the leaf nodes, the agent has to use its heuristic function to obtain the game score. Some clever pruning techniques should be utilized in order to pass the tests. Note that the heuristic function accepts two arguments, first is the game instance, and the second is the player whose score you want to compute.

There are three players implemented in the evaluation system, and you need to beat at least the worst one to get any points. Upon submitting your solution, you get a score for each task as follows:
- 0 <= score < 0.5 if your solution was better than or the same as our worst solution
- 0.5 <= score < 1 if your solution was better than or the same as our middle solution
- 1 <= score if your solution was better or equal to our best solution
The algorithms are compared based on the number of expanded nodes - the number of calls to the legal\_actions function! Thee total score is computed as the minimum taken over scores from all test cases (different than `test_agent.py`).

You can get up to 6 points from this section if your minimal score (minimal over all the tasks) is greater or equal to 1. You get 4 points if it is greater or equal to 0.5 and 2 points if it is greater or equal to 0.

## MonteCarloAgent
In this part, you have to implement Monte Carlo Tree Search applied to 2-player zero-sum games. You will work with the same interface as before, but you will not use any heuristic function. You have to fill all missing parts of the class `MonteCarloAgent`. To help you get started, we prepared a data structure for storing your tree and also prepared methods, which we think might be useful to you. You are not required to use your structure, but the interface (public methods) of `MonteCarloAgent` has to stay the same. Discard all other methods as you see fit. You need to use UCT sampling to pass our evaluation. You can get 4 points if your solution was correct and 0 points otherwise.
