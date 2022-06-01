### Mini-Max Algorithm in Artificial Intelligence

Minimax algorithm is a recurisive or backtracking algorithm which is used in decision making and game theory. It provides an optimal move for the player assuming that opponent is also playing optimally. Minimax algorithm uses recursion to search through the game-tree. Minimax algorithm is mostly used for game playing in AI. Such as Chess, Checkers, tic-tac-toe, go, and various two-players game. This algorithm compute the minimax decision for the current state. In this algorithm, two players play the game, one called MAX and other is called MIN. Both the players fight it as the opponent player gets the minimum benefit while they get the maximum benefit. Both Players of the game are opponent of each other, where MAX will select the maximized value and MIN will select the minimized value. The minimax algorithm performs a depth-first search algrotihm for the exploration of the comlete game tree. The minimax algorithm proceeds all the way down to the terminal node of the tree, then backtrack the tree as the solution.

### Pseudocode for Minimax Algorith

```python
def minimax(node, depth: int, maximizingPlayer: boolean):
    # check ther terminate state
    if depth == 0 or node is terminate(node):
        return evaluation(node)
    
    # maximizingPlayer will try to maximizing the game result
    if maximizingPlayer:
        max_score = -math.inf
        for child in available_moves(node):
            eval_score = minimax(child, depth-1, False)
            max_score = max(max_score, eval_score)
        return max_score
    
    # minimizingPlayer will try to minimizing the game result
    if not maximizingPlayer:
        min_score = math.inf
        for child in available_moves(node):
            eval_score = minimax(child, depth-1, True)
            min_score = min(min_score, eval_score)
        return min_score
```

### Properties of Minimax algorithm

**Complete**: Minimax algorithm is complete, it will definitely find a solution (if exist) in the finite search tree.

**Optimal**: Minimax algorithm is optimal if both opponent are player optimally.

**Time complexity**: As it performs DFS for the game-tree, so the time complexity of minimax algorithm is **O(b^m)**, where b is branching factor of the game-tree, and m is the maximum depth of the tree.

**Space complexity**: Space complexity of minimax algorithm is aslo similar to DFS which is **O(bm)**.

### Limitation of the minimax algorithm

The main drawback of the minimax algorithm is that it gets really slow for complex games such as Chess, go, etc. This type of games has a huge branching factor, and the player has lots of choices to decide. This limitation of the minimax algorithm can be improved from [alpha-beta pruning](./alpha_beta_pruning.md).