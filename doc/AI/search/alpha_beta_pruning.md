### Alpha-Beta Pruning

Alpha-beta pruning is a modified version of the minimax algorithm. It is an optimization technique for the minimax algorithm. As we have seen in the minimax search algorihtm that the number of game states it has examine are exponential in depth of the tree. Since we cannot eliminate the exponent, but we can cut it to half. Hence there is a technique by which without chekcing each node of the game tree we can compute the corerct minimax decision, and this technique is called pruning. This involves two threshold parameter Alpha and Beta for future expansion, so it is called alpha-beta pruning. It is also called Alpha-Beta Algorithm. Alpha-beta pruning can be applied at any depth of a tree, and sometimes it not only prune the tree leaves but also entire sub-tree. The two-parameter can be defined as: 

- *Alpha*: The best (highest-value) choice we have found so far at any point along the path for maximier. The initial value of alpha is -inf.
- *Beta*: The best (lowest-value) choice we have found so far at any point along the path for minimier. The initial value of beta is inf.

The alpha-beta pruning is a standard minimax algorithm returns the same move as the standard algorithm does, but it removes all the nodes which are not really affecting the final decision but making algorithm slow. Hence by pruning these node, it makes the algorithm fast.

### Condition for Alpha-Beta Pruning

The main condition which required for alpha-beta pruning is alpha >= beta.

### Key points about Alpha-Beta Pruning
- The MAX player will only update the value of alpha
- The MIN player will only update the value of min
- While backtracking the tree, the node values will be passed to upper nodes instead of values of alpha and beta
- We will only pass the alpha, beta values to the child nodes

### Pseudocode for Alpha-Beta Pruning
```python
def minimax(node, depth: int, alpha: float, beta: float, maximizingPlayer: bool) -> float:
    # check terminate condition
    if depth == 0 or terminate(node):
        return evaluation(node)
    
    # maximizingPlayer 
    if maximingPlayer:
        max_score = -math.inf
        for child in available_moves(node):
            eval_score = minimax(child, depth-1, alpha, beta, False)
            max_score = max(max_score, eval_score)
            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break
        return max_score
    
    # minimizingPlayer
    if not maximizingPlayer:
        min_score = math.inf
        for child in available_moves(node):
            eval_score = minimax(child, depth-1, alpha, beta, True)
            min_score = min(min_score, eval_score)
            beta = min(beta, eval_score)
            if alpha >= beta:
                break
        return min_score
```

### Move ordering in Alpha-Beta pruning:

The effectiveness of alpha-beta pruning is highly dependent on the order in which each node is examined. Move order is an important aspect of alpha-beta pruning. It can be two types:
- **Worst ordering**: In some cases, alpha-beta pruning algorithm does not prune any of the leaves ofthe tree, and works exactly as minimax algorithm. In this case, it also consumes more time because of alpha-beta factors, such a move of pruning is called worst order. In this case, the best move occurs on the right side of the tree. The time complexity for such an order is O(b^m).
- **Ideal order**: The ideal ordering for alpha-beta pruning occuers when lots of pruning happens in the tree, and the best moves occur at the left side of the tree. We apply DFS hence it first search lfet of the tree and go deep twice as minimax algorithm in the same amount of time. Complexity in ideal ordering is O(b^{m/2}).

### Rules to find good ordering

Following are some rules to ifnd good ordering in alpha-beta pruning:
- Occur the best move from the shallowest node
- Order the nodes in the tree such that the best nodes are checked first
- Use domain knowledge while finding the best move. 
- We can bookkeep the states, as there is a possibility that states may repeat