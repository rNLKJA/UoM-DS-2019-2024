### Informed Search Algorithms

> **Heuristic functions**<br>Heuristic is a function which is used in informed serach, and it finds the most promising path. It takes the current state of the agent as its input and producres the estimation of how close agent is from the goal. The heuristic method, however, might not always give the best solution, but it guaranteed to find good solution in reasonable time. Heuristic function estimates how close a state is to the goal. It is represented by h(n), and it calculates the cost of an optimal path between the pair of states. The value of the heursitic function is always positive.

**Admissbility of the heuristic function is given as: h(n) <= h*(n).

Here h(n) is heurstic cost, and h*(n) is the estimated cost. Hence heuristic cost should be less than or equal to the estimated cost.

### Pure Heuristic Search

Pure heuristic search is the simplest form of heuristic search algorithms. It expands nodes based on their heuristic value h(n). It maintains two lists, OPEN and CLOSED list. In the CLOSED list, it places those nodes which have already expanded and in the OPEN list, it places nodes which  have yet not been expanded.

On each iteration, each node n with the lowest heuristic value is expanded and generates all its successors and n is palced to the closed list. The algorithm continuous unit a goal state is found.

### Best-First Search Algorithm (Greedy Search)

Greedy best-first search algorithm always selects the path which appears best at the moment. It is the combination of depth-first search and breadth-first search algorithm. It uses the heuristic function and search. Best-frist search allows us to take the advantages of both algorithms. With the help of best-first search, at each step, we can choose the most promising node. In the best first search algorithm, we expand the node is closest to the goal node and the closest cost is estimated by heuristic function, i.e. f(n) = g(n) where h(n) = estimated cost from node n to the goal. The greedy best first algorithm is implemented by the priority queue.

#### Best-First Search Algorithm
```
Step 1: Place the start node into the OPEN list.

Step 2: If the OPEN list is empty, Stop and return failure.

Step 3: Remove the node n, from the OPEN list which has the lowerest value of h(n), and places it in the CLOSED list.

Step 4: Expand the node n, and generate the successors of node n.

Step 5: Check each successor of node n, and find whether any node is a goal node or not. If any successor node is goal node, then return success and terminate the search, else proceed to Step 6.

Step 6: For each successor node, algorithm checks for evaluation function f(n), and then check if the node has been in either OPEN or CLOSED list. If the node has not been in both list, then add it to the OPEN list.

Step 7: Return to Step 2.
```

Idea: use an evaluation function for each node - estimate of "desirabiilty" => Expand most desirable unexpanded node.<br>
Implementation: QUEUEINGFN = insert successors in decreasing order of desirability, more desirable state = high hierachy = primary explored node.

| Advantages                                                                                      | Disadvantages                                                              |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Best first search can switch between BFS and DFS by gaining the advantages of both algorithms   | It can behave as an unguided depth-first search in the worst case scenario |
| This algorithm is more efficient than BFS and DFS algorithm                                     | It can get stuck in a loop as DFS                                          |
|                                                                                                 | This algorithm is not optimal                                              |

**Time Complexity**: The worst case time complexity of Greedy best first search is O(b^m).

**Space Complexity**: The worst case space complexity of Greedy best first search is O(b^m). Where m is the maximum depth of the search tree.

**Complete**: Greedy best-first search is also incomplete, even if the given state space is finite.

**Optimal**: Greedy best first search algorithm is not optimal.

### Greedy Search

Evaluation fucntion h(n) (heuristic) = estimate of cost from n to goal

e.g. h_{SLD}(n) = a heuristic function evaluate the straight-line distance 

Greedy search expands the node that appears to be closet to goal

**Time Complexity**: O(b^m), but a good heuristic can give dramatic improvement on searching performance

**Space Complexity**: O(b^m), keeps all nodes in memory

**Complete**: No - can get stuck in loops

**Optimal**: Greedy search is not optimal


#### A* Search Algorithm

```
Step 1: Place the starting node in the OPEN list.

Step 2: Check if the OPEN list is empty or not, if the list is empty then return failure and stops.

Step 3: Select the node from the OPEN list which has the smallest value of evaluation function (g + h), if node n is goal node then return success and stop, otherwise to Step 4.

Step 4: Expand node n and generate all its successors, and put n into the closed list. For each successor n', check wheter n' is already in the OPEN or CLOSET list, if not then compute evaluation function for n' and place into Open list.

Step 5: Else if node n' is already in OPEN and CLOSED, then it should be attached to the back pointer which reflects the lowest g(n') value.

Step 6: Back to step 2.
```

Idea: avoid expanding paths that are already expensive

Evalutation function f(n) = g(n) + h(n)
- g(n) = cost so far to reach n (path cost)
- h(n) = estimated cost to goal from n
- f(n) = estimated total cost of path through n to goal

A* search uses a admissible heuristic i.e. h(n) <= h*(n) is the true cost from n. e.g. h_{SLD}(n) never overestimates the actual road distance.

| Advantages                                                             | Disadvantages                                                                                 |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| A* search algorithm is the best algorithm than other search algorithms | It doesn't always produce the shortest path as it mostly based on heursitcs and approximation |
| A* search algorithm is optimal and complete                            | A* search algorithm has some complexity issues                                                |
| This algorithm can solve very complex problems                         | The main drawback of A* is memory requirement as it keeps all generated nodes in the memory, so it is not proatical for various large-scale problems |

**Points to remember**
- A* algorithm returns the path which occurred first, and it doesn't search for all remaining path.
- The efficiency of A* algorithm depends on the quality of heuristic.
- A* algorithm expands all nodes which satisfy the condition f(n).

**Time Complexity**: The time complexity of A* search algorithm depends on heuristic function, and the number of nodes expanded is exponential to the depth of solution d. So the time complexity is O(b^d), where b is the branching factor.

**Space Complexity**: The space complexity of A* search algorithm is O(b^d).

**Complete**: A* algorithm is complete as long as:
- branching factor is finite
- Cost at every action is fixed

**Optimal**: A* search algorithm is optimal if it follows below two conditions:
- Admissible: the first condition requires for optimality is that h(n) should be an admissible heuristic for A* tree search. An admissible heuristic is optimistic in nature.
- Consistency: Second required condition is consistency for only A* graph-search. Cost function must be monotonic increase.

If the heuristic function is admissible, then A* tree search will always find the least cost path.

### Iterative Improvement Algorithms

In many optimization problems, path is irrelevant; the goal state itself is the solution. 

Then state space = set of "complete" configurations:
- Find optimal configuration, e.g. Travelling Salesperson Problem
- or, find configuration satisfying constraints, e.g. n-queens.

In such cases, can use iterative improvement algorithms, keep a single "current" state and try to improve it.

An iterative improvement algorithm has a constant space, suitable for online as well as offline search.