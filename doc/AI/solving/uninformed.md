### Uninformed Search Algorithms

Uninformed search is a class of general-purpose search algorithms which operats in brute-force way. Uninformed search algorithms do not have additional information about state or search space other than how to traverse the tree, so it is also called blind search.

#### Breadth-first search
- Breadth-first search is the most common search strategy for traversing a tree or graph. This algorithm searches breadthwise in a tree or graph, so it is called beradth-first search.
- BFS algorihtm stars searching from the root node of the tree and expands all successor node at the current level before moving to nodes of next level.
- The breadth-first serach algorithm is an example of a general-graph search algorithm.
- Breadth-first search implemented using FIFO queue data structure.

**Completeness**: BFS is **complete**, which means if the shallowest goal node is at some finite depth, then BFS will find a solution.

**Time complexity** of BFS algorithm can be obtained by the number of nodes traversed in BFS until the shallowest Node. Where d = depth of shallowest solution and b is a node at every state. T(b) = 1+b^2+b^3+...+b^d = O(b^d).

**Space complexity** of BFS algorithm is given by the memory size of frontier which is O(b^d).

**Optimality**: BFS is complete, which means if the shallowest goal node is at some finite depth, then BFS will find a solution.

#### Depth-first search
- Depth-first search is a recursive algorithm for traversing a tree or graph data structure.
- It is called the depth-first search because it starts from the root node and follows each path to its greatest depth node before moving to the nexrt path.
- DFS uses a stack data structure for its implementation.
- The process of the DFS algorithm is similiar to the BFS algorithm.

> Backtracking is an algorithm technique for finding all possible solutions using recursion.

**Completeness**: DFS search algorithm is **complete** within finite state space as it will expand every node within a limited search tree.

**Time complexity** of DFS will be equivalent to the node traversed by the algorithm. It is given by: T(n) = 1 + n^2 + n^3 + ... + n^m = O(n^m). Where m = maximum depth of any node and this can be much larger than d (Shallowest solution depth).

**Space complexity**: DFS algorithm needs to store only single path from the root node, hence space complexity of DFS is equivalent to the size of the fringe set, which is O(bm).

**Optimality**: DFS search algorithm is **non-optimal**, as it may generate a large number of steps or high cost to reach to the goal node.

#### Depth-limited search
A depth-limited search algorithm is similar to depth-first search with a predetermined limit. Depth-limited search can solve the drawback of the infinite path in the Depth-first search. In this algorithm, the node at the depth limit will treat as it has no successor nodes further.

Depth-limited search can be terminated with two *conditions of failure*:
- Standard failure value: it indicates that problem does not have any solution.
- Cutoff failure value: it defines no solution for the problem within a given depth limit.

**Completeness**: DLS Search algorihtm is **complete** if the solution is above the depth-limit.

**Time complexity** of DLS algorithm is O(b^ℓ).

**Space complexity** of DLS algorithm is O(bℓ).

**Optimality**: Depth-limited search can be viewed as a speicial case of DFS, and it is also **non-optimal** even ℓ > d.

#### Uniform cost search   
Uniform-cost search is a searching algorithm used for traversing a weighted tree or graph. This algorithm comes into play when a different cost is available for each edge. The primary goal of the uniform-cost search is to find a path to the goal node which has the lowest cumulative cost. Uniform-cost search expands nodes according to their path costs from the root node. It can be used to solve any graph/tree where the optimal cost is in demand. A uniform-cost search algorithm is implemented by the priority queue. It gives maximum priority to the lowest cumulative cost. Uniform cost search is equivalent to BFS algorithm if the path cost of all edges is the same.

**Completeness**: Uniform-cost search is complete, such as if there is a solution, UCS will find it.

**Time complexity**: Let C* is cost of the optimal solution, and ε is each step to get closer to the goal node. Then the number of steps is C* / ε + 1. Here we have taken + 1, as we start from state 0 and end to C* / ε. Hence, the worst time complexity of Uniform-cost search is O(b^{1 + [C* / ε]}).

**Space complexity**: The same logical is for space complexity so, the worst-case space complexity of Uniform-cost search is O(b^{1 + [C* / ε]}).

**Optimality**: Uniform-cost search is always optimal as it only selects a path with the lowest path cost.

#### Iterative deepening depth-first search 
The iterative deepening algorithm is a combination of DFS and BFS algorithms. This search algorithm finds out the best depth limit and does it by graudally increasing the limit until a goal is found.

This algorithm performs depth-first search up to a certain "depth-limit", and it keeps increasing search depth limit after each iteration until the goal node is found.

The search algorithm combiens the benefits of Breadth-first search's fast search and depth-first search's memory efficiency.

The iterative search algorithm is useful uninformed search when search space is large, and depth of goal node is unknown.

**Completeness**: This algorithm is complete if the branching factor is finite.

**Time complexity**: Suppose b is the branching factor and depth is d then the worst-case time complexity is O(b^d).

**Space complexity**: The space complexity of IDDFS will be O(bd).

**Optimality**: IDDFS algorithm is optimal if path cost is a non-decreasing function of the depth of the node.

#### Bidirectional search     
Bidirectional search algorithm runs two simultaneous searchs, one form initial state called as forward-search and other from goal node called as backward-search, to find the goal node. Bidirectional search replacces one single search graph with two small subgraphs in which one starts the search from an initial vertex and other starts from goal vertex. The search stops when these two graphs intersect each other.

Birectional search can use search techniques such as BFS, DFS, DLS, etc.

**Completeness**: Bidirectional Search is complete if we use BFS in both searches.

**Time complexity**: Time complexity of bidirectional search using BFS is O(b^d).

**Space complexity**: Space complexity of bidirectional search is O(b^d).

**Optimality**: Bidirectional search is optimal.

#### Time Complexity, Space Complexity, Completeness, Optimality Comparisons

- d = depth of shallowest solution
- b = branching factor, a node at every state
- m = maximum depth of any node
- ℓ = depth limit parameter

| Uninformed Search Algorithm            | Time Complexity   | Space Complexity  | Completeness                           | Optimality Comparisons                                                   |
| -------------------------------------- | ----------------- | ----------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| Breadth-first search                   | O(b^d)            | O(b^d)            | Complete                               | Optimal                                                                  |
| Depth-first search                     | O(n^m)            | O(bm)             | Complete                               | Non-optimal                                                              |
| Depth-limited search                   | O(b^ℓ)            | O(bℓ)             | Complete if solution is above ℓ        | Non-optimal                                                              |
| Uniform cost search                    | O(b^{1+[C* / ε]}) | O(b^{1+[C* / ε]}) | Complete if there is a solution        | Optimal                                                                  |
| Iterative deepening depth-first search | O(b^d)            | O(bd)             | Complete if branching factor is finite | Optimal if path cost is a non-decreasing function of depth of the node   |
| Bidirectional search                   | O(b^d)            | O(b^d)            | Complete if both search use BFS        | Optimal                                                                  |

#### Advantages & Disadvantages of Different Search Algorihtms

| Uninformed Search Algorithm            | Advantages      | Disadvantages    |
| -------------------------------------- | --------------- | ---------------- | 
| Breadth-first search                   | BFS will provide a solution if any solution exists<br>If there are more than one solutions for a given problem, then BFS will provide the minimal solution which requires the least number of steps | It requires lots of memeory since each levl of the tree must be saved into memeory to expand the next level<br>BFS neesd lots of time if the solution is far away from the root node. |
| Depth-first search                     | DFS requires very less memory as it only needs to store a stack of nodes on the path from root node to the current node<br>It take less time to reach to the goal node than BFS algorithm (If it traverses in the right path) | There is the possibility that many states keep re-occuring, and there is no guarantee of finding the solution |
| Depth-limited search                   | Depth-limited search is memory efficient | Depth-limited search also has a disadvantage of incompleteness<br>It may not be optimal if the problem has more than one solution |
| Uniform cost search                    | Uniform cost search is optimal because at every state the path with the least cost is chosen | It doesn't care about the number of steps involve in searching and only concered about path cost. Due to which this algorithm may be stuck in a infinite loop |
| Iterative deepening depth-first search | It combines the benefits of BFS and DFS search algorithm in terms of fast search and memory efficiency | The main drawback of IDDFS is that it repeats all the work of the previous phase |
| Bidirectional search                   | Bidirectional search is fast<br>Bidirectional search requires less memory | Implementation of the bidirectional search tree is difficult<br>In bidirectional search, one should know the goal state in advance | 