### Hill Climbing Algorithm in Artificial Intelligence 

Hill climbing algorithm is a local search algorithm which continuously moves in the direction of increasing elevation/value to find the peak of the mountain or best solution to the problem. It terminates when it reaches a peak value where no neighbor has a higher value.

Hill climbing algorithm is a technique which is used for optimizing the mathematical problems. One of the widely discussed examples of Hill climbing algorithm is Traveling-salesman Problem in which we need to minimize the distance traveled by the salesman.

It is also called greedy local search as it only looks to its good immediate neighbor state and not beyond that.

A node of hill climbing algorithm has two components which are state and value.

Hill Climbing is mostly used when a good heuristic is available.

In this algorithm, we don't need to maintain and handle the search tree or graph as it only keeps a single current state.

#### Features of Hill Climbing

- **Generate and Test variant**: Hill Climbing is the variant of Generate and Test method. The Generate and Test method produce feedback which helps to decide which direction to move in the search space.
- **Greedy approach**: Hill-climbing algorithm search moves in the direction which optimizes the cost.
- **No backtracking**: It doesn't backtract the search space, as it doesn't remember the previous state.

#### State-space Diagram for Hill Climbing

The state-space landscape is a graphical representation of the hill-climbing algorithm which is showing a graph between various states of algorithm and Objective function/Cost.

On y-axis we have taken the function which can be an objective function or cost function, and state-space on the x-axis. If the function on y-axis is cost then, the goal of search is to find the global minimum and local minimum. If the function y-axis is objective function, then the goal of the search is to ifnd the global maximum and local maximum.

<img src="./hill-climbing-algorithm-in-ai.png" />

#### Different regions in the state space landscape