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

**Local Maximum**: Local maximum is a state which is better than its neighbor states, but there is also another state which is higher than it.

**Global Maximum**: Global maximum is the best possible state of state space landscape. It has the highest value of objective function.

**Flat local maximum**: It is a flat space in the landscape diagram where an agent is currently present.

**Shoulder**: It is a plateau region which has an uphill edge.

#### Types of Hill Climbing Algorithm
- Simple hill climbing
- Steepest-Ascent hill climbing
- Stochatic hill climbing

##### Simple hill climbing
Simple hill climbing is the simplest way to implement a hill climbing algorithm. It only evaluates the neighbor node state at a time and selects the first one which optimizes current cost and set it as a current state. It only checks it's one successor state, and if it finds better than the current state, then move else be in the same state. This algorithm has the following features:

- Less time consuming.
- Less optimal solution and the solution is not guaranteed.

```
Step 1: Evaluate the inital state, if is goal state then return success and STOP

Step 2: Loop Unitl a solution is found or there is no new operator left to apply

Step 3: Select and apply an operator to the current state

Step 4: Check new state:
    - If it is goal state, then return success and quit
    - Else if it is better than the current state then assign new state as a current state
    - Else if not better than the current state then return to step 2

Step 5: EXIT
```

##### Steepest-Ascent hill climbing
The steepest-ascent hill climbing algorithm is a variation of simple hill climbing algorithm. This algorithm examines all the neighbor nodes of the current state, and selects one neighbor node which is cloest to the goal state. This algorithm consumes more time as it searches for multiple neighbors.

```
Step 1: Evaluate the initial state, if it is goal state then return success and stop, else make current state as initial state

Step 2: Loop until a solution is found or the current state does not change
    - Let SUCCESSOR be a state such that any successor of the current state will be better than it
    - For each operator that applied to the current state:
        - Apply the new operator and generate a new state
        - Evaluate the new state
        - If it is goal state, then return it and quit, else compare it to the SUCCESSOR
        - If it is better than SUCCESSOR, then set new state as SUCCESSOR
        - It SUCCESSOR is better than current state, then set current state to SUCCESSOR

Step 3: EXIT
```

##### Stochastic hill-climbing

Stochastic hill climbing does not examine for all its neighbor before moving. Rather, this search algorithm selects one neighbor node at random and decides whether to choose it as a current state or examine another state.

### Problems in Hill Climbing Algorithm

1. **Local Maximum**: A local maximum is a peak state in the landscape which is better than each of its neighboring states, but there is another state also present which is higher than the local maximum.

    <img src="./hill-climbing-algorithm-in-ai2.png" />
    
    Backtracking technique can be a solution of the local maximum in state space landscape. Create a list of the promising path so that the algorithm can backtrack the search space and explore other pahts as well.
    
2. Plateau: A plateau is the flat area of the search space in which all neighbor states of the current state continas the same value, because of this algorithm does not find any best direction to move. A hill-climbing search might be lost in the plateau area.

    <img src="./hill-climbing-algorithm-in-ai3.png" />
    
    The solution for plateau is to take big steps or very little steps while searching, to solve the problem. Randomly select a state which is far away from the current state so it is possible that the algorithm could find non-plateau region.
    
3. Ridges: A ridge is a special form of the local maximum. It has an area which is higher than its surrounding areas, but itself has a slop, and cannot be reached in a single move.

    <img src="./hill-climbing-algorithm-in-ai4.png" />
    
    With the use or bidirectional search, or by moving in different directions, we can improve this problem.
    
### Simulated Annealing

A hill climbing algorithm which never makes a move towards a lower value guaranteed to be incomplete because it can get stuck on a local maximum. And if an algorithm applies a random walk, by moving a successor, then it may complete but not efficient. Simulated Annealing is an algorithm which yields both efficiency and completeness.

In mechanical term Annearling is a process of hardening a metal or glass to high temperature than cooling gradually, so this allows the metal to reach a low-energy crystalline state. The same process is used in simulated annealing in which algorithm  picks a random move, instead of picking the best move. If the random move improves the state, then it follows the same path. Otherwise, the algorithm follows the path which has a probability of less than 1 or it moves downhill and chooses another path.