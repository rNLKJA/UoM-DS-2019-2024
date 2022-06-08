### Search Algorithm in Aritificial Intelligence

Search algorithms are one of the most important areas of Artificial Intelligence. This topic will explain all about the search algorithms in AI.

#### Problem-solving agents

In artificial intelligence, Search techniques are universal problem-solving methods. Rational agetns or problem solving agents in AI mostly used these search stretegies or algorithms to solve a specific problem and provide the best result. Problem-solving agents are the goal-based agents and use atomic representation.

#### Search algortihm terminologies
   
- **Search**: Searching is a step by step procedure to solve a search-problem in a given search space. A search problem can have three main factors:
    
    - **Search Space**: Search space represents a set of possible solutions, which a system may have.
    - **Start State**: It is a state from where agent begins the search.
    - **Goal test**: It is a function which observe the current state and returns whether the goal state is achieved or not.
    
- **Search tree**: A tree representation of search problem is called search tree. The root of the search tree is the root node which is corresponding to the initial state.
- **Actions**: It gives the description of all the available actions to the agent.
- **Transition model**: A description of what each action do, can be represented as a transition model.
- **Path Cost**: It is a function which assigns a numeric cost to each path.
- **Solution**: It is an action sequence which leads from the start node to the goal state.
- **Optimal Solution**: If a solution has the lowest cost among all solutions.

#### Properties of search algorithms

- **Completeness**: A search algorithm is said to be comlete if it guarantees to return a solution if at least any solution exists for any random input.
- **Optimality**: If a solution found for an algorithm is guaranteed to be the best solution (lowest path cost) among all other solutions, then such a solution for is said to be an optimal solution.
- **Time Complexity**: Time complexity is a measure of time for an algorithm to complete its task.
- **Space Complexity**: It is the maximum storage space required at any point during the search, as the complexity of the problem.

#### Types of search algorithms

| Uniformed/Blind Search               | Informed Search   | 
| ------------------------------------ | ----------------- |
| Breadth first search                 | Best first search |
| Uniform cost search                  | A* search         |
| Depth first search                   |                   |
| Depth limited search                 |                   |
| Iterative deeping depth first search |                   |
| Bidirectional search                 |                   |

[**Uniformed/Blind Search**](./uninformed.md)

The uniformed search does not contain any domain knowledge such as closeness, the location of the goal. It operates a brute-force way as it only includes information about how to travese the tree and how to identify leaf and goal nodes. Uninformed search applies a way in which search tree is searched without any information about the search space like initial state operators and test for the goal, so it is also called blind search. It examines each node of the tree until it achieves the goal node.

[**Informed Search**](./informed.md)

Informed search algorithms use domain knowledge. In an informed search, problem information is available which can guide the search. Informed search strategies is also called a Heuristic search. A heuristic is a way which might not always guaranteed for best solutions but guaranteed to find a good solution in reasonable time. Informed search can solve much complex problem which could not be solved in another way.