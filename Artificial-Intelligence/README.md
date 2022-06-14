<img src="https://images.unsplash.com/photo-1531746790731-6c087fecd65a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1406&q=80" width=100% />

[[Wikipedia](https://en.wikipedia.org/wiki/Artificial_intelligence)] Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

**Four Approaches to defining AI**
- Thinking like a human
    - figure out how we think through introspection or experimentation (cognitive modelling).
    - self-awareness is important: "I think therefore I am".
    - Humans feel emotions and don't always think (or act) rationally.
- Thinking rationally
    - the problem is how do we define rationality, this could be defined via *logic* since *logic* has greatly influenced AI.
- Acting like a human
- Acting rationally
    - The rational agent: perform actions that will (most likely) achieve one's goals.
    
    > A rational agent acts to maximize the expectation of the performance measure, conditioned on the percept sequence to date, and whatever built-in knowledge the agent possesses.

>  "It is a branch of computer science by which we can create intelligent machines which can behave like a human, think like humans, and able to make decisions." (Java T Point) 

**Goals of Artificial Intelligence**
- Replicate human intelligence
- Solve knowledge-intensive tasks
- An intelligent connection between perception and action
- Building a machine that can perform tasks that requires human intelligence such as:
    - Proving a math theorem
    - Play chess game
    - Plan some surgical operation
    - Self-driving a car
- Creating some systems can exhibit intelligent behaviour, learn new things by itself, demonstrate, explain, and can advise its user.

| **Advantages of Artificial Intelligence** | **Disadvantages of Artificial Intelligence** |
|:---------------------------------------- |:------------------------------------------- |
| High accuracy with less error             | High cost                                    |
| High-speed                                | Can't think out of the box                   |
| High reliability                          | No feelings and emotions                     |
| Useful for risky areas                    | Increase dependency on machines              |
| Digital Assistant                         | No original creativity                       |
| Useful as a public utility                |                                              |

**Artificial Intelligence Types**

| **Based on Capabilities**                          | **Based on functionality**                               |
| -------------------------------------------------- | -------------------------------------------------------- |
| Weak AI / Narrow AI | Reactive Machines | 
| General A    | Limited Memory       |
| Super AI             | Theory of Mind       |
|                                                 | Self-Awareness       |

---

## Intelligent Agent 

An agent is any entity that perceives and acts in an environment. The percept sequence is defined as the complete history of content the agent has perceived. Agent behaviour is defined by an agent function mapping the given percept sequence to an action. An agent program is responsible for implementing the agent function within some physical system.

We evaluate agent behaviour through the performance measure. A rational agent acts to maximize the expectation of the performance measure, conditioned on the percept sequence to date, and whatever built-in knowledge the agent possesses.

A self-learning agent may undertake actions to modify future percepts, and/or adjust the agent function as it accumulates experience.

- Types of Agent
    
    **Agent Architectures**<br>
    Agents can be grouped into five classes based on their degree of perceived intelligence and capability. All these agents can improve their performance and generate better action over time. These are given below:
    - *Simple reflex agent*：only chooses actions based on the current percept and ignores all preceding information.
    - *Model-based reflex agent*：maintains some internal state which depends on the percept history, useful if the current environment cannot be fully described by the current percept.
    - *Goal-based agents*：makes decisions to achieve a set of predefined goals, in addition to maintaining an internal state. In practice usually superseded by utility-based agents.
    - *Utility-based agent*：compares the desirability of different environment states via a utility function. This allows the comparison of different goal states and action sequences and tradeoffs between different goals.
    - *Learning Agent*: can learn from its past experiences or it has learning capabilities. It starts to act with basic knowledge and then can act and adapt automatically through learning. 
    
    *Difference between a goal-based agent and a utility-based agent*.
        - While a goal-based agent can identify action sequences that achieve a given goal, utility-based agents can choose between alternative action sequences that satisfy a goal, by using a utility measure on the sequence of states in the solution.
    
    Agent model: characterise requirements for an agent in terms of its percepts, actions, environment and performance measure.
    
    Agent types: choose and justify the choice of agent type for a given problem.
    
    Environment types: characterise the environment for a given problem.

    <details close>
    <summary>Questions [True/False]</summary>

    - An agent that senses only partial information about the state cannot be perfectly rational.
        - FALSE
        - Rationality refers to the optimality of decisions according to some performance measure conditioned upon the evidence provided by the percept sequence, in addition to whatever prior knowledge it has.
    - There exist task environments in which no pure reflex agent can behave rationally.
        - TRUE
        - Pure reflex agents cannot maintain a history of past actions/percepts, so no environment where optimality requires conditioning on the previous state(s) satisfies this criterion.
    - Suppose an agent selects its action uniformly at random from the set of possible actions. There exists a deterministic task environment in which this agent is rational.
        - TRUE
        - Consider the situation where all actions are taken to yield the same rewards.
    - An agent which chooses the action with the highest expected value under the performance measure is superior to all other agents in the long run.
        - 'Highest expected value' was left deliberately ambiguous as a discussion point.
        - TRUE
            - If we assume this references to maximization of the total cumulative performance measure in the future, then the agent is optimal by definition.
        - FALSE
            - If this instead refers to immediate reward observed at each stage, choosing the action with the highest expected value at the current timestep is greedy, and may be suboptimal, as aficionados of dynamic programming can attest
            - Sometimes it is necessary to sacrifice immediate reward for long-term optimality. 
    - Every agent is rational in an unobservable environment.
        - FALSE
        - It is possible to build prior knowledge into an agent about acceptable behaviour in the absence of observations. An agent lacking this prior knowledge may take suboptimal actions in an unobservable environment.

    </details>

- Intelligent Agent

    Any agent is composed of four parts:
    - *Percepts*: percepts or observations of the environment, made by sensors
    - *Actions*: which affect the environments, made actuators
    - *Environment*: where the agent exists
    - *Performance measure*: performance measurement evaluate the desirability of environment states

    In short, an agent can be anything that perceives its environment through sensors and act upon that environment through actuators. An Agent runs in the cycle of perceiving, thinking, and acting.
    
    An AI system can be defined as the study of the rational agent and its environment. The agents sense the environment through sensors and act on their environment through actuators. An AI agent can have mental properties such as knowledge, belief, intention, etc.
    
    <details close>
    <summary>Questions</summary>
        
    - Do agent functions exist which cannot be implemented by any agent program? If yes, give a counterexample.
        - Yes
        - if the agent function requires solving an undecidable problem - e.g. suppose the agent function required one to solve a Diophantine equation or decide whether an arbitrary Turing machine halts. THen there is no agent program capable of implementing this agent function.
    - Suppose we keep the agent program fixed but speed up the machine by a factor of two. Does this change the agent function? Will your answer change depending on the environment type?
        - Depends on the program and the environment
            - Dynamic environment
                - It speedup may allow the agent to act sooner to choose different, better actions.
            - Static environment
                - the agent function is unchanged, only speeding up the processing speed.
    
    
    </details>
    
- Agent Environment

    An environment is everything in the world that surrounds the agent, but it is not a part of an agent itself. An environment can be described as a situation in which an agent is present.

    The environment is where the agent lives, operates and provides the agent with something to sense and act upon it. An environment is mostly said to be non-feministic.
    
    **Environment Properties**
    - *Fully/partially observable*: The environment is fully observable if the agent's sensors capture the complete state of the environment at every point in time. Imperfect sensors/models or a fundamental inability to capture the environmental state lead to partial observability.
    - *Single/multi-agent*: Self-explanatory, although care must be taken in deciding which entities in the environment must be considered fellow agents.
    - *Deterministic/stochastic*: if future states of the environment are a special case of a nondeterministic environment that admits a probabilistic model of environmental phenomena.
    - *Episodic/sequential*: If an agent's percept sequence is divided into noninteracting episodes, where the agent executes a single action based on the current percept, the environment is episodic. In a sequential environment, the current action may affect the future environment state and hence future decisions.
    - *Static/Dynamic*:If the environment can change when the agent is deliberating (executing the agent program), then the environment is dynamic, otherwise static.
    - *Discrete/continuous*: If the environment has a finite number of distinct states, then the agent only has to contend with a finite set of percepts, and the environment is discrete, otherwise continuous. Similarly, if an agent can choose between a finite number of actions, the action set is discrete.
    
    <details close>
    <summary>Questions</summary>
    
    For each of the following activities, characterize the task environment in terms of its properties: 
    - Playing soccer
        - Partially observable, stochastic, sequential, dynamic, continuous, multi-agent
    - Shopping for used books on the internet
        - Partially observable, deterministic, sequential, static, single agent.
        - This can be multi-agent and dynamic if we buy books via auction, or dynamic if we purchase on a long enough scale that the book offers change.
    - Bidding on an item at an auction
        - Fully observable, stochastic (assuming non-rational actors), sequential, static, discrete, multi-agent.
    
    </details>

- Turing Test in AI

    Turing Test was introduced by Alan Turing in his 1950 paper, "Computing Machinery and Intelligence", which considered the question, "Can Machine think?"
    
    Problem with Truing test:<br>
    The Turing test is not reproducible, constructive or amenable to mathematical analysis.

---

## Problem-solving 

```
# Problem-solving agent
function SIMPLE-PROBLEM-SOLVING-AGENT(p) returns an action
    inputs: p, a perception of the current environment
    static: s, an action sequence, initially empty
            state, some description of the current world state
            g, a goal, initially null
            problem: a problem formulation
    
    state <- UPDATE-STATE(state, p)
    
    if s is empty then
        g <- FORMULATE-GOAL(state)
        problem <- FORMULATE-PROBLEM(state, g) # need to define the states and operators
                                               # state is described as how the agent stores the perceived information
                                               # for example, to let a robot move to some point, we need to know 
                                               #              the current robot coordinates, e.g. robot joint angle. or
                                               #              to solve an 8 puzzle, the board state is represented by the 
                                               #              the integer location of the tile
                                               
        s <- SEARCH(problem)

    # find valid, optimal, rational actions
    action <- RECOMMENDATION(s, state)
    s <- REMAINER(s, state)
    
    return action
```

- Search Algorithms

    Search algorithms are one of the most important areas of Artificial Intelligence. This topic will explain all about the search algorithms in AI.
    
    Basic idea: an offline, simulated exploration of state space by generating successors of already-explored states (a.k.a. expanding states).
    
    ```
    function GENERAL-SEARCH(problem, strategy) returns a solution or failure
        initialize the search tree using the initial state of the problem
        
        loop do
            if there are no candidates for expansion then return failure
            
            choose a leaf node for expansion according to the strategy
                if the node contains a goal state then return the corresponding solution
                else expand the node and add the resulting nodes to the search tree
        end
    
    # implementation
    
    function GENREAL-SEARCH(problem, QUEING-FN) returns a solution or failure
        nodes <- MAKE-QUEUE( MAKE_NODE(INITIAL-STATE[problem]) )
        loop do
            if nodes is empty then return failure
            
            node <- REMOVE-FRONT(nodes) # since the QUEUE is a priority queue, always pop the first node
            if GOAL-TEST[problem] applied to STATE(node) succeeds then return node
            nodes <- QUEUING-FN( nodes, EXPAND(node, OPERATIONS[problem]) )
                                        # EXPAND function creates new nodes, filling in various fields and using OPERATORS (or ACTIONS) of the problem to create the corresponding states.
        end
    ```
    
    **States vs. Nodes**
    - A state is a (representation of) a physical configuration.
    - A node is a data structure constituting part of a search tree including parent, children, depth, path, and cost g(x).
    
    > States do not have parents, children, depth, or path cost.
    
    <details close>
    <summary>Questions</summary>
    
    - In BFS and DFS, an undiscovered node is marked discovered when it is first generated and marked processed when it has been completely searched. At any given moment, several nodes might simultaneously be in the discovered state.
        - Describe a graph on n vertices and root v such that Θ(n) nodes are simultaneously in the discovered state when conducting BFS starting from v.
            - A graph of depth 1 where v has n children, the tree becomes a list where the search time will be Θ(n)。
        - Repeat the above question, but for DFS.
            - A graph of depth n with a single linear branch of children extending from v. In this case the tree will be a stick.
        - Describe a graph on n vertices and root v such that at some point Θ(n) remains undiscovered, while Θ(n) nodes have been processed during DFS starting from v.
            - A graph of depth n/2 with two linear equal-length branches of children extending from v. Note the case in the second question is not acceptable here as any node along the chain is not marked processed until the recursion returns.
    
    - Recall the iterative deepening search (IDS) strategy
        - Derive the time complexity of iterative deepening search on a graph with branching factor b and minimum goal depth d.
            - Within each iteration, IDS exhaustively enumerates all nodes up to the current depth. At depth n, the size of this set is b^n, Nodes at all levels apart from d are re-generated multiple times.
            - The node in the penultimate layer is generated twice, the nodes in the third-last layer are generated three times, and so on, with the root node generated d times. This implies that an upper bound on the number of generated nodes, and hence the time complexity is:
                - d + (d + 1)b + (d + 2)b^2 + ... + 2d^{d-1} + b^d  = O(b^d). So asymptotically approaches to BFS.
        - Can you think of a situation where IDS performs much worse than a regular depth-first search?
            - Consider the situation where the graph is a linear chain with depth d. DFS terminates in d iteration but IDS takes \sum^d_{n=1} n(n+1)/2 iterations.
        - BFS and IDS are not optimal in general. State a necessary condition on the cost function g: paths → R for both methods to be optimal.
            It- Necessary for g to be a non-decreasing function of depth - otherwise, the cost of a path from the start to the goal state could be decreased by taking a path with a greater number of actions. This can be formally proved via induction.
    
    - A graph G = (V, E) is connected if a path exists between any two vertices v1,v2 ∈ V. A connected component of a graph is a maximal subset Vc ⊆ V such that there exists a path between any two vertices in Vc. Describe an algorithm, using graph search, that accepts an arbitrary graph G as input and counts the number of connected components of G.
        - For undirected graphs with a finite state space, BFS/DFS will eventually enumerate all vertices within a connected component, starting at an arbitrary node within the connected component, since the vertex ordering is irrelevant:
            - Start at an arbitrary node.
            - Run BFS/DFS until the frontier is exhausted - all processed vertices belong to the first connected component.
            - Restart BFS/DFS from an arbitrary undiscovered node until all vertices have been marked processed, counting the number of restarts time.
    
    - Your tutor’s job is to arrange n ill-behaved children in a straight line, facing the front. They are given a list of m statements of the form "child I hate child j". If I hate j, they cannot put me somewhere behind j, because I am then capable of throwing something at j.
        - They have enlisted your help to design an algorithm that orders the line (or says it is not possible), using a depth-first search. Analyze the running time of your algorithm.
            - From a directed graph with the vertices corresponding to children and an edge pointing from I to j if j hates i.
            - Topologically sort the graph as follows:
                - Form a stack that will hold the topologically sorted nodes.
                - Perform DFS, and push nodes σ onto stack only after all descendants of σ in the graph have been fully explored/processed. If there is an edge I -> j, this guarantees that i is only pushed onto the stack after j.
                - The top node on the stack always has no incoming edge from any vertex in the stack.
                - Repeatedly popping nodes off the stack yields a topological ordering where I will always precede j if j hates i.
        - For the class photo, the children must be arranged in rows such that if I hate j, I must be in a lower-numbered row than j, otherwise, it is possible for i to drop something on j. Design an efficient algorithm to find the minimum number of rows needed.
            - Construct the graph as before. The seating position of children not involved in a hate relation is irrelevant (these represent singleton disconnected nodes), so we want to find the height of the connected component of the graph of maximum height. This can be found in O(m+n) time using BFS to enumerate all vertices within each connected component.
    </details>
        
- Uniform Search Algorithm

    Uninformed search is a class of general-purpose search algorithms that operates in a brute-force way. Uninformed search algorithms do not have additional information about state or search space other than how to traverse the tree, so it is also called blind search.

- Heuristic Function

    A heuristic is a function that is used in Informed Search, and it finds the most promising path. It takes the current state of the agent as its input and produces the estimation of how close the agent is to the goal. The heuristic method, however, might not always give the best solution, but it guaranteed to find a good solution in a reasonable time. The heuristic function estimates how close a state is to the goal. It is represented by h(n), and it calculates the cost of an optimal path between the pair of states. The value of the heuristic function is always positive.
    
    **Admissbility of the heuristic function is given as: h(n) <= h*(n).

    Here h(n) is the heuristic cost, and h*(n) is the estimated cost. Hence heuristic cost should be less than or equal to the estimated cost.
    
    Admissible heuristics can be derived from the exact solution cost of a relaxed version of the problem. 
    - For example, if the rules of the 8-puzzle are relaxed so that a tile can move anywhere then h1(n) gives the shortest solution.
        - h1(n) = number of misplaced tiles
        - h2(n) = total Manhattan distance
    - If the rules are relaxed so that a tile can move to any adjacent square, then h2(n) gives the shortest solution.
    
    h(σ) ≤ min(cost(σ → goal)) = h*(σ), ∀σ ∈ Σ
    
    An admissible heuristic defines an optimal cost estimate; it never overestimates the cost to the goal. Note sacrificing admissibility does not completely void the heuristic function - you lose guarantees of optimality, but non-admissible heuristics may allow your search method to run faster as fewer nodes in the priority queue need to be expanded - in general, there is a tradeoff between speed/optimality.
    
    **Consistency/Monotony**
    
    h(σ) ≤ cost(σ, σ′,A) + h(σ′) ∀σ ∈ Σ, A(σ) = σ′
    
    This is the triangle inequality applied to the costs of graph states. Here cost(σ, σ′, A) denotes the cost to the agent of taking action A in state σ to end up in state σ′.
    
    > Admissibility results in optimality but may not be completeness in general.

- Informed Search Algorithm

     Informed Search algorithms have information on the goal state which helps in more efficient searching. This information is obtained by a function that estimates how close a state is to the goal state (Geeks for Geeks, 2021). 
     
    <details close>
    <summary>Questions</summary>

    - Suppose that we run a greedy best-first (graph) search algorithm with the heuristic function for node σ set to the negative of the path cost, h(σ) = −g(σ). What sort of search will the greedy search emulate?
        - This emulates depth-first search, noting that for frontier F: again f(σ) = argmax(σ).
    - The heuristic path algorithm is a best-first search where the evaluation function contains a relative weighting factor w: f(σ) = (2 −w)g(σ) + wh(σ).
        - What kind of search does this perform for w = 0, w = 1, and w = 2?
            - w = 0: uniform cost search
            - w = 1: A* search
            - w = 2: Gready Best First Search
        - For what values is it optimal, assuming that the heuristic h is admissible? Hint: recall the condition for optimality of A∗.
            - Note that f(σ) = (2 - w)(g(σ) + w / (2 - w) h(σ)), a scaled version of A* with heuristic w/(2-w)h(σ). We require this to be admissible, hence require w / (2-w) ≤ 1 ⇒ w ≤ 1. Non-negativity of the heuristic requires then 0 ≤ w ≤1.

    - Recall that state generation refers to the creation of the data structure σ representing the state, whereas state expansion refers to the generation of all children of the present node.
        - Should informed search algorithms apply the goal test upon state generation or state expansion, for the resulting search to be optimal? Does it make a difference from uninformed search algorithms?
            - Applying the goal test on generation does not guarantee optimality in general for informed search methods (and UCS).
            - There may be lower-cost paths living in the frontier that we have not explored. 
            - However, if we apply the goal test on expansion, all nodes in the frontier will have equal or larger path costs. For uninformed search algorithms (except UCS) this does not affect optimality but will affect time complexity. For example, BFS time complexity becomes O(b^{d+1}) for goal test upon expansion.

    - n vehicles occupy squares (1,1) through (n,1) (i.e., the bottom row) of an n×n grid. The vehicles must be moved to the top row but in reverse order; so the vehicle i that starts in (i,1) must end up in (n −I + 1,n). On each time step, every one of the n vehicles can move one square up, down, left, or right, or stay put; but if a vehicle stays put, one other adjacent vehicle (but not more than one) can hop over it. Two vehicles cannot occupy the same square.
        - Find an expression for the size of the state space in terms of n.
            - n^2! / (n^2 - n)!
        - Find an expression for the branching factor in terms of n. 
            - 5^n
        - Give a (non-trivial) admissible heuristic h_i for the number of moves it takes for a single-vehicle I starting in (xi, Yi) to its goal state, (n −I + 1,n).
            Manhattan Distance
        - Is \sum_i h_i an admissible heuristic for this problem? How about max{h1,...,hn} or min{h1,...,hn}?
            - The first will not be admissible general as vehicles can move simultaneously. Note for the 'hop' operation one vehicle must remain and the other jump 2 steps, meaning the total distance travelled by all cars per move is bounded above by n. The total distance D moved by all vehicles from the start to the goal state is bounded below by n min{h_1,...,h_n} ≤ D. Hence the minimum number of steps required to attain D is min{h_1,...,h_n}. 
        
    </details>

- Uninformed Search vs. Informed Search (Geeks for Geeks, 2021)
    
    | Uninformed Search                                     | Informed Search                                  |
    | ----------------------------------------------------- | ------------------------------------------------ |
    | It doesn't use knowledge for the searching process        | It uses knowledge for the searching process      |
    | It finds solutions slow as compared to informed search | It finds solutions more quickly                   |
    | It is always complete                                 | It may or may not be complete                    |
    | Cost is high                                          | Cost is Low                                      |
    | It consumes moderate time                             | It consumes less time                            |
    | No suggestion is given regarding the solution in it   | It provides the direction regarding the solution |
    | It is more lengthy while implementation               | It is less lengthy while implementation          |
    | DFS, BFS, DLS, UCS, IDDFS                             | Best-First search, A* search                     |

- Hill Climbing Algorithm

    The Hill climbing algorithm is a local search algorithm that continuously moves in the direction of increasing elevation/value to find the peak of the mountain or the best solution to the problem. It terminates when it reaches a peak value where no neighbour has a higher value.
    
    Hill climbing algorithm is a technique that is used for optimizing mathematical problems. One of the widely discussed examples of the Hill climbing algorithm is Traveling-salesman Problem in which we need to minimize the distance travelled by the salesman.
    
    It is also called greedy local search as it only looks to its good immediate neighbour state and not beyond that.
    
    A node of the hill-climbing algorithm has two components which are state and value.
    
    Hill Climbing is mostly used when a good heuristic is available.
    
    In this algorithm, we don't need to maintain and handle the search tree or graph as it only keeps a single current state, past information is not useful.
    
    ```
    function HILL-CLIMBING(problem) returns a solution state
        inputs: problem
        local variables: current, a node
                         next, a node
        current <- MAKE-NODE(INITIAL-STATE[problem])
        loop do
            next <- a highest-valued successor of current
            if VALUE[next] < VALUE[current] then return current
            current <- next
        ends
    ```
    
    The problem with the hill-climbing algorithm is it can get stuck on local maxima.
    
- Means-Ends Analysis

    Means-Ends Analysis is a problem-solving technique used in Artificial intelligence for limit search in AI programs. It is a mixture of Backward and Forward search techniques. The MEA process is centred on the evaluation of the difference between the current state and the goal state.

- Summary Search Strategies/Algorithm Properties

    A strategy is defined by picking the order of node expansion. Strategies are evaluated along the following dimensions: completeness, time complexity, space complexity, and optimality.
    - Time complexity calculates several nodes generated/expanded.
    - Space complexity calculate the maximum number of nodes in memory.
    - Completeness evaluation is the strategy/algorithm always find a solution if one exists.
    - Optimality evaluation is the strategy/algorithm always finding a least-cost solution.

    |                                                            | Time Complexity | Space Complexity | Completeness                                     | Optimality Comparisons                                                   |
    | ---------------------------------------------------------- | --------------- | ---------------- | ------------------------------------------------ | ------------------------------------------------------------------------ |
    | (Uninformed Search) Breadth-first search                   | O(b^d)          | O(b^d)           | Complete                                         | Optimal                                                                  |
    | (Uninformed Search) Depth-first search                     | O(n^m)          | O(bm)            | Complete if search tree is finite                | Non-optimal                                                              |
    | (Uninformed Search) Depth-limited search                   | O(b^ℓ)          | O(bℓ)            | Complete if solution is above ℓ                  | Non-optimal                                                              |
    | (Uninformed Search) Uniform cost search                    | O(b^{1+[C*/ε]}) | O(b^{1+[C*/ε]})  | Complete if there is a solution                  | Optimal                                                                  |
    | (Uninformed Search) Iterative deepening depth-first search | O(b^d)          | O(bd)            | Complete if branching factor is finite           | Optimal if path cost is a non-decreasing function of depth of the node   |
    | (Uninformed Search) Bidirectional search                   | O(b^{d/2})      | O(b^{d/2})       | Complete                                         | Optimal if both search use BFS                                           |
    | (Informed Search) Best-First search                        | O(b^m)          | O(b^m)           | Incomplete                                       | Non-optimal                                                              |
    | (Informed Search) Greedy search                            | O(b^m)          | O(b^m)           | Incomplete                                       | Non-optimal                                                              |
    | (Informed Search) A* search                                | O(b^d)          | O(b^d)           | Complete if finite branching factor & fixed cost | Optimal if heuristic functions is admissible and consistency             |
    
    - d = depth of shallowest solution
    - b = branching factor, a node at every state
    - m = maximum depth of any node
    - ℓ = depth limit parameter
    
- Problem formulation usually requires abstracting away real-world details to define a state space that can feasibly be explored. 
- There are lots of variety of uniform search strategies.
- Iterative deepening search uses only linear space and not much more time than other uninformed search algorithms.

- Heuristics help reduce search costs, however, finding an optimal solution is still difficult. 
- Greedy best-first is not optimal but can be efficient.
- A* search is complete and optimal but is prohibitive in memory.
- Hill-climbing methods operate on complete-state formulations, require less memory, but are not optimal.

---

## Adversarial Search 

- Adversarial Search
    
    Adversarial search is a search, where we examine the problem which arises when we try to plan ahead of the world and other agents are planning against us.
    
    Usually, a game will have an unpredictable opponent and time limits for play. Thus our agent should use adversarial search to find a solution with a form contingency plan and approximate the best solution in time bounds.
    
    **Types of Games**: game types could be described in the following table.
    
    |                       | Deterministic                      | Chance                                    |
    | --------------------- | ---------------------------------- | ----------------------------------------- |
    | Perfect information   | chess<br>checkers<br>go<br>othello | backgammon<br>monopoly                    |
    | Imperfect information |                                    | bridge<br>poker<br>scrable<br>nuclear war |
    
    Games illustrate several important points about AI.
    - perfection is unattainable => must approximate and make trade-offs 
    - uncertainty limits the value of look-ahead 
    - programs could use TDLearning (self-learning agent) to learn the game from past game experience or use supervised learning like gradient decent search to find the best solution.
        - problems using unsupervised learning with gradient descent search are not a good approach for learning in game playing search.
            - Delayed reinforcement: reward resulting from an action may not be received until several time steps later, which also slows down the learning.
            - Credit assignment: need to know which action(s) was responsible for the outcome.
    
- Minimax Search

    > Minimax is a perfect play for deterministic, perfect-information games.

    The Minimax algorithm is a recursive or backtracking algorithm that is used in decision-making and game theory. It provides an optimal move for the player assuming that the opponent is also playing optimally. The Minimax algorithm uses recursion to search through the game tree. The Minimax algorithm is mostly used for game playing in AI. Such as Chess, Checkers, tic-tac-toe, go, and various two-players game. This algorithm computes the minimax decision for the current state. In this algorithm, two players play the game, one called MAX and the other called MIN. Both the players fight it as the opposing player gets the minimum benefit while they get the maximum benefit. Both Players of the game are opponents of each other, where MAX will select the maximized value and MIN will select the minimized value. The minimax algorithm performs a depth-first search algorithm for the exploration of the complete game tree. The minimax algorithm proceeds down to the terminal node of the tree, then backtrack the tree as the solution.
    
    ```
    function MINIMAX-DECISION(game) returns an operator
        for each op in OPERATIONS[game] do
            VALUE[op] <- MINIMAX_VALUE( APPLY(op, game), game )
        end
        return the op with the highest VALUE[op]
    
    function MINIMAX-VALUE(state, game) returns a utility value
        if TERMINAL-TEST[game](state) then
            return UTILITY[game](state)
        else if MAX is to move in state then
            return the highest MINIMAX-VALUE of SUCCESSORS(state)
        else if MIN is to move in state then
            return the lowest MINIMAX-VALUE of SUCCESSORS(state)
    ```

    *Properties of minimax algorithm*
    - **Complete**: Minimax algorithm is complete, it will find a solution (if exist) in the finite search tree.
    - **Optimal**: Minimax algorithm is optimal if both opponents are players optimally.
    - **Time complexity**: As it performs DFS for the game tree, so the time complexity of the minimax algorithm is **O(b^m)**, where b is the branching factor of the game tree, and m is the maximum depth of the tree.
    - **Space complexity**: The space complexity of the minimax algorithm is also similar to DFS which is **O(bm)**.

    In the real world, usually, there is a resource limit for the minimax algorithm. For example, a player has limited time or there is not enough computational power for successors' exploration. What we can do is use two approaches:
    - cutoff test: e.g. depth limit (perhaps add quiescence search)
        
        MINIMAX-CUTOFF is identical to MINIMAX-VALUE except we add depth limit as an additional terminal check, and use the utility function to replace the evaluation function. 
        
    - evaluation function = estimated desirability of position
    These two approaches are effective in shortening the time complexity and space complexity of minimax search. 

    *Limitation of the minimax algorithm*:
    - The main drawback of the minimax algorithm is that it gets really slow for complex games such as Chess, go, etc. This type of game has a huge branching factor, and the player has lots of choices to decide. This limitation of the minimax algorithm can be improved from alpha-beta pruning.

- Alpha-beta Pruning

    Alpha-beta pruning is a modified version of the minimax algorithm. It is an optimization technique for the minimax algorithm. As we have seen in the minimax search algorithm that the number of game states it has to examine is exponential in depth of the tree. Since we cannot eliminate the exponent, we can cut it in half. Hence there is a technique by which without checking each node of the game tree we can compute the correct minimax decision, and this technique is called pruning. This involves two threshold parameters Alpha and Beta for future expansion, so it is called alpha-beta pruning. It is also called Alpha-Beta Algorithm. Alpha-beta pruning can be applied at any depth of a tree, and sometimes it not only prunes the tree leaves but also the entire sub-tree. The two-parameter can be defined as: 

    - *Alpha*: The best (highest-value) choice we have found so far at any point along the path for maximizer. The initial value of alpha is -inf.
    - *Beta*: The best (lowest-value) choice we have found so far at any point along the path for the minimizer. The initial value of beta is inf.

    The alpha-beta pruning is a standard minimax algorithm that returns the same move as the standard algorithm does, but it removes all the nodes which are not affecting the final decision but making the algorithm slow. Hence pruning this node, makes the algorithm fast.
    
    Alpha-beta pruning is a modified version of the minimax algorithm. It is an optimization technique for the minimax algorithm.
    
    ```
    function MAX-VALUE(state, game, α, β) returns the minimax value of the state
        inputs: state, the current state in the game
                game, game description
                α, the best score for MAX along the path to state
                β, the best score for MIN along the path to state

        if CUTOFF-TEST(state) then return EVAL(state)
        for each s in SUCCESSORS(state) do
            α <- MAX(α, MIN-VALUE(s, game, α, β))
            if α >= β then return β
        end
        return α
        
    function MIN-VALUE(state, game, α, β) returns the minimax value of state
        if CUTOFF-TEST(state) then return EVAL(state)
            β <- MIN(β, MAX-VALUE(s, game, α, β))
            if β <= α then return α
        end
        return β
    ```
    
    Pruning does not affect the final results. 
    
    The order of search nodes is important in alpha-beta pruning. If we have the worst ordering, the time complexity will be the same as minimax O(b^m). However, if we have an ideal ordering, then the time complexity will reduce in half since the best node is always on the left side of the tree, and complexity will be O(b^{m/2}). As a result, the agent could double the depth of search.
    
    <details close>
    <summary>Questions</summary>
    
    - Here you will investigate the fundamentals of minimax and alpha-beta pruning. Consider noughts and crosses on a 3 ×3 grid. Define Xn as the number of rows, columns or diagonals with exactly n crosses and no noughts. Define On analogously. Assume our utility function assigns +1 to any position with X3 = 1 and −1 to any position with O3 = 1, with all other positions having zero utility. For nonterminal positions, use the evaluation function f(s) = 3X_2(s) + X_1(s) −(3O_2(s) + O_1(s)) . Assume the cross player goes first.
        - Find a (possibly approximate) expression for the size of the state space in terms of n, i.e. how many possible configurations exist.
            - b = 9, m = 9, naively obvious bounds are 9^9, 9! or 3^9. Note number of possible games > several unique states as a game considers state orders.
        - Construct the whole game tree starting with an empty board down to depth 2 (i.e., one X and one O on the board), taking symmetry into account. Mark the evaluations of all positions at depth 2.
            - Root has a minimax value of +1
            - At depth of 1, MIN encounters the following values:
                - X at top left: -1
                - X at top mid: -2
                - X at centre: +1
            - Continuous
        - Use minimax to mark the backed-up values for the positions at depth 1, and use these values to choose the optimal move for the X player.
            - See above
        - Circle the nodes at depth 2 that would not be evaluated if alpha-beta pruning was applied, assuming the nodes were generated in the optimal order for alpha-beta pruning.
            - In total, 8 of the 12 leaf nodes can be pruned by alpha-beta with optimal move ordering.
        
    - Why is minimax less reasonable than expected minimal as a practical decision-making principle for a complex agent acting in the real world?
        - The real world is often stochastic and in complex environments, with inherent randomness, it is generally necessary to use probability distribution to parameterize our uncertainty. 
        - More practically, minimax assumes an optimal adversary - it is more likely that optimal responses to an agent's actions are not guaranteed. 
        - Minimax makes decisions based on the potentially low-probability worst-case scenario. 
        - Recall that expects minimal uses the expected utility over its children. This weights events and their corresponding outcomes according to their probability in the expectation, so may be more robust to highly negative but very rare events.
    
    - The minimax algorithm returns the best move for MAX under the assumption that MIN plays optimally. What happens when MIN plays suboptimally?
        - Assertion: For every game tree, the utility obtained by MAX using minimax decisions against a suboptimal MIN will never be lower than the utility obtained playing against an optimal MIN.
            - Consider a MIN node whose children are terminal nodes. For a suboptimal opponent, the utility of the node is greater or equal to the value of MIN played optimally, U(σ_min) >= U*(σ_min).
            - Hence the value of the parent MAX node, which assumes the maximum among all its children, is greater than the case for an optimal MIN player.
            - Repeat this argument until the root node and conclude that the assertion is true, the max player always enjoys a payoff greater or equal to the case of an optimal MIN player.
    
    - Minimax and alpha-beta as presented assume a zero-sum setting with bounded utility functions. This question prompts you to consider the consequences of removing these assumptions.
        - Describe how the minimax and alpha-beta algorithms change for two-player, non-zero-sum games in which each player has a distinct utility function and both utility functions are known to both players.
            - For non-zero-sum games between two players A and B, a single value is not sufficient to describe the utility of a node, and should instead be replaced by a tuple of values (v_A, v_B), giving the minimax value/utility of the state for each respective player. At each stage, the player whose turn it is to move selects the action that yields the highest v_i value.
            - Alpha-beta pruning is not possible in this context as the game is not necessarily adversarial and the two players are not necessarily competing against each other - the utility function may be entirely independent, and neither player may be particularly fussed about what the other player does. There may no longer be situations where one player will never let the other player down a particular branch of the game tree.
        - What happens in the almost cooperative case, where the utility functions differ by at most a positive constant k > 0, i.e. UA = UB + k?
            - The optimal strategy in this scenario is for both players to cooperate to reach the maximum-valued leaf - the agents work together to achieve a mutually desirable goal.
    
    </details>
    
- [ ] Temporal Difference Learning (TDLeaf(λ))

    Temporal difference (TD) learning refers to a class of model-free reinforcement learning methods that learn by bootstrapping from the current estimate of the value function. These methods sample from the environment, like Monte Carlo methods, and perform updates based on current estimates, like dynamic programming methods.

    While Monte Carlo methods only adjust their estimates once the outcome is known, TD methods adjust predictions to match later, more accurate, predictions before the outcome is known (Wikipedia, 2022).
    
    From the above, we know supervised learning is for single-step prediction, but Temporal Difference (TD) learning is for multi-step prediction. 
    - Correctness of prediction not known until several steps later
    - Intermediate steps provide information about the correctness of the prediction
    - TD learning is a form of reinforcement learning
    
    TDLeaf(λ) algorithm combines temporal difference learning with minimax search, the basic idea is to update the weight in the evaluation function to reduce differences in rewards predicted at different levels in the search tree. Good functions should be stable from one move to the next.
    
    **Notations**
    - eval(s, w): evaluation function for state s with parameters w = [w_1, ..., w_k]
    - s_1, ..., s_N: the N states that occurred during a game
    - r(S_N): reward based on outcome of game {+1, 0, -1}
    - s_i^l: best leaf found at max cut-off depth using minimax search starting at state s_i
    
    To convert evaluation score into a reward, we could using r(s_i^l, w) = tanh(eval(s_i^l, w)) where tanh squashes the evaluation score into the range [-1, +1].
    
    ```
    # Algorithm
    
    For i = 1, ..., N-1 Compute temporal difference between successive states
        d_i = r(s_{i+1}^l, w) - r(s_i^l, w)
    
    Update each weight parameter w_j as follows
        w_j <- w_j + \eta \sum^{N-1}_{i=1} \partial r(s_i^l, w) / \partial w_j [\sum^{N-1}_{m=1} λ^{m-i}d_m]
    where \eta is the learning rate
    
    if λ = 0: weights adjusted to move r(s_i, w) towards r(s_{i+1}, w)
        i.e. the predicted reward at the next state
    if λ = 1: weights adjusted to move r(s_i, w) towards r(s_N)
        i.e. the final true reward (better if eval() is unrealistic)
    ```
    
    By using the reinforcement learning in gameplay, with a large number of games experience, the probability of making a move that leads to a loss should go to zero, and the probability of making a move that leads to a win should approach 1 so that the agent will favour the optimal action. This assumes that every move in every possible state has been visited sufficiently many times so that we can obtain a viable estimate of its usefulness.
    
    <details close>
    <summary>Questions</summary>
    
    Recall for a general state s, and parametric evaluation function Eval(s; θ) we would like
Eval(s; θ) ≈ U(s), where U(s) is the backed-up utility of the terminal state that arises from
playing the game out from state s, using minimax up to depth l with the evaluation function
Eval both players.
    
    We have to play s out to completion to make a single weight update based on s. But playing outs to completion generates multiple intermediate states {s_t,s_{t+1},...,s_N} that we may use to refine our weight updates.
    - Assume the objective function is ℓ(θ; s) = 1/2 (U(s) −Eval(s; θ))^2. Show that the gradient descent weight update from playing a state s until completion is: θ ←θ + η (U(s) −Eval(s; θ)) ∇_θ Eval(s; θ).
        
        ```
        This follows from the chain rule:
            ∇_θ ℓ(θ; s) = −∇_θ Eval(s; θ)(U(s) −Eval(s; θ))
        Substitute into the update rule θ ← θ −η∇_θ ℓ(θ; σ) to obtain the result.
        ```
    
    - By using minimax with depth l to look ahead, we only calculate rewards from states of depth l in the future. If so is the state whose reward ultimately propagates back to s via minimax, the weight update rule is: θ ← θ + η U(s) −Eval(s^l; θ)∇_θEval(s^l; θ) To be clear, here Eval(SL; θ) is the evaluation obtained for state s by applying Eval to the leaf nodes of a depth l minimax search from s. The weight update rule tries to make Eval(s_l; θ) a good approximation to the true terminal utility. If {s_t,s_{t+1},...,s_N} are the states encountered in a training game starting at time step t, we define the evaluation function at the terminal state s_N as the utility of s_t. The contribution to the parameter update by passing through state s_t is θ ←θ + η Eval(s^l_N; θ) −Eval(s^l_t; θ)∇_θ Eval(s^l_t; θ). Show the total update to the parameter vector for all N −1 moves made from state s_1 is: θ ← θ + η \sum^{N-1}_{i=1} Eval(s^l_N; θ) − Eval(s^l_i; θ)∇_θ Eval(s^l_i; θ). 
        
        ```
        The parameter update for all N - 1 is the sum of all incremental updates ∆θ_t due to passing through states s_t:
            θ ← θ + \sum^{N−1}_{t=1} ∆θ_t
              = θ + η\sum^{N-1}_{t=1} Eval(s^l_N; θ) −Eval(s^l_i; θ) ∇_θ Eval(s^l_i; θ)
        ```
    
    - Finally, we note that Eval(s_N; θ) −Eval(s^l_i,θ) is a telescoping series: Eval(slN; θ) −Eval(sli; θ) = \sum^{N-1}+{m=1} Eval(s^l_{m+1}; θ) −Eval(s^l_m; θ) Define the temporal differences d_m = Eval(s^l_{m+1}; θ) −Eval(s^l_m; θ). Show the update rule reduces to: θ ←θ + η\sum^{N-1}_{i=1} (∇_θ Eval(s^l_i; θ) \sum^{N-1}_{m=i}d_m).
        - Straight forward from the definitions.
    - We can insert a decay term λ ∈[0,1] which controls the effect of this telescoping: θ ← θ + η\sum^{N-1}_{i=1} (∇_θ Eval(s^l_i; θ) \sum^{N-1}_{m=i} λ^{m-i}d_m). This is the final update rule for TDLeaf(λ). Identify the update rules for the λ = 0 and λ = 1 cases, how does the behaviour of the updates qualitatively change as we increase λ?
    
        ```
        For λ = 0, (noting that lim_{x→0^+}x_0 = 1), only m = i term in the inner sum is non-zero, and the update looks like:
            θ ← θ + η\sum^{N-1}_{t=1} ∇_θ Eval(s^l_t; θ)[Eval(s^l_t+1; θ) −Eval(s^l_t; θ)]
        This is known as the one-step temporal difference update, training is guided to minimize the change in reward between consecutive statuses.
        
        For λ = 1, the telescope collapses to yield:
            θ ← θ + η\sum^{N-1}_{t=1} ∇_θ Eval(s^l_t; θ)[U(s) −Eval(s^l_t; θ)]
        Here training is being guided to minimize the difference between each state's s_t predicted reward with the final state's utility at time step N. This is equivalent to gradient descent using only terminal state utilities as targets.
        Setting 0 < λ < 1 controls how far into the future the algorithm should look ahead in determining its update. Increasing λ interpolates between the short-sighted one-step λ = 0 behaviour and the long-sighted λ = 1 behaviour.
        ```
    
    </details>

- Monte Carlo Search Tree

    In computer science, Monte Carlo tree search (MCTS) is a heuristic search algorithm for some kinds of decision processes, most notably those employed in software that plays board games. In that context, MCTS is used to solve the game tree.
    
    MCTS is an algorithm that figures out the best move out of a set of moves by Selecting → Expanding → Simulating → Updating the nodes in the tree to find the final solution. This method is repeated until it reaches the solution and learns the policy of the game (SAGAR, 2018).
    
- Backtracking
    
    Backtracking refers to a general method of building a sequence of decisions to solve a graph problem incrementally. Suppose we have a partial solution to a problem we would like to extend: A = (a_1, a_2, ..., a_k). To choose next solution component, a_{k+1}:
    - Recursively evaluate every possible consistent with past decisions. When we establish that a partial solution cannot be extended into a complete solution, or is worse than the current best solution, we terminate the recursive call, thereby pruning regions of state space from the search tree which cannot contain a solution (or an optimal solution if we have some notion of optimality). We backtrack to the deepest node with unexpanded children and invoke recursion again.
    - Choose a valid successor (or the 'best' one if we have optimality criterion), A ←A \cup a+{k+1}. Repeat until A becomes a complete solution.
    
    This is implemented as depth-first/recursive traversal with additional logic at each call that helps narrow the search space, using problem constraints to prune subtrees as early as possible.
    
    ```python
    def backracking_dfs(A, k):
        if A = (a_1, a_2, ..., a_k) in solution:
            return A
        else:
            k += 1
            
            # enumerate all possible candidates extending deepest partial solution
            candidate_queue = construct_candidates(A, k)
            
            # while valid children exist, extend solution by recursion
            while candidate_queue is not None:
                A[k] = candidate_queue.pop() # add current candidate to A
                result = backtrack_dfs(A, k)
                
                if result != failure:
                    return result
                A[k] = null # backtrack, remove candidate from A
        return failure
    ```
    
---

## Knowledge Represent 

- Knowledge-Based Agent

    An intelligent agent needs knowledge about the real world to make decisions and reasoning to act efficiently. Knowledge-based agents are those agents who have the capability of maintaining an internal state of knowledge, reasoning over that knowledge, updating their knowledge after observations and taking actions. These agents can represent the world with some formal representation and act intelligently.
    
    Knowledge-based agents are composed of two main parts:
    - Knowledge-base
    - Inference system
    
    A knowledge-based agent must be able to do the following:
    - An agent should be able to represent states, actions, etc.
    - An agent should be able to incorporate new percepts
    - An agent can update the internal representation of the world
    - An agent can deduce the internal representation of the world
    - An agent can deduce appropriate actions

- Knowledge Representation

    Humans are best at understanding, reasoning, and interpreting knowledge. Human knows things, which is knowledge and as per their knowledge they perform various actions in the real world. But how machines do all these things comes under knowledge representation and reasoning. Hence we can describe Knowledge representation as follows:
    - Knowledge representation and reasoning (KR, KRR) is the part of Artificial intelligence which concerned with AI agents thinking and how thinking contributes to the intelligent behaviour of agents.
    - It is responsible for representing information about the real world so that a computer can understand and can utilize this knowledge to solve complex real-world problems such as diagnosis of a medical condition or communicating with humans in natural language.
    - It is also a way that describes how we can represent knowledge in artificial intelligence. Knowledge representation is not just storing data in some database, but it also enables an intelligent machine to learn from that knowledge and experiences so that it can behave intelligently like a human.

- Knowledge Representation Techniques

    There are mainly four ways of knowledge representation which are given as follows:
    - Logical representation
    - Semantic Network Representation
    - Frame Representation
    - Production Rules
    
- Propositional Logic

    Propositional logic (PL) is the simplest form of logic where all the statements are made by propositions. A proposition is a declarative statement that is either true or false. It is a technique of knowledge representation in logical and mathematical form.
    
- Rules of Inference

    In artificial intelligence, we need intelligent computers which can create new logic from old logic or by evidence, so generating conclusions from evidence and facts is termed inference.
    
- First-order logic

    First-order logic is another way of knowledge representation in artificial intelligence. It is an extension of propositional logic. FOL is sufficiently expressive to represent the natural language statements concisely. First-order logic is also known as Predicate logic or First-order predicate logic. First-order logic is a powerful language that develops information about the objects more easily and can also express the relationship between these objects. First-order logic (like natural language) does not only assume that the world contains facts like propositional logic but also assumes the following things in the world:
    - **Objects**: A, B, people, numbers, colours, wards, theories, squares, pits, ...
    - **Relations**: It can be unary relation such: red, round, is adjacent, or n-any relation such as the sister of, the brother of, has colour, comes between, ...
    - **Functions**: Father of, best friend, third inning of, end of, ...
   
- Knowledge Engineering in FOL

    The process of constructing a knowledge base in first-order logic is called as knowledge- engineering. In knowledge engineering, someone who investigates a particular domain, learns the important concepts of that domain, and generates a formal representation of the objects, is known as a knowledge engineer.
    
- Inference in First-order logic

    Inference in First-Order Logic is used to deduce new facts or sentences from existing sentences. Before understanding the FOL inference rule, let's understand some basic terminologies used in FOL.
    
- Unification in FOL

    Unification is the process of making two different logical atomic expressions identical by finding a substitution. Unification depends on the substitution process.
    It takes two literals as input and makes them identical using substitution.
    Let Ψ1 and Ψ2 be two atomic sentences and 𝜎 be a unifier such that, Ψ1𝜎 = Ψ2𝜎, then it can be expressed as UNIFY(Ψ1, Ψ2).

- Resolution in FOL

    Resolution is a theorem proving technique that proceeds by building refutation proofs, i.e., proofs by contradictions. It was invented by a Mathematician John Alan Robinson in the year 1965.

    Resolution is used, if there are various statements given, and we need to prove a conclusion of those statements. Unification is a key concept in proofs by resolutions. Resolution is a single inference rule which can efficiently operate on the conjunctive normal form or clausal form.

- Forward Chaining and Backward Chaining
    
    In artificial intelligence, forward and backward chaining is one of the important topics, but before understanding forward and backward chaining let's first understand from where these two terms came.
    
- Forward Chaining vs. Backward Chaining

    | Forward Chaining | Backward Chaining |
    | ---- | ---- |
    | Forward chaining starts from known facts and applied inference rules to extract more data unit it reaches the goal | Backward chaining starts from the goal and works backward through inference rules to find the required facts that support the goal |
    | It is a bottom-up approach | It is a top-down approach |
    | Forward chaining is known as a data-driven inference technique as we reach the goal using the available data | Backward chaining is known as a goal-driven technique as we start from the goal and divide into sub-goal to extract the facts |
    | Forward chaining reasoning applies a breadth-first search strategy | Backward chaining reasoning applies a depth-first search strategy |
    | Forward chaining tests for all the available rules | Backward chaining only tests for few required rules |
    | Forward chaining is suitable for the planning, monitoring, control, and interpretation application | Backward chaining is suitable for diagnostic, prescription, and debugging applications |
    | Forward chaining can generate an infinite number of possible conclusions | Backward chaining generates a finite number of possible conclusions |
    | It operates in the forward direction | It operates in the backward direction |
    | Forward chaining is aimed for any conclusion | Backward chaining is only aimed for the required data |

- Reasoning in AI

    The reasoning is the mental process of deriving logical conclusions and making predictions from available knowledge, facts, and beliefs. Or we can say, "Reasoning is a way to infer facts from existing data." It is a general process of thinking rationally, to find valid conclusions.
    
- Inductive vs. Deductive Reasoning

    | **Basis for comparison** | **Deductive Reasoning**  | **Inductive Reasoning** |
    | ------------------------ | ------------------------ | ----------------------- |
    | Definition               | Deductive reasoning is the form of valid reasoning, to deduce new information or conclusion from known related facts and information | Inductive reasoning arrives at a conclusion by the process of generalization using specific facts or data |
    | Approach                 | Deductive reasoning follows a top-down approach | Inductive reasoning follows a bottom-up approach |
    | Starts from              | Deductive reasoning starts from premises        | Inductive reasoning starts from conclusion       |
    | Validity                 | In deductive reasoning conclusion must be true if the premises are true | In inductive reasoning, the truth of premises does not guarantee the truth of conclusions |
    | Usage                    | Usage of deductive reasoning is difficult, as we need facts that must be true | Use of inductive reasoning is fast and easy, as we need evidence instead of facts. We often use it in our daily life |
    | Process                  | Theory -> Hypothesis -> Patterns -> Confirmation | Observations -> Patterns -> Hypothesis --> Theory |
    | Argument                 | In deductive reasoning, arguments may be valid or invalid | In inductive reasoning, arguments may be weak or strong |
    | Structure                | Deductive reasoning reaches from general facts to specific facts | Inductive reasoning reaches from specific facts to general facts |

---

## Uncertain Knowledge R 

- Probabilistic Reasoning in AI
    To represent uncertainty, where we are not sure about the predicates, we need uncertain reasoning or probabilistic reasoning.
    
    Probabilistic assertions summarize the effects of:
    - laziness: failure to enumerate exceptions, qualifications, etc.
    - ignorance: lack of relevant facts, initial condition, etc.
    
    Subjective or Bayesian probability:
    - Probabilities related propositions to one's state of knowledge.
    
    These are not claims of some probabilistic tendency in the current situation (but might be learned from past experiences of similar situations).
    
    > NOTE: P( cause | effect ) = P( effect | cause ) * P( cause ) / P( effect )
    
    The probabilities of propositions change with new evidence.
    
    Choice of an event depends on personal preference, however, you might want to check other theories for better decision-making:
    - *Utility Theory* is used to represent and infer preferences.
    - *Decision theory* = utility theory + probability theory
    
    **Prior or unconditional probabilities**: probabilities that are calculated before any (new) evidence.
    
    **Posterior or conditional probabilities**: probabilities that are calculated after any evidence shows up.
    
    Typically, we are interested in the posterior joint distribution of the query variables Y 
    given specific values e for the evidence variables E.
    
    Let the hidden variables be H = X - Y - E, then the required summation of joint entries is done by summing out the hidden variables:
    
    P(Y|E = e) = aP(Y,E = e) = a \sum_n P(Y, E = e, H = h)
    
    The terms in the summation are joint entries because Y, E, and H together exhaust the set of random variables.
    
    Obvious problems:
    - Worst-case time complexity O(d^n) where d is the largest arity.
    - Space complexity O(d^n) to store the joint distribution.
    
    
    **Short Summary**
    - Probability is a rigorous formalism for uncertain knowledge.
    - Joint probability distribution specifies the probability of every atomic event.
    - Queries can be answered by summing over atomic events.
    - For nontrivial domains, we must find a way to reduce the joint size.
    - Independence and conditional independence provide the tools.
    
- Bayes Theorem in AI
    Bayes' theorem is also known as Baye's rule, Bayes' Law, or Bayesian reasoning, which determines the probability of an event with uncertain knowledge. In probability theory, it relates the conditional probability and marginal probabilities of two random events. Bayes' theorem was named after the British mathematician Thomas Bayes. The Bayesian inference is an application of Bayes' theorem, which is fundamental to Bayesian statistics. It is a way to calculate the value of P(B|A) with the knowledge P(A|B). Bayes' theorem allows updating the probability prediction of an event by observing new information from the real world.
    
    ```tex
    Bayes Law: 
        P(H,M) = P(H|M)P(M) = P(M|H)P(H)
        P(H|M) = P(H,M)/P(M)
               = P(M|H)P(H) / P(M)

        P(M|H) = P(H,M)/P(H)
               = P(H|M)P(M) / P(H)
    
    Elements:
        - P(H|M): posterior probability
        - P(H): prior probability
        - P(M|H): sensor model
        - P(M): nomralisatio factor
        
    Normalisation:
        P(M) = P(M|H)P(H) + P(M|¬H)P(H)
        
    Odds Ratio:
        r = P(A|B) / P(¬A|B)
        let p = P(A|B)
            r = p / (1 - p) => r - rp = p => r = p + rp => r = p(1 + r) => p = r / 1 + r (DanielLC, Rubdy & AstraSequi, 2020)
            
    Others:
        P(A|B) + P(A|¬B) = 1
    ```
    
    Conditional Probability: P(X|H) = P(X|H)P(H) / P(X)
    - The likelihood P(H|H) is the conditional probability of the data X given fixed H.
    - The prior P(H) represents information we have that is not part of the collected data X - consider this our pre-existing degree of belief in the hypothesis H before observing any data.
    - The evidence P(H) is the average over all possible values of H, calculated using the law of total probability.
         - P(X) = \sum_h P(X|H = h)P(H = h)
     
     P(H|X) is the posterior distribution, which represents our updated beliefs around the hypothesis now we have observed data X. Bayes' Theorem may also be conveniently expressed in terms of the posterior odds, which circumvents the calculation of the evidence p(H).
     
     > p(H|X) / (H^c|X) = P(X|H)  * P(H) / P(X|H^c) * P(H^c)
     
     Note that H^c may be replaced with ¬H for binary random variables. Conditional probabilities are probabilities as well, so it is straightforward to extend Bayes' Theorem to incorporate extra conditioning on another event Y, provided P(H|Y), P(X|Y) > 0:
     
     > P(H|X, Y) = P(X|H, Y) P(H|Y) / P(X, Y)
     
     Bayes' Theorem is a simple consequence of the definition of conditional probability but has deep consequences. The basic intuition here is that go in one direction. e.g. P(X|H) is easier than finding the probability P(H|X) in the other direction.

    > Odds Ratio = P(M|H)P(H) / P(M|H^c)P(H^c)

- Bayesian Belief Network

    Bayesian networks are probabilistic, because these networks are built from a probability distribution, and also use probability theory for prediction and anomaly detection. Real-world applications are probabilistic, and to represent the relationship between multiple events, we need a Bayesian network. It can also be used in various tasks including prediction, anomaly detection, diagnostics, automated insight, reasoning, time series prediction, and decision-making under uncertainty.
    
    Bayesian network is a simple, graphical notation for conditional independence assertions and hence for compact specification of the full joint distribution.
    
    **Syntax**
    - A set of nodes, one per variable.
    - A directed, acyclic graph (link approximate to "directly influences").
    - A conditional distribution for each node given its parents: P( X_i | Parents(X_i) ).
    
    In the simplest case, conditional distribution is represented as a conditional probability table (CPT) giving the distribution over X_i for each combination of parent values.
    
    A CPT for Boolean X_i with k Boolean parents has 2^k rows for the combinations of parent values. Each row requires one number p for X_i = True (the number for X_i = false is just 1 - p).
    
    If each variable has no more than k parents, the complete network requires O(n * 2^k) numbers. i.e. grows linearly with n, vs. O(2^n) for the full joint distribution.
    
    **Global semantics**: it defines the full joint distribution as the product of the local conditional distributions:
    - P(x_1, ..., x_n) = \prod_{i=1}^n P( x_i | parent(X_i) )
    
    **Local semantics**: each node is conditionally independent of its nondescendants given its parents.
    
    > Theorem: Local semantics === global semantics
    
    **Construct Bayesian Networks**
    ```
    1. Choose an ordering of variable X_1, ..., X_n.
    2. For i = 1 to n:
        add X_i to the network
        select parents from X_1, ..., X_{i-1} such that P(X_i|Parents(X_i)) = P(X_i|X_1,...,X_{i-1})
    
    This choice of parents guarantees the global semantics 
                            P(X_1,...,X_n) = \prod^n_{i=1} P(X_i|X_1,...,X_{i-1}) (chain rule)
                                           = \prod^n_{i=1} P(X_i|Parents(X_i)) (by construction)
    ```
    
    **Inference tasks**
    - *Simple queries*: compoute posterior marginal P(X_i|E = e)
    - *Conjunctive queries*: P(X_i, X_j | E = e) = P(X_i|E = e) = P(X_j | X_i, E = e), probabilistic inference required for P(outcome|action, evidence)
    - *Value of information*: which evidence to seek next?
    - *Sensitivity analysis*: which probability values are most critical?
    - *Explanation*： why do I need a new starter motor?
    
    **Variable elimination**: Basic operations
    
    Variable elimination carries out summations right-to-left, storing intermediate results (factors) to avoid recomputation.
    
    - *Summing out* a variable from a product of factors:
        - move any constant factors outside the summation
        - add up submatrices in pointwise product of remaining factors
    - *Pointwise product* of factors f_1 and f_2:
        - f_1(x_1, ..., x_j, y_1 , ...., y_k) \times f_2(y_1, ..., y_k, z_1, ...., z_l) = f(x_1, ..., x_j, y_1, ..., y_k, z_1, ..., z_l)
        - e.g. f_1(a, b) \times f_2(b, c) = f(a, b, c)
        
    **Inference by enumeration**
    
    A slightly intelligent way to sum out variables from the joint without actually constructing its explicit representation.
    - Recursive depth-first enumeration: O(n) space, O(d^n) time.
    
    > Enumeration is inefficient: repeated computation.
    
    **Irrelavent variable**: Theorem: Y is irrelevant unless Y \in Ancestors({X} \cup E).
    
    **Short Summary**
    - Bayes net provides a natural representation for (casually induced) conditional independence.
    - Topology + CPTs = compact representation of joint distribution.
    - Generally easy for (non)experts to construct.
    - Exact inference by enumeration.
    - Exact inference by variable elimination.

---

## Constraint Satisfaction Problems 

Constraint satisfaction problems (CSPs) are mathematical questions defined as a set of objects whose state must satisfy several constraints or limitations. CSPs represent the entities in a problem as a homogeneous collection of finite constraints over variables, which is solved by constraint satisfaction methods. CSPs are the subject of research in both artificial intelligence and operations research since the regularity in their formulation provides a common basis to analyze and solve problems of many seemingly unrelated families. CSPs often exhibit high complexity, requiring a combination of heuristics and combinatorial search methods to be solved in a reasonable time (Wikipedia).

Standard Search Problem: in a standard search problem, a state is a 'black box', any old data structure that supports goal test, eval, or successor.

CSP has a state which is defined by variables `V_i` with values from Domain `D_i`. The goal test is a set of constraints specifying allowable combinations of values for subsets of variables. A simple example of a formal representation language, CSP allows useful general-purpose algorithms with more power than standard search algorithms.

In other words, we define CSPs in terms of:
- A set of variables X = {X_1, ..., X_n}. Each variable assumes values in its respective domain, D = {D_1, ..., D_n}.
- Constraints C expresses restrictions on the domain of individual variables when considering relationships between variables. For the case of binary constraints, we can represent a CSP as a graph <V, E> where the vertices are the variables and the edges are constraints between pairs of variables.

The CSP is solved when values are assigned to all variables in X without violating the given constraints. The basic idea is to eliminate large regions of search space by identifying combinations of variables/values that violate constraints, then pruning appropriately - this is just backtracking.

**Constraint graph**
- *Binary CSP*: each constraint relates at most two variables.
- *Constraint graph*: nodes are variables, and arcs show constraints.

**Varieties of CSPs**
- Discrete Variables
    - finite domains with size `d`, where `n` is the number of variables in the CSP. Finding a CSP solution requires a time complexity of O(d^n).
        - e.g. Boolean CSPs include Boolean satisfiability (NP-complete).
    - infinite domains (integers, strings, etc.)
        - e.g., job scheduling, variables are start/end days for each job
        - need a constraint language, e.g. StartJob_1 + 5 <= StartJob_3
        - linear constraints solvable, nonlinear undecidable
    - Discrete variables have a finite domain with roughly O(d^n) complete assignments, where n is the number of variables in the CSP. For example, scheduling.
- Continuous Variables
    - e.g. start/end times for Hubble Telescope observations
    - linear constraints solvable in polynomial time by linear programming (LP) methods. For example, Start/End times for Hubble Telescope observations.

**Varieties of Constraints**
- *Unary* constraints involve a single variable. e.g. SA != Green
- *Binary* constraints involve pairs of variables. e.g. SA != WA
- *Higher-order* constraints involve 3 or more variables, e.g. cryptarithmetic column constraints
- *Preference* (soft constraints), e.g. red is better than green often representable by a cost for each variable assignment. Preference is commonly used in constrained optimization problems.

**Real-World CSPs**<br>
In the real world, CSP could be any *scheduling* or *planning* problems.

**Search Methods in Constraints Satisfactory Problems**
- **Applying Standard Search**
    
    ```
    Initial state: the empty assignment, the empty set
    Successor function: assign a value to an unassigned variable that does not conflict with the current assignment.
                        => fail if no legal assignments (not fixable).
    Goal test: the current assignment is complete
    ```
    
    The standard search is the same for all CSPs, every solution appears at depth n with n variables (could apply depth-first search). It will generate branching factor b = (n - l)d at depth l, hence a search tree will have n!d^n leaves. Since the goal is to solve CSPs, therefore path is irrelevant here, so can also use complete-state formulation to find a valid solution.
    
- **Backtracking Search**
    
    Variables assignments are commutative, i.e. [WA = red then NT = green] is the same as [WA = green then WA = red]. Backtracking search only needs to consider assignments to variable at each node => b = d and there are d^n leaves in a search tree. When applying a depth-first search for CSPs with single-variable assignments is called a backtracking search. Backtracking search is the basic uninformed algorithm for CSPs.
    
    ```
    function BACKTRACKING-SEARCH(csp) return solution/failure
        RECURSIVE-BACKTRACKING({}, csp)
    end function
    
    function RECURSIVE-BACKTRACKING(assignment, csp) 
        if assignment is complete then 
            return assignment
        
        var <- SELECT-UNASSIGNED-VARIABLE(Varaibles[csp], assignment, csp)
        
        for each value in ORDER-DOMAIN-VALUES(var, assignment, csp) do
            if value is consistent with assignment given Constraints[csp] then
                add {val = value} to assignment
                result <- RECURSIVE-BACKTRACKING(assignment, csp)
                
                if result != failure then
                    return result
                
                remove {val = value} from assignment # because current assignment value cannot reach the goal state
        return failure
    end function
    ```
    
    Constraint satisfaction problems are NP-hard - there exists no known algorithm for finding solutions to them in polynomial time. This can be attributed to the fact that for an n-variable with maximal domain size d, there are an exponential number of possible assignments, O(d^n), we have to sift through to find a satisfactory assignment. Nevertheless, CSPs tend to have more structure than generic graph search problems, allowing us to use various heuristic methods to often find a solution in an acceptable amount of time.
    
    To improve backtracking efficiency, we could use general-purpose methods that can give huge gains in speed:
    - Which variable should be assigned next?
        
        `var <- SELECT-UNASSIGNED-VARIBALE(Variables[csp], asssignment, csp)`
        
        Solution: use MRV to find the next value
        > **Minimum Remaining Values (MRV)**: choose the successor with the fewest legal values. Selects successor states that are more likely to fail (end as dead leaves of recursion tree).
        
        But, there might exist a tie-breaking among MRV variables or choice of first value, hence we could assign value use
        > **Degree Heuristic**: choose the successor involved in the largest number of constraints on other potential successors. More constraints -> lower branching factor of recursion subtree.
        
    - In what order should its value be tried?
    
        `ORDER-DOMAIN-VALUES(var, assignment, CSP)`
        
        Solution: order value use the Least constraining value
        > **Least constraining value**: Given a variable, assign the value that makes the fewest choices of variables for neighbouring candidates illegal. This permits the maximum remaining flexibility for remaining variables, making it likely to find a complete solution in future.
        
        In short, we looking for a value that brings the most flexibility to the future search tree.
            
    - Can we detect inevitable failure early?
    
        > **Forward Checking**: keep track of remaining legal values for unassigned variables, terminate search when any variable has no legal values
        
        Forward checking propagates information from assigned to unassigned variables but doesn't provide early detection for all failures. Constraint propagation repeatedly enforces constraints locally.
        
        ```
        function FORWARD-CHECKING(csp) returns a new domain for each var
            for each variable X in csp do
                for each unassigned variable Y connected to X do
                    for each value d in Domain(Y)
                        if d is inconsistent with Value(X)
                            Domain(Y) = Domain(Y) - d
            return csp // modified domains
        ```

    - Filter over a CSP
        
        Filter over a CSP notes a solution method but yields a pruned search graph which is faster to traverse. The basic idea is to use local constraints to eliminate illegal variable assignments, by pruning values of the domain for each variable that violates binary constraints. This leads us to the arc-consistency (AC-3) algorithm, which we briefly sketch below. Define an arc as a directed edge in the constraint graph in a CSP. The constraint graph is undirected, but we interpret an undirected edge as two directed edges pointing in opposite directions:
        - A CSP variable is arc-consistent if every value in its domain satisfies all relevant binary constraints.
        - For X, Y variables involved in a constraint, X -> Y arc-consistent if every value X = x has legal assignment Y = y satisfying the constraint on (X, Y).
        - The AC-3 algorithm enumerates all possible arcs in a queue Q and makes X_i arc-consistent w.r.t. X_j by reducing the domains of variables accordingly.
        - If D_i is unchanged, move on to the next arc. Otherwise append all arcs (X_k, X_i) where X_k is adjacent to X_i to the queue.
        - AC-3 is not a solution method but defines an equivalent problem that is faster to traverse via backtracking, as the variables have smaller domains.
        
        > **Arc consistency**: the simplest form of constraint propagation makes each arc consistent.
        
        X -> Y is arc consistent if for every x of X there is at least one value of y of Y that satisfies the constraint between X and Y.
        
        ```
        # Arc-consistency ALgorithm: AC-3 
        function AC-3(csp) returns the CSP, possibly with reduced domains
            input: csp, a binary CSP with variables {X1, X2, ..., Xn}
            
            local variables: queue, a queue of arcs, initially all the arcs in CSP
            while the queue is not empty do
                (Xi, Xj) <- REMOVE-FRIST(queue) # decompose arc into two variables
                if REMOVE-INCONSISTENT-VALUE(Xi, Xj) then
                    for each Xk in NEIGHBORS(Xi) do
                        add (Xk, Xi) to queue
        end function
        
        function REMOVE-INCONSISTENT-VALUE(Xi, Xj) true iff succeeds
            removed <- false
            for each x in Domain(Xi) do
                if no value y Domain(Xi) allows (x, y) to satisfy the constraint Xi <-> Xj then
                    delete x from Domain(Xi)
                    removed <- true
            return removed
        end function
        ```
    
        AC-3 will have a time complexity of O(n^2d^3), it can be reduced to O(n^2d^2). It is hard to detect all arcs since the problem is NP-hard.
        
        Alternative approach: exploit the structure of the consistent graph to find **independent subproblems**.
        
    - Can we take advantage of the problem structure?
    
        Suppose each subproblem has c variables out of n total, the worst-case solution cost n/c * d^c, it's a linear cost in n.
        Unfortunately, completely independent subproblems are rare in practice. However, other graph structures are easy to solve.
        
        **Tree Structure CSPs** 
        
        Theorem: if the constraint graph has no loops, the CSP can be solved in O(nd^2) time. Compare to general CSPs, where the worst-case time is O(d^n). This property also applies to logical and probabilistic reasoning.
        
        ```
        1. Choose a variable as root, order variables from root to leaves such that every node's parent precedes it in the ordering
        2. For j from n down to, apply MakeArcConsistent(Parent(Xj), Xj)
        3. For j from I to n, assign Xj consistently with Parent(Xj)
        ```
        
        **Nearly tree-structured CSPs**
        - Conditioning: instantiate a variable, prune its neighbours' domains
        - Cutset conditioning: instantiate (in all ways) a set of variables such that the remaining constraint graph is a tree
        - Cutset: a set of variables that can be deleted so a constraint graph forms a tree
        
        Cutset size c => runtime O(d^c * (n-c) * d^2), very fast for small c.
        
        **Interactive algorithms for CSPs - Local Search**
        
        Recall from the hill-climbing search, that hill-climbing typically works with "complete" states, i.e. all variables are assigned.
        
        Local search then tries to change one variable assignment at a time.
        
        To apply to CSPs:
        - allow states with unsatisfied constraints (variable selection)
        - Operators reassign varaible values (value selection)
        
        Variable selection: randomly select any conflicted variable.
        Value selection: by min-conflicts heuristic, choose a value that violates the fewest constraints. i.e. Hillclimb with h(n) = total number of violated constraints.
        
    **Short Summary**
    - CSPs are a special kind of problem: 
        - states defined by values of a fixed set of variables
        - goal test defined by constraints on variables values
    - Backtracking = depth-first search with one variable assigned per node
    - Variable ordering and value selection heuristics help significantly
    
    - Forward checking prevents assignments that guarantee later failure
    - Constraint propagation (e.g., arc consistency) does additional work to constrain values and detect inconsistencies
    - The CSP representation allows analysis of problem structure
    
    - Tree-structure CSPs can be solved in linear time
    - Iterative min-conflicts are usually effective in practice.
    
    - The degree heuristic relates to what variable we should assign next and states that we should pick the variable that has the most constraints on the remaining variables.
    - Whereas the least constraining value relates to what order the values should be tried once we have already picked a variable in the CSP. The least constraining value states that we should choose the least constraining value (ie. the one that least restricts the values for other variables)
    
    <details close>
    <summary>Questions</summary>
    
    - The n-Queens problem is a classic benchmark search problem. Given an n x n chessboard, the aim is to place a total of n queens on the board, so that no queen data can capture any other person.
        - Derive a nontrivial upper bound on the size of the state space of general n.
            - Stupid: 2^n^{2} (n^2 squares, 2 possible states for each).
            - Very Very Bad: n^2! / ( (n^2 - n)!n! ) (Permutations of n queens on n x n board).
            - Very Bad: n^2 (1 queen per row)
            - OK: n(n-2)(n-4)...= \prod^{[n/2]-1}_{i=1}(n-2i). (Each queen eliminates at least two squares in next row)
            - Good: No known formula for the exact number of solutions
        - Develop a CSP formulation for this problem. Specify the relevant variables together with their domains, and the constraints between relevant variables.
            - Variables: Position of the queen in row r
            - Represent positions of queens through array Q[1,...,n]. Element r of the array, Q[r], indicates the position of the queen in row r.
            - Domain: Q[r] ∈ {1,...,n}
            - Constraints:
                - Let i != j be row indexes.
                - Q[i] != Q[j] (same column placement forbidden)
                - |Q[j] - Q[i]| != |j - i| (diagonal placment forbidden)
            - Constraint graph
                - Nodes: Variables Q[i]
                - Edges represent constraints between variables. Every queen can theoretically attack every other queen on the chessboard, so our constraint graph is a clique; we have an edge connecting every pair of variables in the CSP. In n-Queens it suffices to consider binary constraints as the queen in row I independently threatens the queens in rows j, k.
        - Design an algorithm, in pseudo-code, that counts the number of valid n-Queens configurations using backtracking. 
            ```python
            def nQueens(Q: List[int], r: int) -> int:
                # Q: vector holding column indices for queen assignments
                # r: row index for recursion
                
                if r == n:
                    return 1
                    
                solutions = 0
                
                for i in range(n):
                    # check if placement is legal, given previous placements
                    legal = True
                    for j in range(r):
                        # I loops over the proposed placement column
                        if (Q[j] == i) or (abs(Q[j] - i) == (r - j)):
                            legal = False
                            break
                
                    # if legal, extend recursively
                    if legal:
                        Q[r] = i
                        solutions += nQueens[Q, r+1]
                
                return solution
                
            Q = [None for _ in range(n)]
            return nQueens(Q, 0)
            ```
    - [Apparent paradox] Explain why it is a good heuristic to choose the variable that is most constrained but the value that is least constraining in a CSP search.
        - The search tree for solutions grows exponentially, but most branches are invalid combinations of assignments.
        - The Minimum Remaining Values (MRV) heuristic chooses the variable most likely to cause a failure - if the search fails early, backtrack and prune the search space. If the current partial solution cannot be expanded into a complete solution, is it better to know earlier instead of wasting time searching exponentially many dead-ends?
        - Because we need to find a single solution, we want to be generous and select the value that allows the most future assignments to avoid conflict. This makes it more likely the search will find a complete solution.
        - This asymmetry makes it better to use MRV to prune exponentially growing search space by choosing least-promising successors first, but increasing the probability of success for all successors via the least constraining value heuristic.
    
    - Use the AC-3 algorithm to show that arc consistency can detect the inconsistency of the partial assignment {WA = Green, V = Red} for the problem of colouring the map of Australia in Figure
    
        <img src='https://github.com/chuangyu-hscy/UoM-DS-2019-2024/blob/main/Artificial-Intelligence/img/map-coloring-constriants-problem.png?raw=true'  />
    
        - Wish to colour states that no neighbouring states share the same colour.
        - To detect inconsistency, reduce the domain of some variable in the graph to zero => no possible colour assignments for that variable.
        
        ```
        Initial value domains
            - WA = {G}
            - V = {R}
            - NT = {R, G, B}
            - SA = {R, G, B}
            - Q = {R, G, B}
            - NSW = {R, G, B}
        
        Pop WA-SA arc from the queue (degree heuristic)
            - SA = {R, B}
        Pop SA-V arc from queue
            - SA = {B}
        Pop NT-WA arc from queue
            - NT = {R, B}
        Pop NT-SA arc from queue
            - NT = {R}
        Pop NT-Q arc from queue
            - Q = {G, B}
        Pop SA-Q arc from queue
            - Q = {G}
        No legal assignment for NSW, CSP is inconsistent, terminate.
        ```
    - What is the time complexity of running the AC-3 arc consistency algorithm on a tree-structured CSP? Give your answer in terms of the number of edges on the constraint graph, E, and the maximum possible domain size, D.
        - A tree-structured CSP has no loops in the constraint graph. We first choose an arbitrary node to serve as the root, convert all undirected edges to directed edges pointing away from the root, and then topologically sort the resulting directed acyclic graph.
        - Make the resulting graph directed arc-consistent by performing a backward pass from the tail to the root, enforcing arc consistency for all arcs Parent(X_i) -> X_i. This will prune the domain of the variables in the constraint graph. Throw a failure value if this is not possible.
        - Now start at the root, and perform a forward assignment using standard backtracking. But note we already enforced arc consistency on all the arcs, so irrespective of the assignment we choose for any given node, there will be at least one consistent assignment to all its children, guaranteeing a solution. Hence this step never actually has to backtrack.
        - Checking consistent requires O(D^2) operations (pairwise comparison), and no arc must be considered more than once in the absence of backtracking, for a worst-case runtime O(ED^2).
    
    - Briefly describe the operation of the most constrained-variable heuristic for the forward checking search algorithm.
        - When backtracking, choose the variable with the most constraints concerning the other unassigned variables.
    
    </details>

---

## Making Collective Decisions - Auctions 

**Game Theory**the theory of strategic decision making.
- Agent design: GT is used to determine what is the best strategy against rational players.
- Mechanism design: GT is used to define the rules of the environment.

From the above section, we know single-agent search or pairs of agents for adversarial search. In practice, complex applications can involve multi-agent systems, where multiple agents are competing for a space resource, e.g.
- farmers need a water allocation in the irrigation system
- mobile phone companies competing for ratio spectrum
- advertisers competing for ad space on high-profile media

Traditional approach: a centralised authority allocates each agent. But there is an issue that how to make sure that the scarce resource goes to those who value it the most, and that the owners of the resources maximise their financial return.

**Mechanism design for allocating scarce resources**

Mechanism design is the problem of how to design a "game" that results in maximizing a global utility function in a multi-agent system, given that each agent pursues their own (selfish) rational strategy.

In AI, mechanism design aims to construct smart systems out of simpler (possible uncooperative) systems to achieve goals that are beyond the reach of any single system. Here we focus on mechanism design for auctions.

Formally, an auction takes place between an auctioneer agent, who allocates a good or service among a set of agents called bidders.
- a language to describe the allowable strategies an agent can follow
- a protocol for communicating bids from bidders to the auctioneer
- an outcome rule, used by the auctioneer to determine the outcome

*Dimension of auction protocols*
- Winner determination: which bidder wins, and what do they pay?
    - _First-price auctions_ bidder with the highest bid is allocated, the bidder.
    - _second-price auctions_ bidder with the highest bid wins but pays the auctioneer the price of the second-highest bidder.
- Knowledge of bids: who can see the bids?
    - _Open cry_:  every bidder can see all bids from all other bidders.
    - _Sealed-bid_: bidder can see only its bids, not those of other bidders.
- Order of bids: in what order can a bid be made?
    - _One-shot_ each bidder make only on bid.
    - _Ascending_ each successive bid must exceed the previous bid.
    - _Descending_ auctioneer starts from a high price, and each bid must be lower than the previous bid.
- Number of goods: How many goods are for sale?
    - _Single good_: only one indivisible good is for sale
    - _Many goods_: many goods are available in the auction
    
    So bids can include both the price and number of goods wanted by the bidder (e.g., auctioning mobile phone spectrum), known as a combinatorial auction.

*Why can a sealed-bid auction help prevent collusion*
- Bidders cannot see each others' bids. Hence they cannot use their bids to send a price signal to other bidders.

*Factors affecting mechanism design*: A major factor is how bidding agents put a value on the good to be auctioned.
*Private value*: For each bidder, I have a utility value v_i that reflects the worth of goods to the bidder.
*Common value*: The worth of the good is the same for all bidders, but that worth is unknown a priori and must be estimated by each bidder.

**Properties of auctions**

*Desirable properties of an auction*
What kinds of properties characterise an effective mechanism design for auctions?
- Efficient: the goods go to the agent who values them the most
- Discourage collusion: the auction mechanism should discourage illegal or unfair agreements between two or more bidders to manipulate prices.
- Dominant strategy: there exists a dominant strategy for bidders, where a strategy is dominant if it gives the bidder a better pay-off than any other strategy.
- Truth-revealing: the dominant strategy results in bidders revealing their true value for the good.

*What does this have to do with AI?*
- Modern online auctions require autonomous agents who can bid on our behalf.
- These agents to model user's preferences for their bidding strategy。
- These agents need a representation language for bids.

**Types of auctions**

- ***English auction (ascending-bid)***

    Typically a *first-price, open-cry, ascending* auction.]
    
    Protocol and outcome ruel:
    - auctioneer start by asking for a minimum (reserve) price
    - auctioneer invites bids from bidders, which must be higher than the current highest price received (perhaps requiring a minimum bid increment).
    - auctioneer ends when no further bids are received, and good is sold if the final bid exceeds the reserve price.
    - the price paid by the winner is their final (highest) bid.
    
    > **Domain Strategy**: Keep bidding in small bids while the current cost is below your utility value v_i for the good.
    
    *Properties of English auction*
    - Is efficient, provided the reserve is realistic (too high -bidder who values so god may not bid; too low - seller may lose revenue).
    - Can suffer from the winner's curse: has the winner valued the good too highly because no one else made that high bid?
    - Can be susceptible to collusion:
        - bidders can agree beforehand to keep bids artificially low
        - auctioneers can plant "dummy" or bogus bidders to inflate prices
    
- ***Dutch auction (descending-bid)***

    Typically an *open-cry, descending* auction.
    
    Protocole and outcome rule:
    - auctioneer starts by asking for an extremely high initial value.
    - auctioneer repeatedly lowers the price of the good in small steps.
    - auction ends when someone makes a bid at the current offer price.
    - the price paid by the winner is the price when their bid was made.
    
    Can suffer from similar problems to the English auction.
    
    Dutch auctions were used for flower auctions in the Netherlands.
    
- ***First-price sealed-bid auction***

    Protocol and outcome rule:
    - each bidder makes a single bid
    - bid sent to auctioneer so that bidders cannot see each other's bid
    - the winner is the bidder who made the highest bid
    - the price paid by the winner is the highest bid
    
    > **Dominant strategy**: not clear - bid less than your true value v_i, but how much less depends on other agent's bids, which are unknown.
    
    Often used for tender bids for government contracts. 
    
    *Properties of first-price sealed-bid auction*
    - May not be efficient, since the agent with the highest value v_i might not win the auction.
    - Much simpler communication than English and Dutch auctions.
    - Sealed bids make it harder for the type of collusion that occurred in the German spectrum auction.

- ***Vickrey auction***

    Typically a *second-price, sealed-bid* auction.
    
    Protocol and outcome rule:
    - essentially the same as a first-price, sealed-bid auction
    - however, the price paid by the winner is the price of the second-highest bid
    
    > **Dominant strategy**: can show that it is to simply bid your value v_i for the good.
    
    Why does the winner pay the price of the second-highest bid?
    - you have nothing to lose by bidding your true value since if it is much higher than the second-highest value, you still only pay the second-highest price. This helps overcome the winner's curse.
    
    *Properties of Vickrey auction*
    - Is efficient, and truth-revealing, since the dominant strategy is to bid your value v_i.
    It- Harder for collusion to occur.
    - Can be a bit counter-intuitive for human bidders.
    - Its computation simplicity makes it popular for use in multi-agent AI systems and online auctions.

**Short Summary**
- Auctions are a mechanism to allocate resources in multi-agent environments.
- Appropriate mechanism design can achieve desirable behaviour among selfish agents.
- Types of auctions in theory practical case studies of online auctions.

<details close>
<summary>Questions</summary>

- Your class will conduct a set of in-class auctions to gain experience with each type of action. The basic process is the following:
    1. The auctioneer reveals the goods to be auctioned.
    2. You are given a card that indicates your redemption value (V ) of the good. You can think of this as your private value for the good.
    3. Think of a rational bidding strategy for each type of auction. Record the bids made.
    4. The auctioneer conducts the auction.
    5. If you win the auction, you ’pay’ the auctioneer the appropriate amount based on the type of auction. Your profit/loss is P = V −W.

    Follow the above process for (1) an English auction, (2) a Dutch auction, (3) a first-price, sealed-bid auction and (4) a second-price, sealed-bid auction. In the case of a sealed-bid auction, you should write your name and bid on a piece of paper, which is discreetly handed to the auctioneer.    
    - What strategy did you use in each auction?
    - How did the amount paid to the auctioneer vary between auctions?
    - How did the profit of the winning bidder vary between auctions?

    | Auction Type                       | Dominant Strategy                                                                                                 | Amount Paid to Auctioneer     | 
    | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------- |
    | English Auction                    | Keep bidding in small bids while the current cost is below your utility value v_i for the good                    | Highest Bids                  |
    | Dutch Auction                      | NAN                                                                                                               | Current Auctioneer Call Value |
    | First-Price, Sealed-Bid            | NAN                                                                                                               | Highest Bids                  |
    | Second-Price, Sealed-Bid (Vickrey) | can show that it is to simply bid your value v_i for the good                                                     | Second-Highest Bids           |

    - Solution
        - Straightforward using definitions of auction types. Note in the Dutch and first-price, sealed-bid auctions there is no dominant strategy. Bidder I has a private valuation v_i but their optimal action depends on the private values of other bidders, unknown to i. When the price reaches v_i or below, the bidder must decide to bid and claim the item at a higher price than necessary, or wait and risk losing the item to another bidder. Their behaviour will depend on a probability model p_i(v_i,...,v_n) over the private valuations of all n bidders. This is similar to the first-price sealed-bid auction, where agents' bids depend on the estimation of the private valuations of other bidders.  
</details>

---

## Robotics 
    
Artificial intelligence could help robots to perceive the world in a better,  more efficient way.

- Robots: Mechanical machines
- Effectors: Any parts that make robots move
- Sensors: The component that makes robots perceive the world's information
    - Range finders: sonar (land, underwater), laser range finder, radar (aircraft), tactile sensors, GPS
    - Image sensors: cameras (visual, infrared)
    - Proprioceptive sensors: shaft decoders (joints, wheels), inertial sensors, force sensors, torque sensors

Configuration of robot specified by 6 numbers => 6 degrees of freedom (DOF). 6 is the minimum number required to position the end-effector arbitrarily. For dynamical systems, add velocity for each DOF.

**Non-holonomic**: a robot that has several controls less than the number of DOFs. It means a robot cannot generally transition between two infinitesimally close configurations.

**Sources of uncertainty in interaction**<br>
There are two major sources of uncertainty for any interacting mobile agent (human or robotics):
- Everything they perceive (percepts)
- Everything they do (actions)

> Kant<br>Disctinction between 'things-in-themselves', and 'appearances'.

Problem: How do we relate these perceptions (perceived reality) with the real world. <br>
Occasionally, perceived reality breaks down:
- Hallucinations
- Optimal Illusions

**Sources of uncertainty in perception**<br>
A sensor does not always work with 100% accuracy, in the real world, there must be some perception errors that exist, for example, non-precise location, false positive/false negative rate while perceiving the environment, undetected small spatial content, finite time for perception actions.

Must make assumptions about the way the world behaves to interpret the readings at all.
- Some finite resolution sampling is sufficient to detect obstacles (consider an obstacle that consists of hundreds of long pins, sparsely distributed, pointing towards the sensor).
- Must know something about the structure of the robot to decide what an obstacle is.
- Given some sensor reading, only have a finite probability that it is correct - must have some way of dealing with.

**Confirm location**
- Localization: given map and observed landmarks, update pose distribution.
    
    This could also be achieved by using particle filtering to produce an approximate position estimate, it starts with random samples from uniform prior distribution for robot position. Then the agent update likelihood of each sample using sensor measurements and resample according to the updated likelihood. 
    
    It requires continuously updating the distribution for the current state using the latest measurement. Note, that the uncertainty of the robot's state grows as it moves until we find a landmark. Assumes that landmarks are identifiable, otherwise posterior distribution is multimodel.

- Mapping: give place and observed landmarks, update map distribution.
- Simultaneous Localization and Mapping (SLAM): given observed landmarks, update pose and map distribution.
- Probabilistic formulation of SLAM: add landmark location L1, ..., Lk to the state vector, proceed as for localization.

*Difference between Vonoroi diagrams for skeletonisation in robot path planning.*
- It can be computationally difficult to compute Vonoroi diagrams in high-dimensional configuration space.

**Bayesian Inference on Sensors**<br>
Need some way to determine whether an obstacle is there, given multiple measurements from a sensor.

> Bayesian inference is a method for determining the probability that a hypothesis is true, given a set of measurements. Probability ≈ Belief

<details close>
<summary>Example: Obstacle Dectection</summary>
<br>

    - The odds of there being an obstacle present are 1 in 10.
    - The detector has a 5% false-positive rate and a 10% false-negative rate.

    Question:
    - Probability that an obstacle is present if the detector returns positive?
    - Probability that an obstacle is present if the detector returns negative?

    Solution:
    
    1. Find the prior probability

        |                   | Obstacle | Not-obstacle |
        | ----------------- | -------- | ------------ |
        | Prior probability | 0.1      | 0.9          |
    
    2. Construct a sensor model:
    
        | Actual Class \ Prediction Class | Positive | Negative |
        | ------------------------------- | -------- | -------- |
        | Obstacle                        | TP: 0.9  | FN: 0.1  |
        | Not Obstacle                    | FP: 0.05 | TN: 0.95 |
    
    3. Calculate the probability
    
    ```
    P(obstacle|positive) = P(obstacle,positive)/P(positive)
                         = P(positive|obstacle)P(obstacle) / ( P(positive|obstacle)P(obstacle) + P(positive|not-obstacle)P(not-obstacle) )
                         = 0.9 * 0.1 / (0.9 * 0.1 + 0.05 * 0.9) = 0.667

    P(not-obstacle|negative) = P(not-obstacle,negative) / P(negative)
                             = P(negative|not-obstacle)P(not-obstacle) / ( P(negative|obstacle)P(obstacle) + P(negative|not-obstacle)P(not-obstacle) )
                             = 0.95 * 0.9 / (0.1 * 0.1 + 0.95 * 0.9) = 0.0116
    ```
</details>

**Incremental form of Bayes Law**<br>
Bayes Law can be extended to handle multiple measurements.
- Given a set of independent measurements {M_j}.
- What is the probability of hypothesis H?

If measurements are independent, can use incremental form.
- Given the current probability distribution P(H).
- And a new measurement M.
- What is the updated probability distribution P(H).

**Solution**: Use Bayes Law in incremental form: P(H) <-- M -- P(M|H) / P(M) * P(H). Sometimes called Bayesian update rule.

```
P(H|M1, M2) = P(M1, M2|H)P(H) / P(M1, M2)
            = P(M2|H)P(H|M1) / [ P(M2|H)P(H|M1) + P(M2|¬H)P(¬H|M1) ] 
```

**Motion Planning**
- Idea: plan in configuration space defined by the robot's DOFs.
    - The basic problem: ∞^d states. To convert the infinite state to finite state space.
    - Configuration space planning
        - Cell decomposition:
            - divide up space into simple cells, each of which can be traversed "easily" (e.g., convex).
        - Skeletonization:
            - Skeletonization is a process for reducing foreground regions in a binary image to a skeletal remnant that largely preserves the extent and connectivity of the original region while throwing away most of the original foreground pixels. To see how this works, imagine that the foreground regions in the input binary image are made of some uniform slow-burning material. Light fires simultaneously at all points along the boundary of this region and watch the fire move into the interior. At points where the fire travelling from two different boundaries meets itself, the fire will extinguish itself and the points at which this happens form the so-called `quench line`. This line is the skeleton. Under this definition, it is clear that thinning produces a sort of skeleton (Fisher et al., 2003).
            - identify a finite number of easily connect points/lines that form a graph such that any two points are connected by a path on the graph
- Solution is a point trajectory in free C-space.
            - Skeletonization could also achieve by
                - *Voronoi diagram*
                    - the locus of points equidistant from obstacles.
                - *Probabilistic Roadmap*
                    - A probabilistic roadmap is generated by generating random points in C-space and keeping those in free space; create a graph by joining pairs with straight lines.

**Short Summary**
- Percepts and actions are both subject to uncertainty.
- We cannot interpret our percepts without having a model of what they mean, and without (partially invalid) assumptions about how they perform.

<br>
<details close>
<summary><b>Implementation for AI</b></summary>
<br>

If you can't rely on your perceptions or your actions, does that mean that the Agent methods we have discussed are of no use?
- Many problems don't have uncertainty for perceptions and actions, e.g. scheduling, planning, game-playing, and text-based machine translation.
- Can incorporate standard agent methods within a system that handles uncertainty, i.e., re-plan if something goes wrong.
- Can apply uncertainty handlers to the whole system - e.g., Bayesian inference.

Certainly, autonomous robots and computer vision interaction with an environment create many problems that cannot be easily handled with conventional AI techniques.
</details>
