<img src="./img/st_kilda.jpg" width=100% alt='st_kilda melb vic au'>

# Data Science Path from Undergradate to Master

This repository store the notes about everything I learned from [UoM](https://www.unimelb.edu.au/) [Data Science Major](https://study.unimelb.edu.au/find/courses/major/data-science/) and other online sources from 2019 to 2024. 
For me, this is more like a revision notes / cheatsheet for my future data science releated career. 

If you notice something not so clear or errors or contents that confusing to other audiences, just start an issue and let's see what we can improve on it.
Also, if there is a better way for categorisation, don't be hesitate starting a new issue! Let's make DS life more easier!

One thing I like to mention is although I may paraphrase some concept in my own words, sometimes I will directly copy past the concept from others' work. 
You can always find the original page just follow the reference page. Maybe the original content could give you a better understanding.

---
<details close>
<summary><h3>Table of Contents</h3></summary>
<br>
    
- [Notebooks](#notebooks)
    - [Mathematical Concepts](#mathematical-concepts)
        - [Probability](#probability)
        - [Statistics](#statistics)
        - [Discrete Mathes and Operations Research: Linear Programming](#discrete-mathes-and-operations-research-linear-programming)
        - [Techniques in Operations Research](#techniques-in-operational-research)
        - [Voting methods](#voting-methods)
    - [Machine Learning](#machine-learning)
        - [Evaluation Metrics](#evaluation-metrics)
        - [Dimensionality Reduction Methods](#dimensionality-reduction-methods)
        - [Wrapper Methods](#wrapper-methods)
        - [Embeded Methods](#embeded-methods)
        - [Ensemble Learning](#ensemble-learning)
        - [Data Processing Concepts](#data-processing-concepts)
        - [Supervised Learning Methods](#supervised-learning-methods)
        - [Unsupervised Learning Methods](#unsupervised-learning-methods)
    - [Neural Network](#neural-network)
        - [Activation Functions](#activation-functions)
    - [Artificial Intelligence](#artificial-intelligence)
        - [Intelligent Agent](#intelligent-agent)
        - [Problem-solving](#problem-solving)
        - [Adversarial Search](#adversarial-search)
        - [Knowledge Represent](#knowledge-represent)
        - [Uncertain Knowledge R](#uncertain-knowledge-r)
    - [Natural Language Processing](#natural-language-processing)
    - [Frontend Development](#frontend-development)
    - [Backend Development](#backend-development)
    - [Data Structure](#data-structure)
    - [Cluster and Cloud Computing](#cluster-and-cloud-computing)
    - [Database Management System](#database-management-system)
    - [Block Chain](#block-chain)
    - [Online Data Sources](#online-data-sources)
    - [Others](#others)
        - [Concepts](#concepts)
        - [Programming skills](#programming-skills)
        - [Database skills](#database-skills)
        - [Data Science Lifecycle](#data-science-lifecycle)
- [Reference](#reference)
</details>

---

### Mathematical Concepts

[[Elaine & Johathan (2021)](https://www.livescience.com/38936-mathematics.html)] Mathematics is the science that deals with the logic of shape, quantity and arrangement. Math is all around us, in everything we do. It is the building block for everything in our daily lives, including mobile devices, computers, software, architecture (ancient and modern), art, money, engineering and even sports.

#### Math Miscs
- [x] [Minkowski distance](./notebooks/MISC/Minkowski.ipynb)
    
    The Minkowski distance or Minkowski metric is a metric in a normed vector space which can be considered as a generalization of both the Euclidean distance and the Manhattan distance. It is named after the German mathematician Hermann Minkowski.
    
    Check python implementation [here](./src/distance.py).

- [ ] Discrete Variable
- [ ] Continuous Variable
- [ ] [Standard Deviation and Variance]()
- [ ] [Eigenvalue & Eigenvectors]()
- [ ] [Variability of Sums (Covariance)]()
- [ ] [Correlation]()
- [ ] [Conditional Variance]()
- [ ] [Analysis of variance (ANOVA)]()
- [ ] [Likelihood]()
- [ ] [Interquartile Range (IQR)]()
- [ ] [Anomaly detection]()
- [ ] [Exponential Family]()
- [ ] [Expectation Value]()
- [ ] [Monotonic Function]()
- [ ] [Law of Large Number]()
- [ ] [Central Limit Theorem (CLT)]()
- [ ] [Staticstical Hypothesis Testing]()
- [ ] [Statistical Significance]()
- [ ] [Mean Squared Error (MAE)]()
- [ ] [R2-score]()
- [ ] [Mean Absolute Error (MSE)]()
- [ ] [Root Mean Square Error (RMSE)]()
- [ ] [Root Mean Square Log Error (RMSLE)]()
- [ ] [Chi-square Test]()
- [ ] [Fisher Score]()
- [ ] [Mean Absolute Difference]()
- [ ] [Variance Threshold]()
- [ ] [Dispersion Ratio]()
- [ ] [F Test]()
- [ ] [Kruskal-Wallis H test]()
- [ ] [Term Frequency-Inverse Document Frequency (TD_IDF)]()
- [ ] [Underfit & Overfit]()
- [ ] [Cost Function]()

#### Probability

[[Wikipedia](https://en.wikipedia.org/wiki/Probability_theory)] Probability theory is the branch of mathematics concerned with probability. Although there are several different probability interpretations, probability theory treats the concept in a rigorous mathematical manner by expressing it through a set of axioms. Typically these axioms formalise probability in terms of a probability space, which assigns a measure taking values between 0 and 1, termed the probability measure, to a set of outcomes called the sample space. Any specified subset of the sample space is called an event. Central subjects in probability theory include discrete and continuous random variables, probability distributions, and stochastic processes (which provide mathematical abstractions of non-deterministic or uncertain processes or measured quantities that may either be single occurrences or evolve over time in a random fashion). Although it is not possible to perfectly predict random events, much can be said about their behavior. Two major results in probability theory describing such behaviour are the law of large numbers and the central limit theorem.

- [ ] [Axioms]()
- [ ] [Expectation value]()
- [ ] [Probability Density Function (PDF)]()
- [ ] [Culmulative Distribution Function (CDF)]()
- [ ] [Probability Mass Function (PMF)]()
- [ ] [Probability Generating Function (PGF)]()
- [ ] [Moment Generating Function (MGF)]()
- [ ] [Cumulant Generating Function (CGF)]()
- [ ] [Recover Distribution from MGF]()
- [ ] [Skewed Probability Distribution]()
- [ ] [Balanced Probability Distribution]()
- [ ] [Bernoulli Distribution]()
- [ ] [Geometric Distribution]()
- [ ] [Negative Binomial Distribution]()
- [ ] [Hypergeometric Distribution]()
- [ ] [Guassian Distribution]()
- [ ] [Poisson Distribution]()
- [ ] [Uniform Distribution]()
- [ ] [Gamma Random Variables]()
- [ ] [Beta Distribution]()
- [ ] [Pareto Distribution]()
- [ ] [Lognormal Distribution]()
- [ ] [Bivariate Normal Distribution]()
- [ ] [Indenpendence of Random Variables]()
- [ ] [Stochastic Processes]()
- [ ] [Discrete-time Markov Chains]()
- [ ] [Non-Markov Stochastic Process]()
- [ ] [Prior and Posterior]()
- [ ] [Bayesian Inference]()
- [ ] [Probability Cheat Sheet]()

#### Statistics

[[Wikipedia](https://en.wikipedia.org/wiki/Statistics)] Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied. Populations can be diverse groups of people or objects such as "all people living in a country" or "every atom composing a crystal". Statistics deals with every aspect of data, including the planning of data collection in terms of the design of surveys and experiments.

Statistics is a mathematical body of science that pertains to the collection, analysis, interpretation or explanation, and presentation of data, or as a branch of mathematics. Some consider statistics to be a distinct mathematical science rather than a branch of mathematics. While many scientific investigations make use of data, statistics is concerned with the use of data in the context of uncertainty and decision making in the face of uncertainty.

- [ ] [Types of data]()
- [ ] [Descriptive statistics]()
- [ ] [Inferential statistics]()
- [ ] [Exploratory data analysis]()
- [ ] [Linear Model]()
- [ ] [Non Linear Model]()
- [ ] [Devariance]()
- [ ] [Gradient Descent]()
- [ ] [Orthonoality, Eigenthings, Rank, Idempotence, Trace & Quadratic forms]()
- [ ] [Full rank model]()
- [ ] [Less than full rank model]()
- [ ] [Inference for full rank model]()
- [ ] [Inference for less than full rank model]()
- [ ] [Interaction]()
- [ ] [Dispersion]()
- [ ] [Overdispersion]()
- [ ] [Quasi-likelihood]()
- [ ] [Binary Regression]()
- [ ] [L1 & L2 Regularization Methods]()
- [ ] [Expectation Maximization]()
- [ ] [Miture Model]()
- [ ] [Gibb Sampling]()
- [ ] [Markov Chain Monte Carlo (MCMC) Sampling]()
- [ ] [Miture Model - Expectation Maximisation]()
- [ ] [Variational Inference]()
- [ ] [Experimental Design]()

#### Discrete Mathes and Operations Research: Linear Programming
- [ ] Introduction to Linear Programming
- [ ] Geometry of Linear Programming
- [ ] Geometry of LP in higher Dimensions
- [ ] Basic feasible Solutions
- [ ] Fundamental theorem of Linear Programming
- [ ] Simplex method
- [ ] Solution Possibilities
- [ ] Non-standard Formulations
- [ ] Duality Theory
- [ ] Sensitivity Analysis
  
#### Techniques of Operational Research

[[Wikipedia](https://en.wikipedia.org/wiki/Operations_research)] Operational research (OR) encompasses the development and the use of a wide range of problem-solving techniques and methods applied in the pursuit of improved decision-making and efficiency, such as simulation, mathematical optimization, queueing theory and other stochastic-process models, Markov decision processes, econometric methods, data envelopment analysis, neural networks, expert systems, decision analysis, and the analytic hierarchy process.[5] Nearly all of these techniques involve the construction of mathematical models that attempt to describe the system. Because of the computational and statistical nature of most of these fields, OR also has strong ties to computer science and analytics. Operational researchers faced with a new problem must determine which of these techniques are most appropriate given the nature of the system, the goals for improvement, and constraints on time and computing power, or develop a new technique specific to the problem at hand (and, afterwards, to that type of problem).

- [ ] [Golden section search]()
- [ ] [Fibonacci search]()
- [ ] [Newton's method]()
- [ ] [Armijo-Goldstein condition]()
- [ ] [Wolff condition]()
- [ ] [Optimality Conditions]()
- [ ] [Steepest Descent Method]()
- [ ] [Rate of convergence]()
- [ ] [Newton's method - Quasi Netwon method]()
- [ ] [Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS)]()
- [ ] [Lagrange multipliers and sensitivity analysis]()
- [ ] [KKT conditions]()
- [ ] [Constrains qualifications]()
- [ ] [Sufficient optimality conditions]()
- [ ] [Penalty methods for Constrained Optimisation]()
- [ ] [Log Barrier method]()
- [ ] [Exact penalty method]()
- [ ] [Comparison of penalty methods]()

#### Voting Methods

[[Wikipedia](https://en.wikipedia.org/wiki/Voting#Voting_methods)] Voting is a method for a group, such as a meeting or an electorate, in order to make a collective decision or express an opinion usually following discussions, debates or election campaigns. Democracies elect holders of high office by voting. Residents of a place represented by an elected official are called "constituents", and those constituents who cast a ballot for their chosen candidate are called "voters". There are different systems for collecting votes, but while many of the systems used in decision-making can also be used as electoral systems, any which cater for proportional representation can only be used in elections.

- [ ] Voting Criteria
    - Mutual Majority
    - Monotonicity
    - Participation
    - Independence of irrelevant alternatives
    - Later no harm
- [ ] Plurality voting
- [ ] Majority judgement
- [ ] Instant runoff voting
- [ ] Multiple Rounds voting
- [ ] Condorcet Voting
- [ ] Borda count
- [ ] Score voting

### Machine Learning

[[Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)] Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.[1] It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

#### Dimensionality Reduction Methods
[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension. Working in high-dimensional spaces can be undesirable for many reasons; raw data are often sparse as a consequence of the curse of dimensionality, and analyzing the data is usually computationally intractable (hard to control or deal with). Dimensionality reduction is common in fields that deal with large numbers of observations and/or large numbers of variables, such as signal processing, speech recognition, neuroinformatics, and bioinformatics.

##### Feature selection

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Feature selection approaches try to find a subset of the input variables (also called features or attributes). The three strategies are: the filter strategy (e.g. information gain), the wrapper strategy (e.g. search guided by accuracy), and the embedded strategy (selected features are added or removed while building the model based on prediction errors).

Data analysis such as regression or classification can be done in the reduced space more accurately than in the original space.

##### Feature projection

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] Feature projection (also called feature extraction) transforms the data from the high-dimensional space to a space of fewer dimensions. The data transformation may be linear, as in principal component analysis (PCA), but many nonlinear dimensionality reduction techniques also exist.[4][5] For multidimensional data, tensor representation can be used in dimensionality reduction through multilinear subspace learning.

- [ ] [Principal Component Analysis (PCA)](./dr/notebooks/PCA.ipynb)
- [ ] [t-distributed stochastic neighbor embedding (t-SNE)]()
- [ ] [Kernel PCA]()
- [ ] [Graph-based kernel PCA]()
- [ ] [Linear Discriminant Analysis (LDA)]()
- [ ] [Generalized Discriminant Analysis (GDA)]()
- [ ] [Autoencoder]()
- [ ] [Missing Values Ratio]()
- [ ] [Low Variance Filter]()
- [ ] [High Correlation Filter]()
- [ ] [Non-negative matrix factorization (NMF)]()
- [ ] [Uniform Manifold Approximation and Projection (UMAP)]()

##### Dimension reduction

[[Wikipedia](https://en.wikipedia.org/wiki/Dimensionality_reduction)] For high-dimensional datasets (i.e. with number of dimensions more than 10), dimension reduction is usually performed prior to applying a K-nearest neighbors algorithm (k-NN) in order to avoid the effects of the curse of dimensionality.[20]

Feature extraction and dimension reduction can be combined in one step using principal component analysis (PCA), linear discriminant analysis (LDA), canonical correlation analysis (CCA), or non-negative matrix factorization (NMF) techniques as a pre-processing step followed by clustering by K-NN on feature vectors in reduced-dimension space. In machine learning this process is also called low-dimensional embedding.[21]

For very-high-dimensional datasets (e.g. when performing similarity search on live video streams, DNA data or high-dimensional time series) running a fast approximate K-NN search using locality-sensitive hashing, random projection, "sketches", or other high-dimensional similarity search techniques from the VLDB conference toolbox might be the only feasible option.

#### Evaluation Metrics
- [ ] [TP, FP, TN, FN]()
- [ ] [Confusion Matrix]()
- [ ] [Precision, Recall, F1-score]()
- [ ] [Area Under the Curve - Receiver Operating Characteristics (AUC-ROC)]()
- [ ] [Log loss]()
- [ ] [Entropy]()
- [ ] [Mutual Information]()
- [ ] [Information Gain]()
- [ ] [Joint Mutual Information]()

#### Wrapper Methods
- [ ] [Step-wise Forward Feature Selection]()
- [ ] [Backward Feature Elimination]()
- [ ] [Exhaustive Feature Selection]()
- [ ] [Recursive Feature Elimination]()
- [ ] [Boruta]()

#### Embeded Methods
- [ ] [Lasso Regularization (L1)]()
- [ ] [Ridge Regularization (L2)]()
- [ ] [Random Forest Importance]()

#### Ensemble Learning
- [ ] [Bagging (Bootstrap Aggregating)]()
- [ ] [Boosting]()

#### Data Processing Concepts
- [ ] [One Hot Encoding]()
- [ ] [Dummy Encoding]()
- [ ] [Normalisation]()
- [ ] [Standardisation]()
- [ ] [Discretisation]()

#### Supervised Learning Methods
- [ ] [K Nearest Neighbors](./notebooks/SL/KNN.ipynb)
- [ ] [Regression](./notebooks/SL/Regression.ipynb)
    - [ ] [Linear Regression](./notebooks/SL/LinearRegression.ipynb)
    - [ ] [Simple Linear Regression]()
    - [ ] [Multiple Linear Regression]()  
    - [ ] [Logistic Regression](./notebooks/SL/LogisticRegression.ipynb)
    - [ ] [Backward Elimination]()
    - [ ] [Polynomial Regression]()
- [ ] [Perceptron](./notebooks/SL/Perceptron.ipynb)
- [ ] [Navie Bayes](./notebooks/SL/NaiveBayes.ipynb)
- [ ] [Support Vector Machine (SVM)](./notebooks/SL/SVM.ipynb)
- [ ] [Decision Tree]()
- [ ] [Random Forest]()
- [ ] [AdaBoost]()
- [ ] [XGBoost]()
- [ ] [Light GBM]()
- [ ] [Recommending System]()

#### Unsupervised Learning Methods
- [ ] [k-means clustering (KMean)](./notebooks/KMean.ipynb)
- [ ] [Hierarchical Clustering]()
- [ ] [Anomaly detection]()
- [ ] [Indenpendent Component Analysis (IDA)]()
- [ ] [Apriori algorithm]()
- [ ] [Singular value decomposition]()

### Neural Network
- [ ] [Neuron]()
- [ ] [Layers]()
- [ ] [Epoch]()
- [ ] [Neural Network]()
- [ ] [Convolutional Neural Network]()
- [ ] [Genetic Algorithm]()

#### Activation Functions
- [ ] [Linear Activation]()
- [ ] [Heaviside Activation]()
- [ ] [Logistic Activation]()
- [ ] [Sigmoid]()
- [ ] [Rectified Linear Unit (ReLU)]()
- [ ] [tanh]()
- [ ] [Softmax]()
- [ ] [Auto Encoder]()
- [ ] [Genetric Algorithm]()
- [ ] [Ensembler]()

### Artificial Intelligence

[[Wikipedia](https://en.wikipedia.org/wiki/Artificial_intelligence)] Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

>  "It is a branch of computer science by which we can create intelligent machines which can behave like a human, think like humans, and able to make decisions." (Java T Point) 

**Goals of Artifical Intelligence**
- Replicate human intelligence
- Solve knowledge-intensive tasks
- An intelligent connection of perception and action
- Building a machine which can perform tasks that requires human intelligence such as:
    - Proving a math theorm
    - Play chess game
    - Plan some surgical operation
    - Self-driving a car
- Creating some system can exhibit intelligent behavior, learn new things by itself, demonstrate, explain, and can advise to its user.

| **Advantages of Artificial Intelligence** | **Disadvantages of Artificial Intelligence** |
| :---------------------------------------- | :------------------------------------------- |
| High accuracy with less error             | High cost                                    |
| High-speed                                | Can't think out of the box                   |
| High reliability                          | No feelings and emotions                     |
| Useful for risky areas                    | Increase dependency on machines              |
| Digital Assistant                         | No original creativity                       |
| Useful as a public utility                |                                              |

**Artificial Intelligence Types**

| **Based on Capabilities**                          | **Based on functionality**                               |
| -------------------------------------------------- | -------------------------------------------------------- |
| [Weak AI / Narrow AI](./doc/AI/types/narrow_ai.md) | [Reactive Machines](./doc/AI/types/reactive_machines.md) | 
| [General AI](./doc/AI/types/general_ai.md)         | [Limited Memory](./doc/AI/types/limited_memory.md)       |
| [Super AI](./doc/AI/types/super_ai.md)             | [Theory of Mind](./doc/AI/types/theory_of_mind.md)       |
|                                                    | [Self-Awareness](./doc/AI/types/self_awareness.md)       |

#### Intelligent Agent
- [x] [Types of Agent](./doc/AI/agent/type.md)
    
    Agents can be grouped into five classes based on their degree of perceived intelligence and capability. All these agents can improve their performance and generate better action over the time. These are given below:
    - Simple reflex agent
    - Model-based reflex agent
    - Goal-based agents
    - Utility-based agent
    - Learning Agent
    
- [x] [Intelligent Agent](./doc/AI/agent/intelligent_agent.md)

    An agent can be anything that perceiveits environment through sensors and act upon that environment through actuators. An Agent runs in the cycle of perceiving, thinking, and acting.
    
    An AI system can be defined as the study of the rational agent and its environment. The agents sense the environment through sensors and act on their environment through actuators. An AI agent can have mental properties such as knowledge, belief, intention, etc.
    
- [x] [Agent Environment](./doc/AI/agent/agent_environment.md)

    An environment is everything in the world which surrounds the agent, but it is not a part of an agent itself. An environment can be described as a situation in which an agent is present.

    The environment is where agent lives, operate and provide the agent with something to sense and act upon it. An environment is mostly said to be non-feministic.

- [x] [Turing Test in AI](./doc/AI/agent/turing_test.md)

    Turing Test was introduced by Alan Turing in his 1950 paper, "Computing Machinery and Intelligence", which considered the question, "Can Machine think?"

#### Problem-solving
- [x] [Search Algorithms](./doc/AI/solving/search_algo.md)

    Search algorithms are one of the most important areas of Artificial Intelligence. This topic will explain all about the search algorithms in AI.
        
- [x] [Uniform Search Algorithm](./doc/AI/solving/uninformed.md)

    Uninformed search is a class of general-purpose search algorithms which operats in brute-force way. Uninformed search algorithms do not have additional information about state or search space other than how to traverse the tree, so it is also called blind search.

- [x] Heuristic Function

    Heuristic is a function which is used in Informed Search, and it finds the most promising path. It takes the current state of the agent as its input and produces the estimation of how close agent is from the goal. The heuristic method, however, might not always give the best solution, but it guaranteed to find a good solution in reasonable time. Heuristic function estimates how close a state is to the goal. It is represented by h(n), and it calculates the cost of an optimal path between the pair of states. The value of the heuristic function is always positive.
    
    **Admissbility of the heuristic function is given as: h(n) <= h*(n).

    Here h(n) is heurstic cost, and h*(n) is the estimated cost. Hence heuristic cost should be less than or equal to the estimated cost.

- [x] [Informed Search Algorithm](./doc/AI/solving/informed.md)
- [ ] [Hill Climbing Algorithm]()
- [ ] [Greedy Search]()
- [ ] [A* Search]()
- [ ] [Means-Ends Analysis]()

- Summary Properties

    | Uninformed Search Algorithm            | Time Complexity | Space Complexity | Completeness                                     | Optimality Comparisons                                                   |
    | -------------------------------------- | --------------- | ---------------- | ------------------------------------------------ | ------------------------------------------------------------------------ |
    | Breadth-first search                   | O(b^d)          | O(b^d)           | Complete                                         | Optimal                                                                  |
    | Depth-first search                     | O(n^m)          | O(bm)            | Complete                                         | Non-optimal                                                              |
    | Depth-limited search                   | O(b^ℓ)          | O(bℓ)            | Complete if solution is above ℓ                  | Non-optimal                                                              |
    | Uniform cost search                    | O(b^{1+[C*/ε]}) | O(b^{1+[C*/ε]})  | Complete if there is a solution                  | Optimal                                                                  |
    | Iterative deepening depth-first search | O(b^d)          | O(bd)            | Complete if branching factor is finite           | Optimal if path cost is a non-decreasing function of depth of the node   |
    | Bidirectional search                   | O(b^d)          | O(b^d)           | Complete if both search use BFS                  | Optimal                                                                  |
    | Best-First search                      | O(b^m)          | O(b^m)           | Incomplete                                       | Non-optimal                                                              |
    | A* search                              | O(b^d)          | O(b^d)           | Complete if finite branching factor & fixed cost | Optimal if heuristic functions is admissible and consistency             |
    
    - d = depth of shallowest solution
    - b = branching factor, a node at every state
    - m = maximum depth of any node
    - ℓ = depth limit parameter

#### [Adversarial Search](#)
- [ ] [Minimax Search]()
- [ ] [Alpha-beta Pruning]()
- [ ] [Monte Carlo Search Tree]()

#### Knowledge Represent

#### Uncertain Knowledge R

### Natural Language Processing
- [ ] [Word2vec]()
- [ ] [Bag of words (BoW)]()

### [Frontend Development](#)

### [Backend Development](#)

### [Cluster and Cloud Computing](#)

### Data Structure
Data structure is a specialised format for organizing, processing retrieving and storing data. It makes human and machine have a better understanding of data storage. Specifically, it could be use for storing data, managing resources and services, data exchange, ordering and sorting, indexing, searching, scalability in a more efficient way (David & Sarah, 2021).

- [x] [Linked List](./notebooks/DS/LinkedList.ipynb)

    Linked list is a linear data structure that includes a series of connected nodes. Linked list can be defined as the nodes that are randomly stored in the memory. A node in the linked list contains two parts, i.e., first is the data part and second is the address part. The last node of the list contains a pointer to the null. After array, linked list is the second most used data structure. In a linked list, every link contains a connection to another link (Java T Point).
    
    Linked lists are among the simplest and most common data structures. They can be used to implement several other common abstract data types, including lists, stacks, queues, associative arrays, and S-expressions, though it is not uncommon to implement those data structures directly without using a linked list as the basis (Wikipedia).
    
    Linked lists overcome the limitations of array, a linked list doesn't require a fix size, this is because the memory space a linked list is dynamic allocated. Unlike an array, linked list doesn't need extra time if we want to increase the list size. Also linked list could store various type of data instead of fixed one.
    
    **Linked list could classified into the following types:**
    - Singly-linked list
        A linked list only link in one direction, a new node also insert at the end of the linked list.
    - Doubly-linked list
        A node in linked list have two direction pointer, the headers point to the previous node and the next node. 
    - Circular singly linked list
        The last node of the list always point to the first node of the linked list.
    - Circular doubly linked list
        The last node of the list point to the first node, and each node also have a pointer link to the previous node.
    
    | **Advantages**                  | **Disadvantages**                                                                         |
    | :------------------------------ | :---------------------------------------------------------------------------------------- |
    | Dynamic size data structure     | More memory usage compare to an array                                                     |
    | Easy insertion and deletion     | Traversal is not easy because it cannot be randomly accessed                              |
    | Memory consumption is efficient | Reverse traversal is hard, a double-linked list need extra space to store another pointer |
    | Easy implement                  |                                                                                           |

   
    **Time Complexity**

    | Operation | Average case time complexity | Worst case time complexity | Description                                                             |
    | :-------- | :--------------------------: | :------------------------: | :---------------------------------------------------------------------- |
    | Insertion | O(1)                         | O(1)                       | Insert to the end of the linked list                                    |
    | Deletion  | O(1)                         | O(1)                       | Delect only need one operation                                          |
    | Search    | O(n)                         | O(n)                       | Linear search time because it requires search from the start to the end |
   
    n is the number of nodes in the given tree.
   
    **Space Complexity**
    
    | Operation | Space Complexity | 
    | :-------- | :--------------: | 
    | Insertion | O(n)             | 
    | Deletion  | O(n)             |
    | Search    | O(n)             |
  
- [ ] [Stack]()
- [ ] [Queue]()
- [ ] [Tree]()
- [ ] [Graph]()
- [ ] [Sorting Algorithms]()
- [ ] [Searching Algorithms]()

### Database Management System ([ADDYNAMICS](https://www.appdynamics.com/topics/database-management-systems))

Database Management Systems (DBMS) are software systems used to store, retrieve, and run queries on data. A DBMS serves as an interface between an end-user and a database, allowing users to create, read, update, and delete data in the database.

DBMS manage the data, the database engine, and the database schema, allowing for data to be manipulated or extracted by users and other programs. This helps provide data security, data integrity, concurrency, and uniform data administration procedures.

DBMS optimizes the organization of data by following a database schema design technique called normalization, which splits a large table into smaller tables when any of its attributes have redundancy in values. DBMS offer many benefits over traditional file systems, including flexibility and a more complex backup system.

Database management systems can be classified based on a variety of criteria such as the data model, the database distribution, or user numbers. The most widely used types of DBMS software are relational, distributed, hierarchical, object-oriented, and network.

**Distributed database management system**: A distributed DBMS is a set of logically interrelated databases distributed over a network that is managed by a centralized database application. This type of DBMS synchronizes data periodically and ensures that any change to data is universally updated in the database.

**Hierarchical database management system**: Hierarchical databases organize model data in a tree-like structure. Data storage is either a top-down or bottom-up format and is represented using a parent-child relationship.

**Network database management system**: The network database model addresses the need for more complex relationships by allowing each child to have multiple parents. Entities are organized in a graph that can be accessed through several paths.

**Relational database management system**: Relational database management systems (RDBMS) are the most popular data model because of its user-friendly interface. It is based on normalizing data in the rows and columns of the tables. This is a viable option when you need a data storage system that is scalable, flexible, and able to manage lots of information.

**Object-oriented database management system**: Object-oriented models store data in objects instead of rows and columns. It is based on object-oriented programming (OOP) that allows objects to have members such as fields, properties, and methods.

- [ ] [Data Modeling]()
- [ ] [Relational Data Model]()
- [ ] [Normalisation]()
- [ ] [Transaction Processing]()
- [ ] [Concurrency Control]()
- [ ] [File Organization]()
- [ ] [Hasing]()
- [ ] [Raid]()

### Block Chain [Java T Point](https://www.javatpoint.com/blockchain-tutorial)

A blockchain is a constantly growing ledger which keeps a permanent record of all the transactions that have taken place in a secure, chronological, and immutable way.

- Ledger: It is a file is constantly growing.
- Permanent: It means once the transaction goes inside a blockchain, you can put up it permanetly in the ledger.
- Secure: Blockchain placed information in a secure way. It uses very advanced cryptography to make sure that the information is locked inside the blockchain.
- Chronological: Chronological means every transaction happens after the previous one.
- Immutable: It means as you build all the transaction onto the blockchain, this ledger can never be changed.

A blockchian is a chain of blocks which contain information. Each block records all of the recent transactions, and once completed goes into the blockchain as a permanent dataset. Each time a block gets completed, a new block is generated.

> A blockchain can be used for the secure transfer of money, property, contracts, etc. without requiring a third-party intermediary like bank or government. Blockchain is a software protocol, but it could not be run without the Internet (like SMTP used in email).

Blockchain technology has become popular because of the following.
- Time reduction: In the financial industry, blockchain can allow the quicker settlement of trades. It does not take a lengthy process for verification, settlement, and clearance. It is because of a single version of agreed-upon data available between all stakeholders.
- Unchangeable transactions: Blockchain register transactions in a chronological order which certifies the unalterability of all operations, means when a new block is added to the chain of ledgers, it cannot be removed or modified.
- Reliability: Blockchain certifies and verifies the identities of each interested parties. This removes double records, reducing rates and accelerates transactions.
- Security: Blockchain uses very advanced cryptography to make sure that the information is locked inside the blockchain. It uses Distributed Ledger Technology where each party holds a copy of the original chain, so the system remains operative, even the large number of other nodes fall.
- Collaboration: It allows each party to transact directly with each other without requiring a third-party intermediary.
- Decentralized: It is decentralized because there is no central authority supervising anything. There are standards rules on how every node exchanges the blockchain information. This method ensures that all transactions are validated, and all valid transactions are added one by one.

| **Block Chain Tutorial Index**          |                                          |
| --------------------------------------- | ---------------------------------------- |
| [Blockchain Tutorial]()                 | [Blockchain Key Areas]()                 |
| [History of Blockchain]()               | [Blockchain Cryptocurrency]()            |
| [What is Bitcoin]()                     | [Blockchain DAO]()                       |
| [Blockchain Version]()                  | [Blockchain limitation]()                |
| [Role of Bitcoin Miners]()              | [Blockchain Double Spending]()           |
| [Blockchain Block Hashing]()            | [Blockchain Bitcoin Cash]()              |
| [How Block Hashes Work in Blockchain]() | [Bitcoin Forks and SegWit]()             |
| [Basic Components of Bitcoint]()        | [Blockchian Merkle Tree]()               |
| [Blockchain Proof of work]()            | [Blockchain vs. Database]()              |
| [Coinbase Transaction]()                | [Who sets the Bitcoin Price]()           |
| [Key Concepts in Bitcoin]()             | [Getting started with Bitcoin]()         |
|                                         | [How to choose Bitcoin Wallet]()         | 
|                                         | [Sending and Receiving Bitcoin]()        |
|                                         | [Converting Bitcoins to Fiat Currency]() |

### Online Data Sources
- [Data.gov](https://catalog.data.gov/dataset): One of the most comprehensive data souces in the U.S. It could be helpful for web application design or design data visualisation.
- [U.S. Census Bureau](https://www.census.gov/): Demographic information from federal, state, and local governments, and commercial entities in the U.S.
- [Open Data Network](https://www.opendatanetwork.com/): Powerful data search engine cover the fields like finance, public safety, infrastructure, and housing and development. 
- [Google Cloud PUblic Datasets](https://cloud.google.com/datasets): There are a selection of public datasets available through the Google Cloud Public Dataset Program that could access via Google BigQuery.
- [Dataset Search](https://datasetsearch.research.google.com/): The Dataset search is a search engine designed specifically for data sets.
- [国家统计局](https://data.stats.gov.cn/): National Bureau of Statistics of China contains demographic information of provience, city, etc.
- [Australian Bureau of Statistics](https://www.abs.gov.au/census): Australian Census Data.

### Others
#### Concepts
- [ ] [k-anonymity]()
- [ ] [l-diversity]()
- [x] [FAIR principle](https://www.go-fair.org/fair-principles/)
    
    FAIR Guiding Principles of scientific data management and stewardship provide guidelines to improve the **Findability, Accessibility, Interoperability, Reuse** of digital assest. The principle emphasis machine actionabiilty (i.e. the capacity of computational systems to find, access, interoperate, and reuse data with none or minimal human intervention) because humans increasingly rely on computational support to dael with data as a result of the increase in volumne, complexity, and creation speed of data (GO FAIR).
    
    According to Go-Fair, it define the following **FAIRification process**.
    
    Findable:
    - (Meta)data are assigned a globally unique and persistent identifier
    - Data are described with rich metadata
    - Metadata clearly and explicityly include the identifier of the data they describe
    - (Meta)data are registered or indexed in a searchable resource
    
    Accessible:
    - (Meta)data are retrievable by their identifier using a standardised communications protocol
        - The protocol is open, free, and universally implementable
        - The protocol allows for an authentication and authorisation procedure, where necessary
    - Metadata are accessible, even when the data are no longer available
    
    Interoperable:
    - (Meta)data use a formal, accessible, shared, and broadly applicable language for knowledge representation
    - (Meta)data use vocabularies that follow FAIR principles
    - (Meta)data include qualified references to other (meta)data
    
    Reusable:
    - (Meta)data are richly described with a plurality of accurate and relevant attributes
        - (Meta)data are released with a clear and accessible data usage license
        - (Meta)data are associated with detailed provenance
        - (Meta)data meet domain-relevant community standards
    
    In short, follow the FAIR principle, a dataset will contain a meaningful metadata that describes where to find the data, how to access the data, how to use the data and finally help user optimise the reuse of data.
    
- [x] 5W1H

    5W1H stands for What? Who? Where? When? Why? How? This method consists of asking a systematic set of questions to collect all the data necessary to draw up a report of the existing situation with the aim to identifying the true nature of the problem and describing the context precisely (Humanperf Software, 2018).
    
    By asking the right questions, make the situation easy understand and problem-solving process rational and efficient (David, 2019).
    - What: description of the problem;
    - Who: the responsible parties;
    - Where: the location of the problem;
    - When: temporal characteristics of the problem (at what point in time, how often)
    - How: the effects of the problem?
    - Why: reasons, cause of the problems?

    In the real world, a situation may not have so many questions could be asked. However, it's a good practice to keep your mind focus on the situation or problem itself.
    
- [x] CI/CD
    > Isaac Sacolick: CI/CD is a best practice for devops and agile development. Here's how software development teams automate continuous integration and delivery all the way through the CI/CD pipeline.
    
    Continuous Integration (CI) is a coding philosophy and set of practices that derive development teams to frequently implement small code changes and check them into a version control repository. So the team could continuous intergrate and validate changes. Continuous integrations establishes an automated way to build, package and test their applications. This encourage the programmer commit more frequently which leads to better collaboartion and code code quality.
    
    Continouse Delivery (CD) picks up where continous intergration ends, and automates application delivery to selected environments, including production, development, and testing environments. Continuous delivery is an automated way to push code changes to these environments which allows the developer could continuous update small changes when CI is valid.
    
- [x] Scalability, Horizontal & Vertical Scaling

    Scalability is the property of a system to handle a growing amount of work by adding resources to the system (Wikipedia). It is a measure of a system's ability to increase or decrease in performance and cost in resopnse to changes in application and system processing demands.
    
    Scalability can be measured over multiple dimensions, such as:
    - Administrative scalability: The ability for an increasing number of organizations or users to access a system.
    - Functional scalability: The ability to enhance the system by adding new functionality without disrupting existing activities.
    - Geographic scalability: The ability to maintain effectiveness during expansion from a local area to a larger region.
    - Load scalability: The ability for a distributed system to expand and contract to accommodate heavier or lighter loads, including, the ease with which a system or component can be modified, added, or removed, to accommodate changing loads.
    - Generation scalability: The ability of a system to scale by adopting new generations of components.
    - Heterogeneous scalability is the ability to adopt components from different vendors.

    Most of time, people may talk scale a system horizontally or vertially:
    
    - Horizontal Scaling
        
        Horizontal scaling refers to adding addtional nodes or machines to respond new demands (CloudZero, 2021). For example, if a web server has a increase demand on network traffic, by horizontal scaling, we add more server to increase the access nodes for future users.
        
        | Advantages                                    | Disadvantages                                     |
        | --------------------------------------------- | ------------------------------------------------- |
        | Scaling is easier from a hardware perspective | Increased complexity of maintenance and operation |
        | Fewer periods of downtime                     | Increased inital costs                            |
        | Increase resilience and fault tolerance       |                                                   |
        | Increaseed performance                        |                                                   |
        
    - Vertical Scaling
        
        Vertical scaling refers to distribute more resources or add more power to the current machine (CloudZero, 2021). For example, by upgrading CPUs, increase RAM size to increase the server computing power.
        
        | Advantages                         | Disadvantages                   |
        | ---------------------------------- | ------------------------------- |
        | Cost-effective                     | Higher possibility for downtime | 
        | Less complex process communication | Single point of failure         |
        | Less complicated maintainance      | Upgrade limitation              |
        | Less need for software changes     |                                 |
    
    Depends on the demands, you may choose horizontal or vertical scaling based on factors like: cost, future-proofing, topographic distribution, reliability, upgradeability and flexibility, or performance and complexity.
    
- [x] Customer Relationship Management (CRM)

    Customer relationship management (CRM) is a technology for managing all your company’s relationships and interactions with customers and potential customers. The goal is simple: Improve business relationships to grow your business. A CRM system helps companies stay connected to customers, streamline processes, and improve profitability.

    When people talk about CRM, they are usually referring to a CRM system, a tool that helps with contact management, sales management, agent productivity, and more. CRM tools can now be used to manage customer relationships across the entire customer lifecycle, spanning marketing, sales, digital commerce, and customer service interactions.

    A CRM solution helps you focus on your organization’s relationships with individual people — including customers, service users, colleagues, or suppliers — throughout your lifecycle with them, including finding new customers, winning their business, and providing support and additional services throughout the relationship.
    
    More information please check [here](https://www.salesforce.com/crm/what-is-crm/).
    
- [ ] Game Theory

#### Programming skills
- [ ] [Regex]()
- [ ] [Linux System Commands]()
- [ ] [Shell Script]()
- [ ] [Python Decoration Function]()
- [ ] [Basic Web Scrapping]()
- [x] Classmethod vs. Staticmethod
    - A class method takes cls as the first parameter while a static method needs no specific parameters.
    - A class method can access or modify the class state while a static method can’t access or modify it.
    - In general, static methods know nothing about the class state. They are utility-type methods that take some parameters and work upon those parameters. On the other hand class methods must have class as a parameter.
    - We use @classmethod decorator in python to create a class method and we use @staticmethod decorator to create a static method in python.
    
    ```python
    # example of use of classmehtod and staticmethod
    class Distance:
        # a static method calculate the minkowski distance based on given array X, Y
        @staticmethod
        def Minkowski(X, Y, p):
            return np.power(np.sum(np.abs(X - Y)), (1 / p))

        # a class method calculate the Manhatten distance based on Distance object method
        @classmethod
        def Manhatten(clf, X, Y):
            return clf.Minkowski(X, Y, p=1)
    ```
    
#### Database skills
- [ ] [MySQL]()
- [ ] [Sqlite]()
- [ ] [MongoDB]()

#### Data Science Lifecycle
- [x] SMART Goals
    
    SMART goals stands for Specific, Measurable, Achievable, Relevant, and Time-Bound (Kat, 2021).
    
    Specific: You need to have a specific goal for effectiveness, in general you could ask question like:
    - What needs to be accomplished?
    - Who's responsible for it?  
    - What steps need to be taken to achieve it?
    
    > e.g. Grow the number of monthly users of Techfirm’s mobile app by optimizing our app-store listing and creating targeted social media campaigns.
    
    Measurable: Goals must be measurable, set milestones to check your working progress. Or setting a trackable benchmark.
    
    > Increase the number of monthly users of Techfirm’s mobile app by 1,000 by optimizing our app-store listing and creating targeted social media campaigns for four social media platforms: Facebook, Twitter, Instagram, and LinkedIn.
    
    Achievable: Goals must be realistic, realistic goals brings true outcomes!
    
    > Increase the number of monthly users of Techfirm’s mobile app by 1,000 by optimizing our app-store listing and creating targeted social media campaigns for three social media platforms: Facebook, Twitter, and Instagram.
    
    
    Relevant: Current goals must align with the project targets, think about the big picture, ask question: Why are you setting the goal that you're setting?
    
    > Grow the number of monthly users of Techfirm’s mobile app by 1,000 by optimizing our app-store listing and creating targeted social media campaigns for three social media platforms: Facebook, Twitter, and Instagram. Because mobile users tend to use our product longer, growing our app usage will ultimately increase profitability.
    
    Time-Bounded: To properly measure success, you and your team need to be on the same page about when a goal has been reached. Find a precise time-bound could help you track with a designed time framework.
    
    > Grow the number of monthly users of Techfirm’s mobile app by 1,000 within Q1 of 2022. This will be accomplished by optimizing our app-store listing and creating targeted social media campaigns, which will begin running in February 2022, on three social media platforms: Facebook, Twitter, and Instagram. Since mobile is our primary point of conversion for paid-customer signups, growing our app usage will ultimately increase sales.
    
- [x] OSEMN Framework
    
    OSEMN stands for Obtain, Scrub, Explore, Model, and iNterpret.
    
    <img src="https://miro.medium.com/max/1400/1*eE8DP4biqtaIK3aIy1S2zA.png" alt="Data Science Process (a.k.a the O.S.E.M.N. framework)" width=100%>
    
    **Obtain**: When the project start, we need to obtain data from available data sources. For example query from database, or using web scrapping techniques.
    
    **Scurb**: Clean the data and convert data into a canonical format for future task. For example, you may need to eliminate outliers or missing values and standardising data into a uniform format.
    
    **Explore**: Check dataset insights to find meaningful feature for future machine learning work.
    
    **Model**: Construct models based on explored features.
    
    **iNterpret**: Explain the model by checking the ability for unseen data, or using visualisation to cheek its performance.
    
- [x] Team Data Science Process Framework (TDSP)
    
    The Team Data Science Process (TDSP) is an agile, iterative data science methodology to deliver predictive analytics solutions and intelligent applications efficiently. TDSP helps improve team collaboration and learning by suggesting how team roles work best together. TDSP includes best practices and structures from Microsoft and other industry leaders to help toward successful implementation of data science initiatives. The goal is to help companies fully realize the benefits of their analytics program.
    
    Detailed description is available on [Microsoft Doc](https://docs.microsoft.com/en-us/azure/architecture/data-science-process/overview).
    
    The framework main focus on the lifecycle outlines the major stages that projects typically execute, often iteratively:
    - Business understanding
    - Data acquisition and understanding
    - Modeling
    - Deployment
    
    <img src="https://docs.microsoft.com/en-us/azure/architecture/data-science-process/media/overview/tdsp-lifecycle2.png" witdh=100%>
    
    It provides a standard workflow for working on a big data analysis project or construct a data oriented product. Microsoft also provides a [template repository](https://github.com/Azure/Azure-TDSP-ProjectTemplate) for project initiate.
    
- [x] Cross-Industry Standard Process for Data Mining Framework (CRISP-DM)

    From [Wikipedia](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining): Cross-industry standard process for data mining, known as CRISP-DM,[1] is an open standard process model that describes common approaches used by data mining experts. It is the most widely-used analytics model.[2]

    In 2015, IBM released a new methodology called Analytics Solutions Unified Method for Data Mining/Predictive Analytics[3][4] (also known as ASUM-DM) which refines and extends CRISP-DM.
    
    It go through six main phases:
    - Business understanding: understand business content and requirements
    - Data understanding: understand the data based on meta data or provided data dictionary
    - Data preparation: processing data in a nice and clean format
    - Modeling: construct data product
    - Evaluation: check data product performance
    - Deployment: publish the product

    The sequence of the phases is not strict and moving back and forth between different phases is usually required. The arrows in the process diagram indicate the most important and frequent dependencies between phases. The outer circle in the diagram symbolizes the cyclic nature of data mining itself. A data mining process continues after a solution has been deployed. The lessons learned during the process can trigger new, often more focused business questions, and subsequent data mining processes will benefit from the experiences of previous ones.
    
- [x] Agile Project Management ([Atlassian explanation](https://www.atlassian.com/agile))

    Agile is an iterative approach to project management and software development that helps teams deliver value to their customers faster and with fewer headaches. Instead of betting everything on a "big bang" launch, an agile team delivers work in small, but consumable, increments. Requirements, plans, and results are evaluated continuously so teams have a natural mechanism for responding to change quickly.
    
    Agile project management is an iterative approach to managing software development projects that focuses on continuous releases and incorporating customer feedback with every iteration.

    Software teams that embrace agile project management methodologies increase their development speed, expand collaboration, and foster the ability to better respond to market trends.

    Usually, project team use scrum framework, scrum describes a set of meetings, tools, and roles that work in concert to help teams structure and manage their work.
    
    In an agile project management, the basic workflow is:
    - TO DO: Work that has not been started
    - IN PROGRESS: Work that is actively being looked at by the team
    - CODE REVIEW: Work that is completed and awaiting review
    - DONE: Work that is completely finished and meets the team's definition of done.
    
    Except these four statuses, it could also include state like BLOCK, ON HOLD to explicitly indicate a project working states. Also, these project statuses can also be shared with the rest of the organization. When building a workflow, think about what metrics are important to report on and what non-team members might be interested in learning. For example, a well designed workflow answers the following questions:
    - What work has the team completed?
    - Is the backlog of work increasing or keeping pace with the team?
    - How many items are in each status?
    - Are there any bottlenecks that are slowing the team down?
    - How long does it take to complete an average task?
    - How many work items didn't pass our quality standards the first time around?
    
    Optimizing the workflow leads to better productivity and keep the development team fully utilized.
    
    | Advantage of Agile                                  | Disadvantage of Agile                                                                                                         |
    | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
    | Faster feedback cycles                              | Critical path and inter-project dependencies may not be clearly defined as in waterfall                                       |
    | Identifies problems early                           | There is an organizational learning curve cost                                                                                |
    | Higher potential for customer satisfaction          | True agile execution with a continuous deployment pipeline has many technical dependencies and engineering costs to establish |
    | Better visibility / accountability                  |                                                                                                                               |
    | Dedicated teams drive better productivity over time |                                                                                                                               |
    | Flexible prioritization focused on value delivery   |                                                                                                                               |
    
- [x] Waterfall Project Management （[wrike explanation](https://www.wrike.com/project-management-guide/faq/what-is-waterfall-project-management/)）

    Waterfall project management is the most straightforward way to manage a project.

    Waterfall project management maps out a project into distinct, sequential phases, with each new phase beginning only when the previous one has been completed. The Waterfall system is the most traditional method for managing a project, with team members working linearly towards a set end goal. Each participant has a clearly defined role, and none of the phases or goals are expected to change.

    Waterfall project management works best for projects with long, detailed plans that require a single timeline. Changes are often discouraged (and costly). In contrast, Agile project management involves shorter project cycles, constant testing and adaptation, and overlapping work by multiple teams or contributors.
    
    - Requirements: The manager analyzes and gathers all the requirements and documentation for the project.
    - System design: The manager designs the project’s workflow model.
    - Implementation: The system is put into practice, and your team begins the work.
    - Testing: Each element is tested to ensure it works as expected and fulfills the requirements.
    - Deployment (service) or delivery (product): The service or product is officially launched.
    - Maintenance: In this final, ongoing stage, the team performs upkeep and maintenance on the resulting product or service.
    
    The waterfall project management approach entails a clearly defined sequence of execution with project phases that do not advance until a phase receives final approval. Once a phase is completed, it can be difficult and costly to revisit a previous stage. Agile teams may follow a similar sequence yet do so in smaller increments with regular feedback loops. 
    
    | Advantage of Waterfall                                                            | Disadvantage of Waterfall                                                                                                         |
    | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
    | Requires less coordination due to clearly defined phases sequential processes     | Harder to break up and share work because of stricter phase sequences teams are more specialized                                  |
    | A clear project phase helps to clearly define dependencies of work                | Risk of time waste due to delays and setbacks during phase transitions                                                            |
    | The cost of the project can be estimated after the requirements are defined       | Additional hiring requirements to fulfill specialized phase teams whereas agile encourages more cross-functional team composition |
    | Better focus on documentation of designs and requirements                         | Extra communication overhead during handoff between phase transitions                                                             |
    | The design phase is more methodical and structured before any software is written | Product ownership and engagement may not be as strong when compared to agile since the focus is brought to the current phase      |

## Reference

All references' style follow the APA7 format based on [UoM APA7 Guide](https://library.unimelb.edu.au/recite/referencing-styles/apa7).

**Repositories**
- MLfromscratch, https://github.com/python-engineer/MLfromscratch/.
- Azure-TDSP-ProjectTemplate, https://github.com/Azure/Azure-TDSP-ProjectTemplate.

**Math Miscs**
- Wikipedia. (14, Jan 2022). *Minkoski distance*. https://en.wikipedia.org/wiki/Minkowski_distance.
- Wikipedia. (21, Apr 2022). *Probability theory*. https://en.wikipedia.org/wiki/Probability_theory.
- Wikipedia. (20, Feb 2022). *Statistics*. https://en.wikipedia.org/wiki/Statistics.
- Wikipedia. (6, May 2022). *Operational research*. https://en.wikipedia.org/wiki/Operations_research.
- Wikipedia. (21, May 2022). *Voting*. https://en.wikipedia.org/wiki/Voting#Voting_methods.
- Elaine, J H & Jonathan, G. (11, Nov 2021). *What is mathematics*. https://www.livescience.com/38936-mathematics.html. 

**Machine Learning Articles**
- Onel, H. (11, Sep 2018). *Machine Learning Basics with the K-Nearest Neighbors Algorithm*. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
- Wikipedia. (26, May 2022). *Machine learning*. https://en.wikipedia.org/wiki/Machine_learning.

**Artificial Intelligence Articles**
- Wikipedia. (30, May, 2022). *Artificial Intelligent*. https://en.wikipedia.org/wiki/Artificial_intelligence.
- Java T Point. (2022). *Artificial Intelligent*. https://www.javatpoint.com/artificial-intelligence-tutorial.

**Data Structure**
- David, L & Sarah, L. (Mar, 2021). *Data Structure*. Search Data Management. https://www.techtarget.com/searchdatamanagement/definition/data-structure.
- Java T Point. (2022). *Linked list*. https://www.javatpoint.com/ds-linked-list.

**Others**
- GO FAIR. (2021). *FAIR Principles**. https://www.go-fair.org/fair-principles/.
- Isaac, S. (15, Apr 2022). *What is CI/CD? Continuous integration and continuous delivery explained*. https://www.infoworld.com/article/3271126/what-is-cicd-continuous-integration-and-continuous-delivery-explained.html.
- Humanperf Software (3, May 2018). *NowIUnderstand Glossary: the 5W1H method*. https://www.humanperf.com/en/blog/nowiunderstand-glossary/articles/5W1H-method.
- David, G. (16, May 2021). *The 5W1H Method: Project management defined and applied*. https://www.wimi-teamwork.com/blog/the-5w1h-method-project-management-defined-and-applied/.
- Wikipedia. (7, May 2022). *Scalability*. https://en.wikipedia.org/wiki/Scalability.
- CloudZero. (30, Jun 2021). *Horizontal Vs. Vertical Scaling: How Do They Compare?*. https://www.cloudzero.com/blog/horizontal-vs-vertical-scaling.
- Kat, B. (26, Dec 2021). *How to write SMART goals*. https://www.atlassian.com/blog/productivity/how-to-write-smart-goals.
- Geeks For Geeks. (24, Aug 2021). *Class method vs Static method in Python*. https://www.geeksforgeeks.org/class-method-vs-static-method-python/.
- Cher, H L. (3, Jan 2019). *5 Steps of a Data Science Project Lifecycle*. https://towardsdatascience.com/5-steps-of-a-data-science-project-lifecycle-26c50372b492.
- Salesforce. (2021). *CRM 101: What is CRM*. https://www.salesforce.com/crm/what-is-crm/.
- Wikipedia. (5, Apr 2022). *Cross-industry standard process for data mining*. https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining.
- Microsoft. (2022). *What is the Team Data Science Process?*. https://docs.microsoft.com/en-us/azure/architecture/data-science-process/overview.
- Atlassian. (2022). *Agile Coach*. https://www.atlassian.com/agile.
- Wrike. (2022). *What Is Waterfall Project Management?*. https://www.wrike.com/project-management-guide/faq/what-is-waterfall-project-management/.
- APPDYNAMICS. (2022). *What is Database Management Systems (DBMS)?*. https://www.appdynamics.com/topics/database-management-systems.
- Wikipedia. (16, May 2022). *Dimensionality reduction*. https://en.wikipedia.org/wiki/Dimensionality_reduction.
