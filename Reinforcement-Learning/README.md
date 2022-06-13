<img src="./img/ricardo-gomez-angel-z6CcN8rlftY-unsplash.jpg" width=100% />

<div align=center><h3>Reinforcement Learning</h3></div>

Reinforcement learning is a feedback-based learning method, in which a learning agent gets a reward for each right action and gets a penalty for each wrong action. The agent learns automatically with these feedbacks and improves its performance. In reinforcement learning, the agent interacts with the environment and explores it. The goal of an agent is to get the most reward points, and hence its performance. 

- [ ] [What is Reinforcement Learning](#what-is-reinforcement-learning)
- [ ] [Terms used in Reinforcement Learning](#terms-used-in-reinforcement-learning)
- [ ] [Key features of Reinforcement Learning](#key-features-of-reinforcement-learning)
- [ ] [Elements of Reinforcement Learning]()
- [ ] [Approaches to implementing Reinforcement Learning](#approaches-to-implement-reinforcement-learning)
- [ ] [How does Reinforcement Learning Work]()
- [ ] [The Ballman Equation]()
- [ ] [Types of Reinforcement Learning]()
- [ ] [Reinforcement Learnnig Algorithm]()
- [ ] [Markov Decision Proces]()
- [ ] [What is Q-Learning]()
- [ ] [Difference between Supervised Leraning and Reinforcement Learning]()
- [ ] [Applications of Reinforcement Learning]()
- [ ] [Conclusion]()

### What is Reinforcement Learning

Reinforcement learning is a feedback-based machine learning technique in which an agent learns to behave in an environment by performing the actions and seeing the results of actions. For each good action, the agent gets positive feedback, and for each bad action, the agent gets negative feedback or penalty. 

In reinforcement learning, the agent learns automatically using feedbacks without any labeled data, unlike supervised learning. Since there is no labeled data, so the agent is bound to learn by its experience only.

RL solves a specific type of problem where decision making is sequential, and the goal is long-term, such as game-playing, robotics, etc.

The agent interacts with the environment and explores it by itself. The primary goal of an agent in reinforcement learning is to improve the performance by geeting the maximum positive rewards.

The agent learns with the process of hit and trail, and based on the experience, it learns to perform the task in a better way. Hence we can say that:

> Reinforcement learning is a type of machine learning method where an intelligent agent (computer program) interacts with the environment and learns to act within that.

### Terms used in Reinforcement Learning
- Agent: 
    - An entity that can perceive/explore the environment and act upon it.
- Environment: 
    - A situation in which an agent is present or surrounded by. In RL, we assume the stochastic environment, which means it is random in nature.
- Action: 
    - Actions are the moves taken by an agent within the environment.
- State:
    - State is a situation return by the environment after each action taken by the agent.
- Reward:
    - A feedback to the agent from the environment to evaluate the action of the agent.
- Value:
    - It is expected long-term returned with the discount factor and opposite to the short-term reward.
- Q-value:
    - It is mostly similar to the value, but it takes one additional parameter as a current action.
    
### Key Features of Reinforcement Learning
- In RL, the agent is not instructed about the environment and what actions need to be taken.
- It is based on the hit and trial process.
- The agent takes the next action and changes states according to the feedback or the previous action.
- The agent may get a delay reward.
- The environment is stochastic, and the agent needs to explore it to reach to get the maximum positive rewards.

### Approaches to implement Reinforcement Learning
There are mainly three ways to implement reinforcement learning in machine learning, which are:
- Value-based:
    
    The value-based approach is about to find the optimal value function, which is the maximum value at a state under any policy. Therefore, the agent expects the long-term return at any state(s) under policy π.

- Policy-based:
    
    Policy-based approach is to find the optimal policy for the maximum future rewards without using the value function. In this approach, the agent tries to apply such a policy that the action performed in each step helps to maximize the future reward. The policy-based approach has mainly two types of policy:
    - Deterministic: The same action is produced by the policy (π) at any state.
    - Stochastic: In this policy, probability determines the produced action.

- Model-based:
    
    In the model-based approach, a virtural model is created for the environment, and the agent explores that environment to learn it. There is no particular solution or algorithm for this approach because the model representation is different for each environment.