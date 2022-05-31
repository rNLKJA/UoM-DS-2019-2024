### Agent Environment in AI

An environment is everything in the world which surrounds the agent, but it is not a part of an agent itself. An environment can be described as a situation in which an agent is present.

The environment is where agent lives, operate and provide the agent with something to sense and act upon it. An environment is mostly said to be non-feministic.

### Features of Environment

As per Russel and Norvig, an environment can have various features from the point of view of an agent:
- **Fully observable vs. Partially Observable**
    
    - If an agent sensor can sense or access the complete state of an environment at each point of time then it is a fully observable environment, else it is partially observable.
    - A fully observable environment is easy as there is no need to maintain the internal state to keep track history of the world.
    - An agent with no sensors in all environments then such an environment is called as unobservable.

- **Determinstic vs. Stochastic**

    - If an agent's current state and selected action can completely determine the next state of the environment, then such environment is called a determinstic environment.
    - A stochastic environment is random in nature and cannot be determined completely by an agent.
    - In a determinstic, fully observable environment, agent does not need to worry about uncertainty.

- **Episodic vs. Sequential**
    
    - In an epsiodic environment, there is a series of one-shot actions, and only the current percept is required for the action.
    - Howver, in Sequential environment, an agent requires memory of past actions to determine the next best actions.

- **Single-agent vs. Multi-agent**
    
    - If only one agent is invloed in an environment, and operating by itself then such an environment is called single agent environment.
    - However, if multiple agents are operating in an envrionment, then such an environment is called a multi-agent environment.
    - The agent design problems in the multi-agent environment are different from single agent environment.

- **Static vs. Dynamic**

    - If the environment can change itself while an agent is deliberating then such environment is called a dynamic environment else it is called a static environment.
    - Static environments are easy to deal because an agent does not need to continuou looking at the world while deciding for an action.
    - However, for dynamic environment, agetns need to keep looking at the world at each action.
    - Taxi driving is an exmaple of a dynamic environment whereas Crossword puzzles are an example of a static environment.

- **Discrete vs. Continuous**

    - If in an environment there are a finite number of percepts and actions that can be performed within it, then such an environment is called a discrete environment else it is called continous environment.
    - A chess gamecomes under discrete environment as there is a finite number of moves that can be performed.
    - A self-driving car is an example of continuous environment.

- **Known vs. Unknown**

    - Known and unknown are not actually a feature of an environment, but it is an agent's state of knowledge to perform an action.
    - In a known environment, the results for all actions are known to the agent. While in unknown environment, agent needs to learn how it works in order to perform an action.
    - It is quite possible that a known environment to be partially observable and an Unknown environment to be fully observable.

- **Accessible vs. Inaccessible**
    
    - If an agent can obtain complete and accurate information about the state's environment, then such an environment is called an Accessible environment else it is called inaccessible.
    - An empty room whose state can be defined by its temperature is an example of an accessible environment.
    - Information abuot an event on earth is an example of Inaccessible environment.

