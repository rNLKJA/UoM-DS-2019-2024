### Reasoning in Artificial Intelligence

The reasoning is the mental process of deriving logical conclusion and make predictions from available knowledge, facts, and beliefs. Or we can say, "Reasoning is a way to infer facts from existing data." It is a general process of thinking rationally, to find valid conclusions.

In artificial intelligence, the reasoning is essential so that the machine can also think rationally as human brain, and can perform like a human.

#### Types of Reasoning
- **Deductive reasoning**

    Deductive reasoning is deducing new information from logically related known information. It is the form of valid reasoning, which means the argument's conclusion must be true when the premises are true.
    
    Deductive reasoning is a type of propositional logical in AI, and it requires various rules and facts. It is sometimes referred to as top-down reasoning, and contradictory to inductive reasoning.
    
    In deductive reasoning, the truth of the premises guarantees the truth of the conclusion.
    
    Deductive reasoning mostly starts from the general premises to the specific conclusion, which can be explained as below example:
        - Premise-1: All the human eats veggies
        - Premise-2: Suresh is human
        - Conclusion: Suresh eats veggies
    
    The general process of deductive reasoning is given as: Theory -> Hypothesis -> Patterns -> Confirmation

- **Inductive reasoning**

    Inductive reaonsing is a form of reasoning to arrive at a conclusion using limited sets of facts by the process of generalization. It starts with the series of specific facts or data and reaches to a general statement or conclusion.
    
    Inductive reasoning is a type of propositional logical, which is also known as cause-effect reasoning or bottom-up reasoning.
    
    In inductive reasoning, we use historical data or various premises to generate a generic rule, for which premises support the conclusion.
    
    In inductive reaonsing, premises provide probable supports to the conclusion, so the truth of premises does not guarantee the truth of the decision.
        - Premise: all of the pigeons we have seen in the zoo are white
        - Conclusion: therefore, we can expect all the pigeons to be white

    The general process of inductive reasoning is given by: Observations -> Patterns -> Hypothesis -> Theory
    
- **Abductive reasoning**

    Abductive reaonsing is a form of logical reasoning which starts with single or multiple observations then seeks to find the most likely explanation or conclusion for the observation.
    
    Abductive reasoning is an extension of deductive reasoning, but in abductive reasoning, the premises do not guarantee the conclusion.
        - Implication: Cricket ground is wet if it raining
        - Axiom: Cricket ground is wet
        - Conclusion: It is raining
        
- **Common sense reasoning**

    Common sense reasoning is an informal form of reasoning, which can be gained through experiences. Common sense reasoning simulates the human ability to make presumptions about events which occurs on every day. It relies on good judgment rahter than exact logic and operates on heuristic knowledge and heuristic rules.
    
    Examples:
    - One person can be at one place at a time
    - If I put my hand in a fire, then it will burn
    
    The above two statemnets are the examples of common sense reasoning which a human mind can easily understand and assume.
    
- **Monotonic reasoning**

    In monotonic reasoing, once the conclusion is taken, then it will remain the same even if we add some other information to existing information in our knowledge base. In monotonic reasoning, adding knowledge does not decrease the set of prepositions that can be derived. To solve monotonic problems, we can derive the valid conclusion from the available facts only, and it will not be affected by new facts. Monotonic reasoning is not useful for the real-time systems, as in real time, facts get changed, so we cannot see monotonic reasoning. Monotonic reasoning is used in conventional reasoning systems, and a logic-based system is monotonic. Any theorem proving is an example of monotonic reasoning.
    
    Example: Earch revolves around the Sun.
    
    It is a true fact, and it cannot be changed even if we add another sentence in knowledge base like, "The moon revolves around the earth" Or "Earth is not round", etc.
    
    *Advantages of Monotonic Reasoning*
    - In monotonic reasoning, each old proof will always remain valid.
    - If we deduce some facts from available facts, then it will remain valid for always.
    
    *Disadvantage of Monotonic Reasoning*
    - We cannot represent the real worl scenarios using Monotonic reasoning.
    - Hypothesis knowlege cannot be expressed with monotonic reasoning, which means facts should be true.
    - Since we can only derive conclusions from the old proofs, so new knowledge from the real world cannot be added.
    
- **Non-monotonic reasoning**

    In non-monotonic reasoning, some conclusions may be invalided if fwe add some more information to our knowledge base. Logical will be said as non-monotonic if some conclusions can be invalidated by adding more knowledge into our knowledge base. Non-monotonic reasoning deals with incomplete and uncertain models. "Human perceptions for various things in daily life" is a general example of non-monotonic reasoning.
    
    Example: Let suppose the knowledge base contains the following knowledge:
    - Brids can fly
    - Penguins cannot fly
    - Pitty is a bird
    
    So from the above sentences, we can conclude that Pitty can fly.
    
    However, if we add one another sentence into knowledge base "Pitty is a penguin", which concludse "Pitty cannot fly", so it invalidates the above conclusion.
    
    *Advantages of non-monotonic reasoning*
    - For real-world systems such as Robot navigation, we can use non-monotonic reasoning.
    - In non-monotonic reasoning, we can choose probabilistic facts or can make assumption.
    
    *Disadvantages of non-monotonic reasoning*
    - In non-monotonic reasoning, the old facts may be invalidated by adding new sentences.
    - It cannot be used for theorem proving.