### Forward Chaining vs. Backward Chaining

**Following is the difference between the forward chaining and backward chaining:**
- Forward chaining as the name suggests, start from the known facts and move forward by applying inference rules to extract more data, and it continues until it reaches to the goal, whereas backward chaining starts from the goal, move backward by using inference rules to determine the facts that satisfy the goal.
- Forward chaining is called a data-driven inference technique, whereas backward chaining is called a goal-driven inference technique.
- Forward chaining is known as the down-up approach, whereas backward chaining is known as a top-down approach.
- Forward chaining uses breadth-first search strategy, whereas backward chaining uses depth-first search strategy.
- Forward and backward chaining both applies Modus ponens inference rule.
- Forward chaining can be used for tasks such as planning, design process monitoring, diagnosis, and classification, whereas backward chaining can be used for classification and diagnosis tasks.
- Forward chaining can be like an exhaustive search, whereas backward chaining tries to avoid the unnecessary path of reasoning.
- In forward-chaining there can be various ASK questions from the knowledge base, whereas in backward chaining there can be fewer ASK questions.
- Forward chaining is slow as it checks for all the rules, whereas backward chaining is fast as it checks few required rules only

| Forward Chaining | Backward Chaining |
| ---- | ---- |
| Forward chaining starts from known facts and applied inference rule to extract more data unit it reaches to the goal | Backward chaining starts from the goal and works backward through inference rules to find the required facts that support the goal |
| It is a bottom-up approach | It is a top-down approach |
| Forward chaining is known as data-driven inference technique as we reach to the goal using the available data | Backward chaining is known as a goal-driven technique as we start from the goal and divide into sub-goal to extract the facts |
| Forward chaining reasoning appplies a breadth-first search strategy | Backward chaining reasoning applies a depth-frist search strategy |
| Forward chaining tests for all the available rules | Backward chaining only tests for few required rules |
| Forward chaining is suitable for the planning, monitoring, control, and interpretation application | Backward chaining is suitable for diagnostic, prescription, and debugging application |
| Forward chaining can generate an infinite number of possible conclusions | Backward chaining generates a finite number of possible conclusions |
| It operates in the forward direction | It operates in the backward direction |
| Forward chaining is aimed for any conclusion | Backward chaining is only aimed for the required data |