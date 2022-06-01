### Means-Ends Analysis

Means-Ends Analysis is problem-solving techniques used in Artificial intelligence for limiting search in AI programs. It is a mixture of Backward and Forward search technique. The MEA process centered on the evaluation of the difference between the curernt state and goal state.

### How Means-Ends Analysis works

The means-ends analysis process can be applied recursively for a problem. It is a strategy to control search in problem-solving. Following are the main steps which describes the working MEA technique for solving a problem:
1. Evaluate the difference between initial state and final state
2. Select the various operators which can be applied for each difference
3. Apply the operator at each difference, which reduces the difference between the current state and goal state

### Operator Subgoaling

In the MEA process, we detect the differences between the current state and goal state. Once these differences occur, then we can apply an ooperator to reduce the differences. But sometimes it is possible that an operator cannot be applied to the current state. So we create the subproblem of the state, in which operator can be applied, such type of backward chaining in which operators are selected, and then sub goals are set up to eatablish the preconditions of the operator is called operator subgoaling.

### Algorithm for Means-Ends Analysis

```
Step 1: Compare CURRENT to GOAL, if there are no difference between both then return success and exit

Step 2: Else, select the most significant difference and reduce it by doing the following steps until the success or failure occues
    - select a new operator O which is applicable for the current difference, and if there is no such operator, then signal failure
    - Attempt to apply operator O to CURRENT. Make a description of two states
        - O-START, a state in which O's preconditions are satisfied
        - O-RESULT, the state that would result if O were applied in O-START
    If (First-Part <- MEA(CURRENT, O-START)) AND (Last-Part <- MEA(O-Result, Goal)) are successful, then signal success and return the result of combing First-Part, O and Last-Part.
```