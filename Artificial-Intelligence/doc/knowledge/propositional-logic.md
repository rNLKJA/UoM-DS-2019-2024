### Propositional Logic in Artificial Intelligence

Propositional logic (PL) is the simplest form of logic where all the statements are made by propositions. A proposition is a declarative statement which is either true of false. It is a technique or knowledge representation in logical and mathematical form.

```
# Examples
1. It is Sunday
2. The sun rises from West (False proposition)
3. 3 + 3 = 7 (False proposition)
4. 5 is a prime number.
```

*Following are some basic facts about propositional logic*
- Propositional logic also called Boolean logic as it works on 0 and 1.
- In propositional logic, we use symbolic variables to represent the logic, and we can use any symbol for a representing a proposition, such A, B, C, P, Q, R, etc.
- Propositions can be either true or false, but it cannot be both.
- Propositional logic consists of an object, relations or function, and logical connectives.
- These connectives are also called logical operators.
- The propositions and connectives are the basic elements of the propositional logic.
- A proposition formula which is always false is called Contradiction.
- A proposition formula which has both true and false values is called statement.
- Statements which are questions, commands, or opinions are not propositions such as "Where is Rohini", "How are you", "What is your name", are not propositions.

*Syntax of propositional logic*

The syntax of propositional logic defines the allowable sentences for the knowledge representation. There are two types of propositions:
- Atomic proposition
    
    Atomic propositions are the simple propositions. It consists of a single proposition symbol. These are the sentences which must be either true or false.
    
- Compound proposition

    Compound propositions are constructed by combining simpler or atomic propositions, using parenthesis and logical connectives.
    
### Logical Connectives

Logical connectives are used to connect two simpler propositions or representing a sentence logically. We can create compound propositions with the help of logical connectives. There are mainly five connectives, which are given as follows:
- Negation: A sentence such as ¬P is called negation of P. A literal can be either positive literal or negative literal.
- Conjunction: A sentence which has ∧ connective such as P ∧ Q is called conjunction.
- Disjunction: A sentence which has ∨ connective, such as P ∨ Q is called disjunction.
- Implication: A sentence such as P → Q is called an implication. Implication are also known as if-then rules. 
- Biconditional: A sentence such as P ⇔ Q is a biconditional sentence.

| Connective Symbols | Word           | Technical Term | Example  |
| ------------------ | -------------- | -------------- | -------- |
| ¬                  | NOT            | Negation       | ¬A or ¬B |
| ∧                  | AND            | Conjunction    | A ∧ B    |
| ∨                  | OR             | Disjunction    | A ∨ B    |
| →                  | Implies        | Implication    | A → B    |
| ⇔                 | IF and ONLY IF | Biconditional  | A ⇔ B   |

### Truth Table

In propositional logic, we need to know the truth values of propositions in all possible scenarios. We can combine all the possible combination with logical connectives, and the representation of these combinations in a tabular format is called Truth table. Following are the truth table for all logical connectives.

![propositional-logic-in-ai2](./img/propositional-logic-in-ai2.png)

### Truth Table with 3 propositions

We can build a proposition composing three propositions P, Q and R. This Truth table is made-up of 8n Tuples as we have taken three proposition symbols.

![propositional-logic-in-ai4](./img/propositional-logic-in-ai4.png)

### Precedence of Connectives

Just like arithemtic operators, there is a precedence order for propositional connectors or logical operators. This order should be followed while evaluating a propositional problem. Following is the list of the precendence order for operators:

| Precedence        | Operators     |
| ----------------- | ------------- |
| First Precedence  | Parenthesis   |
| Second Precedence | Negation      |
| Third Precedence  | Conjunction   |
| Fourth Precedence | Disjunction   |
| Fifth Precedence  | Implication   |
| Six Precedence    | Biconditional |

### Logical Equivalence

Logical equivalence is one of the feature of propositional logic. Two propositions are said to be logically equivalent if and only if the columns in the truth table are identical to each other. Let's take two propositions A and B, so logic equivalence, we can write it as A ⇔ B. In below truth table we can see that column for ¬A ∨ B and A → B, are identical hence A is equivalent to B.

![propositional-logic-in-ai5](./img/propositional-logic-in-ai5.png)

### Properties of Operators

- Communtativity: 
    - P ∧ Q= Q ∧ P
    - P ∨ Q = Q ∨ P.
- Associativity: 
    - (P ∧ Q) ∧ R= P ∧ (Q ∧ R)
    - (P ∨ Q) ∨ R= P ∨ (Q ∨ R)
- Identity element: 
    - P ∧ True = P
    - P ∨ True= True
- Distributive: 
    - P ∧ (Q ∨ R) = (P ∧ Q) ∨ (P ∧ R)
    - P ∨ (Q ∧ R) = (P ∨ Q) ∧ (P ∨ R).
- DE Morgan's Law
    - ¬ (P ∧ Q) = (¬P) ∨ (¬Q)
    - ¬ (P ∨ Q) = (¬ P) ∧ (¬Q)
- Double-negation elimination
    - ¬ (¬P) = P

### Limitations of Propositional Logic
- We cannot represent relations like ALL, some, or none with propositional logic:
    - e.g. All the boys are intelligent
    - e.g. Some apples are sweet
- Propositional logic limited expressive power.
- In propositional logic, we cannot describe statements in terms of their properties or logical relationships.