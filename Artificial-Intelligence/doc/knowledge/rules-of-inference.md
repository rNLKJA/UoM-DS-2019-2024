### Rules of Inference in Artifical Intelligence

In artificial intelligence, we need intelligent computers which can create new logic from old logic or by evidence, so generating the conclusions from evidence and facts is termed as inference.

### Inference Rules

Inference rules are the templates for generating valid arguments. Inference rules are applied to derive proofs in artificial intelligence, and the proof is a sequence of the conclusion that leads to the desired goal.

In inference rules, the implication among all the connectives plays an important role. Following are some terminologies related to inference rules:
- Implication: It is one of the logical connectives which can be represented as P → Q. It is a Boolean expression.
- Converse: The converse of implication, which means the right-hand side proposition goes to the left-hand side and vice-versa. It can be written as Q → P.
- Contrapositive: The negation of converse is termed as contrapositive, and it can be represented as ¬ Q → ¬ P.
- Inverse: The negation of implication is called inverse. It can be represented as ¬ P → ¬ Q.

From the above term some of the compound statements are equivalent to each other, which we can prove using truth table:

![rules-of-inference-in-ai](./img/rules-of-inference-in-ai.png)

Hence from the above truth table, we can prove that P → Q is equivalent to ¬ Q → ¬ P, and Q→ P is equivalent to ¬ P → ¬ Q.

### Types of Inference Rules

#### Modus Ponens

The Modus Ponens rule is one of the most important rules of inference, and it states that if P and P → Q is true, then we can infer that Q will be true. It can be represented as:

![rules-of-inference-in-ai2](./img/rules-of-inference-in-ai2.png)

#### Modus Tollens

The Modus Tollens rule state that if P→ Q is true and ¬ Q is true, then ¬ P will also true. It can be represented as:

![rules-of-inference-in-ai4](./img/rules-of-inference-in-ai4.png)

#### Hypothetical Syllogism

The Hypothetical Syllogism rule state that if P → R is true whenever P → Q is true, and Q → R is true. 

#### Disjunctive Syllogism

The Disjunctive syllogism rule state that if P ∨ Q is true, and ¬P is true, then Q will be true. It can be represented as:

![rules-of-inference-in-ai7](./img/rules-of-inference-in-ai7.png)


##### Addition

The Addition rule is one the common inference rule, and it states that If P is true, then P ∨ Q will be true.

![rules-of-inference-in-ai9](./img/rules-of-inference-in-ai9.png)

##### Simplification

The simplification rule state that if P∧ Q is true, then Q or P will also be true. It can be represented as:

![rules-of-inference-in-ai11](./img/rules-of-inference-in-ai11.png)

##### Resolution

The Resolution rule state that if P∨Q and ¬ P∧R is true, then Q∨R will also be true. It can be represented as

![rules-of-inference-in-ai13](./img/rules-of-inference-in-ai13.png)