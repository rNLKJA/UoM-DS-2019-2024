### Resolution
Resolution is a theorem proving technique that proceeds by building refutation proofs, i.e., proofs by contradictions. It was invented by a Mathematician John Alan Robinson in the year 1965.

Resolution is used, if there are various statements are given, and we need to prove a conclusion of those statements. Unification is a key concept in proofs by resolutions. Resolution is a single inference rule which can efficiently operate on the conjunctive normal form or clausal form.

**Clause**: Disjunction of literals (an atomic sentence) is called a clause. It is also known as a unit clause.

**Conjunctive Normal Form**: A sentence represented as a conjunction of clauses is said to be conjunctive normal form or CNF.

### The resolution inference rule:
The resolution rule for first-order logic is simply a lifted version of the propositional rule. Resolution can resolve two clauses if they contain complementary literals, which are assumed to be standardized apart so that they share no variables.

![ai-resolution-in-first-order-logic](./ai-resolution-in-first-order-logic.png)

Where l_i and m_j are complementary literals.

This rule is also called the binary resolution rule because it only resolves exactly two literals.

### Steps for Resolution:
Conversion of facts into first-order logic.
Convert FOL statements into CNF
Negate the statement which needs to prove (proof by contradiction)
Draw resolution graph (unification).
To better understand all the above steps, we will take an example in which we will apply resolution.