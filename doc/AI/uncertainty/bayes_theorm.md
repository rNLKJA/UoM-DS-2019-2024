### Bayes' Theorem in Artificial Intelligence

### Bayes' Theorem

Bayes' theorem is also known as Baye's rule, Bayes' Law, or Bayesian reasoning, which detemines the probability of an event with uncertain knowledge. In probability theory, it relates the conditional probability and marginal probabilities of two random events. Bayes' theorem was named after the British matehmatician Thomas Bayes. The Bayesian inference is an application of Bayes' theorem, which is fundamental to Bayesian statistics. It is a way to calculate the value of P(B|A) with the knowledge P(A|B). Bayes' theorem allows updating the probability prediction of an event by observing new information of the real world.

**Example**

If cancer corresponds to one's age then by using Bayes' theorem, we can determine the probability of cancer more accurately with the help of age.
Bayes' theorem can be derived using product rule and conditional probability of event A with known event B, as from the product rule we can write:
P(A ∩ B) = P(A|B)P(B) or similarly, the probabiilty of event B with known event A: P(A ∩ B) = P(B|A)P(A).

Equating right hand side of both the equations, we will get: P(A|B) = P(B|A)P(A) / P(B).

The above equation is called as Bayes' rule or Bayes' theorem. This equation is basic of most modern AI systems for probabilistic inference.

It shows the simple relationship between joint and conditional probabilities. Here,
- P(A|B) is known as posterior, which we need to calculate, and it will be read as probabilty of hypothesis A when we have occurred an evidence B.
- P(B|A) is called the likelihood, in which we consider that hypothesis is true, then we calculate the probability of evidence.
- P(A) is called the prior probability, probability of hypothesis before considering the evidence.
- P(B) is called the marginal probability, pure probability of an evidence.

In the bayes' rule, in general, we can write P(B) = P(A) * P(B|A), hence bayes' rule can be written as: P(A_i | B) = P(A_i) * P(B | A_i) / \sum P(A_i) * P(B|A_i). Where A_i is a set of mutually exclusive and exhaustive events.

### Applying Bayes' rule:
Bayes' rule allow us to compute the single term P(B|A) in terms of P(A|B), P(B), and P(A). This is very useful in cases where we have good probability of these three terms and want to determine the fourth one. Suppose we want to perceive the effect of some unknown cause, and want to compute that cause, then the Bayes' rule becomes: P(cause|effect) = P(effect|cause)P(cause) / P(effect).