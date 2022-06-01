### Probabilistic Reasoning in Artificial Intelligence

### Uncertainty

To represetn uncertain, where we are not sure about the predicates, we need uncertain reasoning or probabilisitic reasoning.

#### Causes of Uncertainty
- Information occurred from unreliable sources
- Experimental Errors
- Equipment fault
- Temperature variation
- Climate change

#### Probabilistic Reasoning

Probabilistic reasoning is a way of knowledge representation where we can apply the concept of probability to indicate the uncertainty in knowledge. In probabilistic reasoning, we combine probability theory with logic to handle the uncertainty.

We use probability in probabilisitc reasoning because it provides a way to handle the uncertainty that is the result of someone's laziness and ignoreance.

In the real world, there are lots of scenarios, where the certainty of something is not confirmed, such as "It will rain today", "Behavior of someone for some situations", "A match between two teams or two players". These are probale sentences for which we can assume that it will happen but not sure about it, so here we use probabilistic reasoning.

As probabilisitc reasoning uses probability and related terms, so before understanindg probabilistic reasonsing, let's understand some common terms:

- **Probability**: 

    Probability can be defined as a chance that an uncertain event will occur. It is the numerical measure of the likelihood that an event will occur. The value of probability always remain between 0 and 1 that represent ideal uncertainties.
    - 0 <= P(A) <= 1, where P(A) is the probability of an event A
    - P(A) = 0, indicates total uncertainty in an event A
    - P(A) = 1, indicates total certainty in an event A
    
    We can find the probability of an uncertain event by using the formula: Probability of occurrence = Number of desired outcomes / Total number of outcomes.
    
    - P(¬A) = probability of not happening event.
    - P(¬A) + P(A) = 1

- **Event**: Each possible outcome of a variable is called an event.
- **Sample space**: The collection of all possible events is called sample space.
- **Random variables**: Random variables are used to represent the events and objects in the real world.
- **Prior probability**: The prior probability of an event is probability computed before observing new information.
- **Posterior probability**: The probabilty that is calculated after all evidence or information has taken into account. It is a combination of prior probability and new information.

#### Conditional Probability

Conditional probability is a probability of occuring an event when another event has already happened. Let's suppose we want to calculate the event A when evetn B has already occurred, "the probability of A under the conditions of B", it can be written as: P(A|B) = P(A ∩ B) / P(B), where P(A ∩ B) = joint probability of A and B, P(B) = marginal probability of B.

If the probability of A is given and we need to find the probability of B, then it will be given as: P(B|A) = P(A ∩ B) / P(A).

It can be explained by using the below Venn diagram, where B is occurred event, so sample space will be reduced to set B, and now we can only calculate event A when event B is already occurred by dividing the probability of P(A ∩ B) by P(B).