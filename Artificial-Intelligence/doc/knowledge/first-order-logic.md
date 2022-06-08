### First-Order Logic in Artificial intelligence

In the topic of Propositional logic, we have seen that how to represent statements using propositional logic. But unfortunately, in propositional logic, we can only represent the facts, which are either true or false. PL is not sufficient to represent the complex sentences or natural language statements. The propositional logic has very limited expressive power. Consider the following sentence, which we cannot represent using PL logic.
- "Some humans are intelligent", or
- "Sachin likes cricket."

To represent the above statements, PL logic is not sufficient, so we required some more powerful logic, such as first-order logic.

### First-Order logic
- First-order logic is another way of knowledge representation in artificial intelligence. It is an extension to propositional logic.
- FOL is sufficiently expressive to represent the natural language statements in a concise way.
- First-order logic is also known as Predicate logic or First-order predicate logic. First-order logic is a powerful language that develops information about the objects in a more easy way and can also express the relationship between those objects.
- First-order logic (like natural language) does not only assume that the world contains facts like propositional logic but also assumes the following things in the world:
    - Objects: A, B, people, numbers, colors, wars, theories, squares, pits, wumpus, ......
    - Relations: It can be unary relation such as: red, round, is adjacent, or n-any relation such as: the sister of, brother of, has color, comes between
    - Function: Father of, best friend, third inning of, end of, ......
- As a natural language, first-order logic also has two main parts:
    - Syntax
    - Semantics
    
### Syntax of First-Order logic

The syntax of FOL determines which collection of symbols is a logical expression in first-order logic. The basic syntactic elements of first-order logic are symbols. We write statements in short-hand notation in FOL.

#### Basic Elements of First-order logic:

Following are the basic elements of FOL syntax:
- Constant
- Variables
- Predicates
- Function
- Connectives
- Equality
- Quantifier

#### Atomic sentences

Atomic sentences are the most basic sentences of first-order logic. These sentences are formed from a predicate symbol followed by a parenthesis with a sequence of terms.

We can represent atomic sentences as Predicate (term1, term2, ......, term n).

Example: Ravi and Ajay are brothers: => Brothers(Ravi, Ajay). Chinky is a cat: => cat (Chinky).

#### Complex Sentences

Complex sentences are made by combining atomic sentences using connectives.

First-order logic statements can be divided into two parts:
- Subject: Subject is the main part of the statement.
- Predicate: A predicate can be defined as a relation, which binds two atoms together in a statement.

Consider the statement: "x is an integer.", it consists of two parts, the first part x is the subject of the statement and second part "is an integer," is known as a predicate.

### Quantifiers in First-Order Logic
- A quantifier is a language element which generates quantification, and quantification specifies the quantity of specimen in the universe of discourse.
- These are the symbols that permit to determine or identify the range and scope of the varibale in the logical expression. There are type types of quantifier:
    - Universal quantifier, (for all, everyone, everything)
        - Universal quantifier is a symbol of logical representation, which specifies that the statement within its range is true for everything or every instance of a particular thing.
        - The Universal quantifier is represented by a symbol ∀, which resembles an inverted A.
    - Existential quantifier, (for some, at least one) 
        - Existential quantifiers are the type of quantifiers, which express that the statement within its scope is true for at least one instance of something.
        - It is denoted by the logical operator ∃, which resembles as inverted E. When it is used with a predicate variable then it is called as an existential quantifier.
        
#### Free and Bound Variables

The quantifiers interact with variables which appear in a suitable way. There are two types of variables in First-order logic which are given below:
- Free variable: A variable is said to be free variable in a formula if it occurs outside the scope of the quantifier.
- Bound variable: A variable is said to be a bound variable in a formula if it occurs within the scope of the quantifier.