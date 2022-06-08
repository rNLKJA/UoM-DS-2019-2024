### What is Unification?
- Unification is a process of making two different logical atomic expressions identical by finding a substitution. Unification depends on the substitution process.
- It takes two literals as input and makes them identical using substitution.
- Let Ψ1 and Ψ2 be two atomic sentences and 𝜎 be a unifier such that, Ψ1𝜎 = Ψ2𝜎, then it can be expressed as UNIFY(Ψ1, Ψ2).

- The UNIFY algorithm is used for unification, which takes two atomic sentences and returns a unifier for those sentences (If any exist).
- Unification is a key component of all first-order inference algorithms.
- It returns fail if the expressions do not match with each other.
- The substitution variables are called Most General Unifier or MGU.

- Substitute x with a, and y with f(z) in the first expression, and it will be represented as a/x and f(z)/y.
- With both the substitutions, the first expression will be identical to the second expression and the substitution set will be: [a/x, f(z)/y].

### Conditions for Unification:
Following are some basic conditions for unification:
- Predicate symbol must be same, atoms or expression with different predicate symbol can never be unified.
- Number of Arguments in both expressions must be identical.
- Unification will fail if there are two similar variables present in the same expression.

### Unification Algorithm:
```
Algorithm: Unify(Ψ1, Ψ2)

Step. 1: If Ψ1 or Ψ2 is a variable or constant, then:
	a) If Ψ1 or Ψ2 are identical, then return NIL. 
	b) Else if Ψ1is a variable, 
		a. then if Ψ1 occurs in Ψ2, then return FAILURE
		b. Else return { (Ψ2/ Ψ1)}.
	c) Else if Ψ2 is a variable, 
		a. If Ψ2 occurs in Ψ1 then return FAILURE,
		b. Else return {( Ψ1/ Ψ2)}. 
	d) Else return FAILURE. 
Step.2: If the initial Predicate symbol in Ψ1 and Ψ2 are not same, then return FAILURE.
Step. 3: IF Ψ1 and Ψ2 have a different number of arguments, then return FAILURE.
Step. 4: Set Substitution set(SUBST) to NIL. 
Step. 5: For i=1 to the number of elements in Ψ1. 
	a) Call Unify function with the ith element of Ψ1 and ith element of Ψ2, and put the result into S.
	b) If S = failure then returns Failure
	c) If S ≠ NIL then do,
		a. Apply S to the remainder of both L1 and L2.
		b. SUBST= APPEND(S, SUBST). 
Step.6: Return SUBST. 
```