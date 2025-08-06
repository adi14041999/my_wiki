# Story Proofs, Axioms of Probability

## Story Proofs

Story proofs are a powerful technique in combinatorics where we prove identities by interpreting both sides of an equation as counting the same thing in different ways. This is also known as **proof by interpretation** or **bijective proof**.

**Example:** $\binom{n}{k} = \binom{n}{n-k}$

**Identity**: $\binom{n}{k} = \binom{n}{n-k}$

**Story Proof**: Think of choosing $k$ people from a group of $n$ people to be on a committee. The left side $\binom{n}{k}$ counts the number of ways to choose $k$ people for the committee. The right side $\binom{n}{n-k}$ counts the number of ways to choose $n-k$ people to be **left out** of the committee. But choosing $k$ people for the committee is exactly the same as choosing $n-k$ people to leave out! Therefore, both sides count the same thing.

**Key insight**: Every choice of $k$ people corresponds uniquely to a choice of $n-k$ people (the complement), and vice versa.

**Example:** $n \cdot \binom{n-1}{k-1} = k \cdot \binom{n}{k}$

**Identity**: $n \cdot \binom{n-1}{k-1} = k \cdot \binom{n}{k}$

**Story Proof**: Think of choosing $k$ people from $n$ people, with one of them designated as President.

**Left side**: $n \cdot \binom{n-1}{k-1}$
First, choose who will be President ($n$ choices). Then, from the remaining $n-1$ people, choose $k-1$ more people to complete the committee. Total: $n \cdot \binom{n-1}{k-1}$

**Right side**: $k \cdot \binom{n}{k}$
First, choose any $k$ people from $n$ people ($\binom{n}{k}$ ways). Then, from those $k$ people, choose one to be President ($k$ choices). Total: $k \cdot \binom{n}{k}$

**Example:** Vandermonde Identity

**Identity**: $\sum_{k=0}^n \binom{m}{k} \binom{p}{n-k} = \binom{m+p}{n}$

**Story Proof**: Think of choosing $n$ people from a group of $m$ men and $p$ women to form a committee.

**Left side**: $\sum_{k=0}^n \binom{m}{k} \binom{p}{n-k}$
For each $k$ from $0$ to $n$, choose $k$ men from $m$ men ($\binom{m}{k}$ ways). Then, choose $n-k$ women from $p$ women ($\binom{p}{n-k}$ ways). Total for this $k$: $\binom{m}{k} \binom{p}{n-k}$. Sum over all possible values of $k$: $\sum_{k=0}^n \binom{m}{k} \binom{p}{n-k}$

**Right side**: $\binom{m+p}{n}$
Simply choose $n$ people from the total group of $m+p$ people

**Key insight**: The left side partitions the counting by gender composition, while the right side ignores gender entirely. Both approaches must give the same result.

Story proofs are powerful because they:

1. **Provide intuition** - You understand why the identity is true

2. **Are memorable** - The story helps you remember the result

3. **Avoid algebra** - No need for complex manipulations

4. **Generalize well** - The same story often works for related problems

**Key principle**: If two expressions count the same thing, they must be equal.

## Non-naive definition of Probability and Axioms of Probability

### Formal Definition of Probability

Let $S$ be a **sample space** (the set of all possible outcomes of an experiment). An **event** $A$ is a subset of $S$ (i.e., $A \subseteq S$).

A **probability function** $P$ is a function that takes an event $A$ as input and returns a real number $P(A)$ as output, where $P(A) \in [0, 1]$ for any event $A \subseteq S$

### Axioms of Probability

The probability function $P$ must satisfy the following axioms:

**Axiom 1 (Non-negativity)**: For any event $A \subseteq S$,

$$P(A) \geq 0$$

**Axiom 2 (Normalization)**: For the entire sample space $S$,

$$P(S) = 1$$

**Axiom 3 (Additivity)**: For any collection of mutually exclusive events $A_1, A_2, A_3, \ldots$ (i.e., $A_i \cap A_j = \emptyset$ for $i \neq j$),

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

### Key Properties Derived from Axioms

From these axioms, we can derive several important properties:

1. **Probability of the empty set**: $P(\emptyset) = 0$
2. **Complement rule**: $P(A^c) = 1 - P(A)$
3. **Monotonicity**: If $A \subseteq B$, then $P(A) \leq P(B)$
4. **Union rule**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

These axioms provide the mathematical foundation for probability theory and ensure that probability behaves in an intuitive and consistent way.
