# Story Proofs and Axioms of Probability

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

## Formal Definition of Probability

Let $S$ be a **sample space** (the set of all possible outcomes of an experiment). An **event** $A$ is a subset of $S$ (i.e., $A \subseteq S$).

A **probability function** $P$ is a function that takes an event $A$ as input and returns a real number $P(A)$ as output, where $P(A) \in [0, 1]$ for any event $A \subseteq S$.

Philosophically, there are different interpretations of Probability, arguments even. But mathematically speaking, the formal definition of Probability along with the axioms is well defined without any ambiguity. And this definition coupled with the axioms constitute a foundation for this field where every theorem or result can be derived from.

## Axioms of Probability

The probability function $P$ must satisfy the following axioms:

**Axiom 1 (Non-negativity)**: For any event $A \subseteq S$,

$$P(A) \geq 0$$

**Axiom 2 (Normalization)**: For the entire sample space $S$,

$$P(S) = 1$$

**Axiom 3 (Additivity)**: For any collection of mutually exclusive events $A_1, A_2, A_3, \ldots$ (i.e., $A_i \cap A_j = \emptyset$ for $i \neq j$),

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

From these axioms, we can derive several important properties:

1. **Probability of the empty set**: $P(\emptyset) = 0$
2. **Complement rule**: $P(A^c) = 1 - P(A)$
3. **Monotonicity**: If $A \subseteq B$, then $P(A) \leq P(B)$
4. **Union rule**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

**Inclusion-Exclusion Principle (3 events)**: 

$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

**Inclusion-Exclusion Principle (n events)**: For events $A_1, A_2, \ldots, A_n$,

$$P\left(\bigcup_{i=1}^n A_i\right) = \sum_{i=1}^n P(A_i) - \sum_{1 \leq i < j \leq n} P(A_i \cap A_j) + \sum_{1 \leq i < j < k \leq n} P(A_i \cap A_j \cap A_k) - \cdots + (-1)^{n+1} P(A_1 \cap A_2 \cap \cdots \cap A_n)$$

These axioms provide the mathematical foundation for probability theory and ensure that probability behaves in an intuitive and consistent way.

## The Birthday Problem

The Birthday Problem is a classic probability puzzle that asks: **What is the probability that in a group of $n$ people, at least two people share the same birthday?**

This seemingly simple question leads to a surprising result that challenges our intuition about probability.

**Given**: A group of $n$ people chosen randomly from a population where birthdays are uniformly distributed across 365 days (ignoring leap years).

**Find**: The probability that at least two people in the group share the same birthday.

The result is counterintuitive: **In a group of just 23 people, there is approximately a 50% chance that at least two people share the same birthday!**

This seems impossible at first glance- with 365 possible birthdays and only 23 people, how can there be a 50% chance of a match?

We solve this using the **complement rule**: Instead of calculating the probability of at least one match directly, we calculate the probability of **no matches** and subtract from 1.

Let $A$ be the event "at least two people share a birthday". $P(A) = 1 - P(A^c)$, where $A^c$ is "no two people share a birthday". First person: Can have any birthday (365/365 = 1). Second person: Must have a different birthday (364/365). Third person: Must have a different birthday from the first two (363/365). And so on...

**General formula**:

$$P(A^c) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \cdots \times \frac{365-n+1}{365}$$

This can be written more compactly as:

$$P(A^c) = \frac{365!}{(365-n)! \times 365^n}$$

$$P(A) = 1 - \frac{365!}{(365-n)! \times 365^n}$$

Let's calculate some key values:

| Number of People ($n$) | Probability of at least one match |
|------------------------|-----------------------------------|
| 10                    | 11.7%                            |
| 15                    | 25.3%                            |
| 20                    | 41.1%                            |
| **23**                | **50.7%**                        |
| 30                    | 70.6%                            |
| 40                    | 89.1%                            |
| 50                    | 97.0%                            |
| 60                    | 99.4%                            |

The result feels wrong because we're thinking about **individual comparisons** rather than **all possible pairs**.

**Number of pairs**: In a group of $n$ people, there are $\binom{n}{2} = \frac{n(n-1)}{2}$ possible pairs. With 23 people: $\binom{23}{2} = 253$ pairs. With 30 people: $\binom{30}{2} = 435$ pairs. With 50 people: $\binom{50}{2} = 1,225$ pairs.

## De Montmort's Problem

De Montmort's Problem is a classic probability puzzle that asks: **What is the probability that when $n$ cards are dealt from a deck, at least one card appears in a position that matches its value?**

This problem is also known as the **matching problem** or the **coincidence problem** and was first studied by Pierre Raymond de Montmort in the early 18th century.

**Given**: A deck of $n$ cards numbered from 1 to $n$. The cards are shuffled and dealt face up in a row.

**Find**: The probability that at least one card appears in the $k$-th position where the card's value is $k$.

**Example**: For $n = 4$, we have cards [1, 2, 3, 4]. A deal of [2, 1, 4, 3] has no matches, but [1, 4, 2, 3] has a match (card 1 in position 1).

We solve this using the **inclusion-exclusion principle**. Let $A_i$ be the event that card $i$ appears in position $i$.

**Key insight**: We want $P(A_1 \cup A_2 \cup \cdots \cup A_n)$, which we can calculate using inclusion-exclusion.

**Step-by-Step Solution**

1. **Individual probabilities**: $P(A_i) = \frac{1}{n}$ for each card
2. **Pairwise intersections**: $P(A_i \cap A_j) = \frac{1}{n(n-1)}$ for $i \neq j$
3. **Triple intersections**: $P(A_i \cap A_j \cap A_k) = \frac{1}{n(n-1)(n-2)}$ for distinct $i, j, k$
4. and so on...

**Using inclusion-exclusion**:

$$P(A_1 \cup A_2 \cup \cdots \cup A_n) = \sum_{i=1}^n P(A_i) - \sum_{1 \leq i < j \leq n} P(A_i \cap A_j) + \sum_{1 \leq i < j < k \leq n} P(A_i \cap A_j \cap A_k) - \cdots$$

**Calculating the terms**:
**First term**: $\sum_{i=1}^n P(A_i) = n \cdot \frac{1}{n} = 1$.
**Second term**: $\sum_{1 \leq i < j \leq n} P(A_i \cap A_j) = \binom{n}{2} \cdot \frac{1}{n(n-1)} = \frac{n(n-1)}{2} \cdot \frac{1}{n(n-1)} = \frac{1}{2}$.
**Third term**: $\sum_{1 \leq i < j < k \leq n} P(A_i \cap A_j \cap A_k) = \binom{n}{3} \cdot \frac{1}{n(n-1)(n-2)} = \frac{1}{3!} = \frac{1}{6}$.
And so on...

**General pattern**: The $k$-th term is $\frac{1}{k!}$

**Final result**:

$$P(\text{at least one match}) = 1 - \frac{1}{2!} + \frac{1}{3!} - \frac{1}{4!} + \cdots + (-1)^{n+1} \frac{1}{n!}$$

As $n$ approaches infinity, the probability approaches:

$$\lim_{n \to \infty} P(\text{at least one match}) = 1 - \frac{1}{e} \approx 0.632$$

This means that even with infinitely many cards, there's still only about a 63.2% chance that at least one card appears in its "correct" position!