# Random Variables

## Definition of a Random Variable

A **random variable** is a function that maps outcomes from a sample space to real numbers. Formally, if $S$ is a sample space, then a random variable $X$ is a function:

$$X: S \rightarrow \mathbb{R}$$

**Example 1: Coin toss**

- **Sample space**: $S = \{\text{Heads}, \text{Tails}\}$

- **Random variable**: $X(\text{Heads}) = 1$, $X(\text{Tails}) = 0$

- **Interpretation**: $X$ counts the number of heads

**Example 2: Rolling a die**

- **Sample space**: $S = \{1, 2, 3, 4, 5, 6\}$

- **Random variable**: $X(\omega) = \omega$ (identity function)

- **Interpretation**: $X$ gives the face value of the die

**Example 3: Multiple coin tosses**

- **Sample space**: $S = \{\text{HH}, \text{HT}, \text{TH}, \text{TT}\}$

- **Random variable**: $X(\text{HH}) = 2$, $X(\text{HT}) = 1$, $X(\text{TH}) = 1$, $X(\text{TT}) = 0$

- **Interpretation**: $X$ counts the total number of heads

## Probability Mass Function (PMF)

Before discussing distributions, let's understand what a **Probability Mass Function (PMF)** is.

A **Probability Mass Function** is a function that gives the probability that a discrete random variable takes on a specific value. For a discrete random variable $X$, the PMF is defined as:

$$p_X(x) = P(X = x)$$

Key Properties of PMF:

1. **Non-negativity**: $p_X(x) \geq 0$ for all possible values $x$

2. **Sum to 1**: $\sum_x p_X(x) = 1$ (sum over all possible values)

3. **Probability interpretation**: $0 \leq p_X(x) \leq 1$ for each $x$

## Cumulative Distribution Function (CDF)

Another important function for describing random variables is the **Cumulative Distribution Function (CDF)**.

A **Cumulative Distribution Function** is a function that gives the probability that a random variable takes on a value less than or equal to a given number. For a random variable $X$, the CDF is defined as:

$$F_X(x) = P(X \leq x)$$

Key Properties of CDF:

1. **Non-decreasing**: $F_X(x) \leq F_X(y)$ whenever $x \leq y$ (monotonicity)

2. **Bounded**: $0 \leq F_X(x) \leq 1$ for all $x$

3. **Limits**: $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to \infty} F_X(x) = 1$

## Distribution of a Random Variable

The **distribution** of a random variable $X$ describes how the probability mass (or density) is distributed across all possible values that $X$ can take. It tells us the complete probabilistic behavior of $X$. It can be represented in various ways, including a PMF, a cumulative distribution function (CDF), or other methods.

The distribution answers the question: "What is the probability that $X$ takes on a particular value or falls within a particular range?"

Consider the die-rolling random variable $X$:

- **Sample space**: $S = \{1, 2, 3, 4, 5, 6\}$

- **Random variable**: $X(\omega) = \omega$

- **Distribution**: $P(X = k) = \frac{1}{6}$ for $k = 1, 2, 3, 4, 5, 6$

The distribution tells us that:

- Each face is equally likely

- The probability of any specific value is $\frac{1}{6}$

- The probability of rolling an even number is $P(X \in \{2, 4, 6\}) = \frac{1}{2}$

While a PMF is a key concept for describing discrete distributions, it's not the only way to define a distribution (e.g., CDF, table, etc.). 

## Bernoulli Distribution

The **Bernoulli distribution** is the simplest discrete probability distribution, modeling a random experiment with exactly two possible outcomes: success or failure.

A random variable $X$ follows a **Bernoulli distribution** with parameter $p$ (denoted $X \sim \text{Bernoulli}(p)$) if:

- **Possible values**: $X$ takes only two values: $0$ (failure) and $1$ (success)

- **Parameter**: $p \in [0, 1]$ represents the probability of success

- **Probability mass function**:

$$P(X = 1) = p \quad \text{and} \quad P(X = 0) = 1 - p$$

$$P(X = k) = p^k(1-p)^{1-k} \quad \text{for } k \in \{0, 1\}$$

This compact formula gives:

- $P(X = 1) = p^1(1-p)^0 = p$

- $P(X = 0) = p^0(1-p)^1 = 1-p$

**Example: coin toss**

- **Success**: Heads (with probability $p = \frac{1}{2}$)

- **Failure**: Tails (with probability $1-p = \frac{1}{2}$)

- **Random variable**: $X(\text{Heads}) = 1$, $X(\text{Tails}) = 0$

## Binomial Distribution

The **binomial distribution** models the number of successes in a fixed number of independent Bernoulli trials, each with the same probability of success.

A random variable $X$ follows a **binomial distribution** with parameters $n$ and $p$ (denoted $X \sim \text{Binomial}(n, p)$) if:

- **Possible values**: $X$ takes values in $\{0, 1, 2, \ldots, n\}$

- **Parameters**: 

  - $n \in \mathbb{N}$ (number of trials)

  - $p \in [0, 1]$ (probability of success in each trial)

- **Probability mass function**:

$$P(X = k) = \binom{n}{k} p^k(1-p)^{n-k} \quad \text{for } k = 0, 1, 2, \ldots, n$$

The PMF formula $P(X = k) = \binom{n}{k} p^k(1-p)^{n-k}$ has three components:

1. **$\binom{n}{k}$**: Number of ways to choose $k$ successes from $n$ trials

2. **$p^k$**: Probability of $k$ successes

3. **$(1-p)^{n-k}$**: Probability of $n-k$ failures

**Example: Quality Control**

- **Experiment**: Test 100 products from a production line

- **Random variable**: $X$ = number of defective products

- **Distribution**: $X \sim \text{Binomial}(100, 0.02)$ (assuming 2% defect rate)

- **Probability of at most 3 defects**: $P(X \leq 3) = \sum_{k=0}^3 \binom{100}{k} (0.02)^k (0.98)^{100-k}$

Let's derive the binomial PMF formula $P(X = k) = \binom{n}{k} p^k(1-p)^{n-k}$ using combinatorial reasoning.

We want to find the probability of getting exactly $k$ successes in $n$ independent Bernoulli trials, where each trial has success probability $p$.

Since the trials are **independent**, the probability of any specific sequence of outcomes is the product of individual probabilities.

**Example**: For $n = 3$ trials with $k = 2$ successes, one possible sequence is:

- Trial 1: Success (probability $p$)

- Trial 2: Success (probability $p$)

- Trial 3: Failure (probability $1-p$)

**Probability of this specific sequence**: $p \cdot p \cdot (1-p) = p^2(1-p)^1 = p^k(1-p)^{n-k}$

The key insight is that there are **multiple sequences** that result in exactly $k$ successes and $n-k$ failures.

**Question**: How many different ways can we arrange $k$ successes and $n-k$ failures in $n$ positions?

**Answer**: $\binom{n}{k}$ , that is, the number of ways to choose $k$ positions out of $n$ for the successes

Probability of any specific sequence with $k$ successes and $n-k$ failures:

$$P(\text{specific sequence}) = p^k(1-p)^{n-k}$$

Number of different sequences with exactly $k$ successes:

$$\text{Number of sequences} = \binom{n}{k}$$

Total probability using the addition rule (sums probabilities of mutually exclusive sequences):

$$P(X = k) = \binom{n}{k} \cdot p^k(1-p)^{n-k}$$

Let's verify that the binomial PMF $p_X(k) = \binom{n}{k} p^k(1-p)^{n-k}$ satisfies all the required properties of a PMF.

Property 1: Non-negativity

We need to show that $p_X(k) \geq 0$ for all $k \in \{0, 1, 2, \ldots, n\}$.

**Proof**: 

- $\binom{n}{k} \geq 0$ (combinatorial coefficient is always non-negative)

- $p^k \geq 0$ (since $p \in [0, 1]$ and $k \geq 0$)

- $(1-p)^{n-k} \geq 0$ (since $1-p \in [0, 1]$ and $n-k \geq 0$)

Since all three factors are non-negative, their product $p_X(k) = \binom{n}{k} p^k(1-p)^{n-k} \geq 0$ ✓

Property 2: Sum to 1

We need to show that $\sum_{k=0}^n p_X(k) = 1$.

**Proof**: 

$$\sum_{k=0}^n p_X(k) = \sum_{k=0}^n \binom{n}{k} p^k(1-p)^{n-k}$$

This is exactly the **binomial expansion** of $(p + (1-p))^n$:

$$(p + (1-p))^n = \sum_{k=0}^n \binom{n}{k} p^k(1-p)^{n-k}$$

But $p + (1-p) = 1$, so:

$$(p + (1-p))^n = 1^n = 1$$

Therefore, $\sum_{k=0}^n p_X(k) = 1$ ✓

Property 3: Probability Interpretation

We need to show that $0 \leq p_X(k) \leq 1$ for each $k$.

**Proof**: 

- We already showed $p_X(k) \geq 0$ (Property 1)

- Since $\sum_{k=0}^n p_X(k) = 1$ (Property 2) and all terms are non-negative, no individual term can exceed 1

- Therefore, $0 \leq p_X(k) \leq 1$ for each $k$ ✓