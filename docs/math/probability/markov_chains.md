# Markov Chains

## Introduction

To see where the Markov chain comes from, start by considering an **i.i.d. sequence** of random variables $X_0, X_1, \ldots, X_n, \ldots$ where we think of $n$ as time. An i.i.d. sequence has **no dependence** between any of the random variables- each $X_n$ is independent of all previous values. A Markov chain is a sequence
of r.v.s that exhibits one-step dependence.

### State Space and Time

Markov chains "live" in both **space** and **time**.

- **State Space**: The set of all possible values that the random variables $X_n$ can take

- **Time**: The index $n$ represents the evolution of some process over time

**1. State Space Type:**

- **Discrete State Space**: States take values from a countable set (finite or infinite)

- **Continuous State Space**: States take values from a continuous set (e.g., real numbers)

**2. Time Type:**

- **Discrete Time**: Process evolves at discrete time steps ($n = 0, 1, 2, \ldots$)

- **Continuous Time**: Process evolves continuously over time ($t \geq 0$)

### Definition (Markov chain)

A sequence of random variables $X_0, X_1, X_2, \ldots$ taking values in the state space $\{1, 2, \ldots, M\}$ is called a Markov chain if for all $n \geq 0$,

$$P(X_{n+1} = j | X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j | X_n = i)$$

The quantity $P(X_{n+1} = j | X_n = i)$ is called the **transition probability** from state $i$ to state $j$.

If we think of time $n$ as the present, times before $n$ as the past, and times after $n$ as the future, the Markov property says that given the present, the past and future are conditionally independent. The Markov property greatly simplifies computations of conditional probability: instead of having to condition on the entire past, we only need to condition on the most recent value.

## Transition matrix

**Definition (Transition matrix):** Let $X_0, X_1, X_2, \ldots$ be a Markov chain with state space $\{1, 2, \ldots, M\}$, and let $q_{ij} = P(X_{n+1} = j | X_n = i)$ be the transition probability from state $i$ to state $j$. The $M \times M$ matrix $Q = (q_{ij})$ is called the **transition matrix** of the chain.

Note that $Q$ is a nonnegative matrix in which each row sums to 1. This is because, starting from any state $i$, the events "move to 1", "move to 2", $\ldots$, "move to $M$" are disjoint, and their union has probability 1 because the chain has to go somewhere.

**Example:** Suppose that on any given day, the weather can either be rainy or sunny. If today is rainy, then tomorrow will be rainy with probability $1/3$ and sunny with probability $2/3$. If today is sunny, then tomorrow will be rainy with probability $1/2$ and sunny with probability $1/2$. Letting $X_n$ be the weather on day $n$, $X_0, X_1, X_2, \ldots$ is a Markov chain on the state space $\{R, S\}$, where $R$ stands for rainy and $S$ for sunny. We know that the Markov property is satisfied because, from the description of the process, only today's weather matters for predicting tomorrow's.

The transition matrix of the chain is

$$\begin{array}{c|cc}
 & R & S \\
\hline
R & 1/3 & 2/3 \\
S & 1/2 & 1/2 \\
\end{array}$$

The transition probabilities of a Markov chain can also be represented with a diagram. Each state is represented by a circle, and the arrows indicate the possible one-step transitions; we can imagine a particle wandering around from state to state, randomly choosing which arrow to follow. Next to the arrows we write the corresponding transition probabilities.

![image](tr_matrix.png)

Let's trace through a specific realization of the rainy-sunny Markov chain. Suppose we start with $X_0 = R$ (rainy on day 0) and simulate the next 5 days:

**Day-by-day evolution:**

- $X_0 = R$ (start rainy)

- $X_1 = S$ (transition: R→S with probability 2/3)

- $X_2 = R$ (transition: S→R with probability 1/2)  

- $X_3 = S$ (transition: R→S with probability 2/3)

- $X_4 = S$ (transition: S→S with probability 1/2)

- $X_5 = R$ (transition: S→R with probability 1/2)

**Key observations:**

- Each transition depends only on the current state (Markov property)

- This is just one possible realization - different runs would produce different sequences

- The probabilities at each step are determined by the transition matrix