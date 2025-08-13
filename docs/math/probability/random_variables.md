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