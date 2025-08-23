# Joint Distributions

A **joint distribution** describes the probability distribution of two or more random variables simultaneously. It captures not only the individual behavior of each random variable but also how they relate to each other.

For two discrete random variables $X$ and $Y$, the **joint probability mass function** (joint PMF) is defined as:

$$p_{X,Y}(x,y) = P(X = x, Y = y)$$

For continuous random variables, the **joint probability density function** (joint PDF) satisfies:

$$P((X,Y) \in A) = \iint_A f_{X,Y}(x,y) \, dx \, dy$$

**Example:** Let's consider two Bernoulli random variables $X$ and $Y$ with parameters $p$ and $q$ respectively:

- $X \sim \text{Bernoulli}(p)$ where $P(X = 1) = p$ and $P(X = 0) = 1-p$

- $Y \sim \text{Bernoulli}(q)$ where $P(Y = 1) = q$ and $P(Y = 0) = 1-q$

The joint PMF $p_{X,Y}(x,y)$ gives us the probability of each possible combination:

| $X \backslash Y$ | $0$ | $1$ |
|------------------|-----|-----|
| $0$ | $p_{X,Y}(0,0)$ | $p_{X,Y}(0,1)$ |
| $1$ | $p_{X,Y}(1,0)$ | $p_{X,Y}(1,1)$ |

**If $X$ and $Y$ are independent**, then:

$$p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y)$$

This means:

- $p_{X,Y}(0,0) = (1-p)(1-q)$

- $p_{X,Y}(0,1) = (1-p)q$

- $p_{X,Y}(1,0) = p(1-q)$

- $p_{X,Y}(1,1) = pq$

**If $X$ and $Y$ are dependent**, the joint PMF cannot be factored this way, and we need additional information to specify the relationship between them.

From the joint distribution, we can recover the individual (marginal) distributions:

$$p_X(x) = \sum_y p_{X,Y}(x,y)$$

$$p_Y(y) = \sum_x p_{X,Y}(x,y)$$

For our Bernoulli example:

- $p_X(0) = p_{X,Y}(0,0) + p_{X,Y}(0,1) = 1-p$

- $p_X(1) = p_{X,Y}(1,0) + p_{X,Y}(1,1) = p$

- $p_Y(0) = p_{X,Y}(0,0) + p_{X,Y}(1,0) = 1-q$

- $p_Y(1) = p_{X,Y}(0,1) + p_{X,Y}(1,1) = q$