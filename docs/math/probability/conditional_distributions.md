# Conditional Distributions

Conditional distributions describe the probability distribution of one random variable given that another random variable takes on a specific value. They are fundamental to understanding how random variables relate to each other.

## Discrete Case

For two discrete random variables $X$ and $Y$, the **conditional probability mass function** of $X$ given $Y = y$ is defined as:

$$p_{X|Y}(x|y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$$

where $p_Y(y) > 0$.

## Continuous Case

For two continuous random variables $X$ and $Y$, the **conditional probability density function** of $X$ given $Y = y$ is defined as:

$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$$

where $f_Y(y) > 0$.

## Interpretation

The conditional distribution $p_{X|Y}(x|y)$ or $f_{X|Y}(x|y)$ represents:

- The probability (or probability density) that $X = x$ given that we know $Y = y$

- How the distribution of $X$ changes when we condition on different values of $Y$

The conditional distribution formulas work analogously to the familiar conditional probability formula:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

In the discrete case, we have:

$$p_{X|Y}(x|y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$$

This is exactly analogous to:

$$P(X = x | Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}$$

The joint PMF $p_{X,Y}(x,y)$ represents $P(X = x, Y = y)$, and the marginal PMF $p_Y(y)$ represents $P(Y = y)$.

For continuous variables, the PDF values themselves are not probabilities, but the **ratio** of PDFs works the same way:

$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$$

Think of it this way:

1. **Joint PDF** $f_{X,Y}(x,y)$ tells us how "dense" the probability is at point $(x,y)$
2. **Marginal PDF** $f_Y(y)$ tells us how "dense" the probability is along the entire line $Y = y$

3. **Conditional PDF** $f_{X|Y}(x|y)$ tells us how "dense" the probability is at $X = x$ when we're restricted to $Y = y$

**Example:** Consider a chicken that lays eggs according to a Poisson process. Let $N$ be the total number of eggs laid, where $N \sim \text{Poisson}(\lambda)$. Each egg has a probability $p$ of hatching. Let $X$ be the number of eggs that hatch, so $X|N \sim \text{Binomial}(N,p)$. Let $Y$ be the number of eggs that don't hatch, so $X + Y = N$. Find the joint distribution of $X$ and $Y$, and determine if they are independent.

We are given: 
- $N \sim \text{Poisson}(\lambda)$: Total eggs laid

- $X|N \sim \text{Binomial}(N,p)$: Eggs that hatch, given $N$ eggs

- $Y = N - X$: Eggs that don't hatch

- We need to find $p_{X,Y}(x,y)$

Let's find the joint distribution. Since $X + Y = N$, we can write:

$$p_{X,Y}(x,y) = P(X = x, Y = y) = P(X = x, N = x + y)$$

Using the law of total probability and the conditional distribution:

$$p_{X,Y}(x,y) = P(X = x|N = x + y) \cdot P(N = x + y)$$

Given $X|N \sim \text{Binomial}(N,p)$:

$$P(X = x|N = x + y) = \binom{x + y}{x} p^x (1-p)^y$$

And $N \sim \text{Poisson}(\lambda)$:

$$P(N = x + y) = \frac{e^{-\lambda} \lambda^{x + y}}{(x + y)!}$$

Therefore:

$$p_{X,Y}(x,y) = \binom{x + y}{x} p^x (1-p)^y \cdot \frac{e^{-\lambda} \lambda^{x + y}}{(x + y)!}$$

Simplifying:

$$p_{X,Y}(x,y) = \frac{(x + y)!}{x! y!} p^x (1-p)^y \cdot \frac{e^{-\lambda} \lambda^{x + y}}{(x + y)!}$$

$$p_{X,Y}(x,y) = \frac{e^{-\lambda} \lambda^{x + y} p^x (1-p)^y}{x! y!}$$

$$p_{X,Y}(x,y) = \frac{e^{-\lambda} (\lambda p)^x (\lambda(1-p))^y}{x! y!}$$

Let's find the marginal distributions.

$$p_X(x) = \sum_{y=0}^{\infty} p_{X,Y}(x,y) = \sum_{y=0}^{\infty} \frac{e^{-\lambda} (\lambda p)^x (\lambda(1-p))^y}{x! y!}$$

$$p_X(x) = \frac{e^{-\lambda} (\lambda p)^x}{x!} \sum_{y=0}^{\infty} \frac{(\lambda(1-p))^y}{y!}$$

$$p_X(x) = \frac{e^{-\lambda} (\lambda p)^x}{x!} e^{\lambda(1-p)} = \frac{e^{-\lambda p} (\lambda p)^x}{x!}$$

This shows $X \sim \text{Poisson}(\lambda p)$.

Similarly:

$$p_Y(y) = \frac{e^{-\lambda(1-p)} (\lambda(1-p))^y}{y!}$$

So $Y \sim \text{Poisson}(\lambda(1-p))$.

If $X$ and $Y$ were independent, then:

$$p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y)$$

Let's check:

$$p_X(x) \cdot p_Y(y) = \frac{e^{-\lambda p} (\lambda p)^x}{x!} \cdot \frac{e^{-\lambda(1-p)} (\lambda(1-p))^y}{y!}$$

$$p_X(x) \cdot p_Y(y) = \frac{e^{-\lambda p - \lambda(1-p)} (\lambda p)^x (\lambda(1-p))^y}{x! y!}$$

$$p_X(x) \cdot p_Y(y) = \frac{e^{-\lambda} (\lambda p)^x (\lambda(1-p))^y}{x! y!}$$

This equals $p_{X,Y}(x,y)$, so $X$ and $Y$ are **independent**!

The joint distribution is:

$$p_{X,Y}(x,y) = \frac{e^{-\lambda} (\lambda p)^x (\lambda(1-p))^y}{x! y!}$$

And $X$ and $Y$ are independent random variables, each following Poisson distributions with parameters $\lambda p$ and $\lambda(1-p)$ respectively.

**Note:** It's crucial to distinguish between $X|N \sim \text{Binomial}(N,p)$ (conditional distribution where $N$ is random) and $X \sim \text{Binomial}(N,p)$ (unconditional distribution where $N$ would be fixed). The conditional notation is essential here since $N$ is a random variable.