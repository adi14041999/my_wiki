# Multinomial Distribution

The multinomial distribution is a generalization of the binomial distribution to multiple categories or outcomes. It describes the probability distribution of counts across several mutually exclusive and exhaustive categories.

A random vector $\mathbf{X} = (X_1, X_2, \ldots, X_k)$ follows a **multinomial distribution** with parameters $n$ and $\mathbf{p} = (p_1, p_2, \ldots, p_k)$, denoted as $\mathbf{X} \sim \text{Multinomial}(n, \mathbf{p})$, if its joint probability mass function is:

$$P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}$$

where:

- $n$ is the total number of trials

- $x_i$ is the number of outcomes in category $i$ (with $x_i \geq 0$)

- $p_i$ is the probability of category $i$ (with $0 \leq p_i \leq 1$)

- The constraint $\sum_{i=1}^k x_i = n$ must hold

- The constraint $\sum_{i=1}^k p_i = 1$ must hold

The multinomial distribution models:

- **$n$ independent trials** where each trial results in exactly one of $k$ possible outcomes

- **Fixed probabilities** $p_1, p_2, \ldots, p_k$ for each outcome

- **Counts** $X_1, X_2, \ldots, X_k$ representing how many times each outcome occurred

When $k = 2$, the multinomial distribution reduces to the binomial distribution:

- $X_1 \sim \text{Binomial}(n, p_1)$

- $X_2 = n - X_1$ (since $X_1 + X_2 = n$)

- $p_2 = 1 - p_1$

## Marginal Distribution

Let's take a look at the marginal distribution of $X_i$, which is the $i$th component of $\mathbf{X}$. Were we to blindly apply the definition, we would have to sum the joint PMF over all components of $\mathbf{X}$ other than $X_i$. The prospect of $k-1$ summations is an unpleasant one, to say the least.

Fortunately, we can avoid tedious calculations if we instead use the story of the Multinomial distribution: $X_i$ is the number of objects in category $i$, where each of the $n$ objects independently belongs to category $i$ with probability $p_i$. Define success as landing in category $i$. Then we just have $n$ independent Bernoulli trials, so the marginal distribution of $X_i$ is $\text{Bin}(n, p_i)$.

**The marginals of a Multinomial are Binomial.** Specifically, if $\mathbf{X} \sim \text{Mult}_k(n, \mathbf{p})$, then $X_i \sim \text{Bin}(n, p_i)$.