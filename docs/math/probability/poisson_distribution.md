# Poisson Distribution

The **Poisson distribution** is one of the most important discrete probability distributions, modeling the number of events occurring in a fixed interval of time or space when these events happen independently at a constant average rate.

This is arguably the most important discrete distribution in Statistics.

A random variable $X$ follows a **Poisson distribution** with parameter $\lambda > 0$ (denoted $X \sim \text{Poisson}(\lambda)$) if its probability mass function is:

$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!} \quad \text{for } k = 0, 1, 2, 3, \ldots$$

**Parameters**:

- **$\lambda$**: Average number of events in the interval (also called the rate parameter)

- **Support**: $X$ takes values in $\{0, 1, 2, 3, \ldots\}$ (non-negative integers)

**What does the Poisson distribution model?**

- **Rare events**: Events that occur infrequently but consistently

- **Independent occurrences**: Each event is independent of others

- **Constant rate**: Events happen at a steady average rate

- **Fixed interval**: We count events in a specific time period or region

**Examples**:

- Number of phone calls received in an hour

- Number of defects in a square meter of fabric

- Number of accidents at an intersection per day

- Number of customers arriving at a store in 10 minutes

The Poisson distribution can be derived as a limiting case of the binomial distribution.

**Setup**: Consider $n$ independent Bernoulli trials, each with success probability $p = \frac{\lambda}{n}$

**Binomial PMF**: $P(X = k) = \binom{n}{k} p^k(1-p)^{n-k}$

**Take the limit**: As $n \to \infty$ while keeping $np = \lambda$ constant

**Result**: The binomial PMF converges to the Poisson PMF

**Mathematical details**:

$$\lim_{n \to \infty} \binom{n}{k} \left(\frac{\lambda}{n}\right)^k \left(1-\frac{\lambda}{n}\right)^{n-k} = \frac{e^{-\lambda} \lambda^k}{k!}$$

**Proof**: Let's prove this step by step.

Start with the binomial PMF:

$$P(X = k) = \binom{n}{k} \left(\frac{\lambda}{n}\right)^k \left(1-\frac{\lambda}{n}\right)^{n-k}$$

Expand the binomial coefficient:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!} = \frac{n(n-1)(n-2)\cdots(n-k+1)}{k!}$$

Substitute and rearrange:

$$P(X = k) = \frac{n(n-1)(n-2)\cdots(n-k+1)}{k!} \cdot \left(\frac{\lambda}{n}\right)^k \cdot \left(1-\frac{\lambda}{n}\right)^{n-k}$$

$$= \frac{\lambda^k}{k!} \cdot \frac{n(n-1)(n-2)\cdots(n-k+1)}{n^k} \cdot \left(1-\frac{\lambda}{n}\right)^{n-k}$$

Analyze this factor as $n \to \infty$:

$$\frac{n(n-1)(n-2)\cdots(n-k+1)}{n^k} = \frac{n}{n} \cdot \frac{n-1}{n} \cdot \frac{n-2}{n} \cdots \frac{n-k+1}{n}$$

$$= 1 \cdot \left(1-\frac{1}{n}\right) \cdot \left(1-\frac{2}{n}\right) \cdots \left(1-\frac{k-1}{n}\right)$$

As $n \to \infty$, each term $\left(1-\frac{j}{n}\right) \to 1$ for $j = 1, 2, \ldots, k-1$.

Therefore:

$$\lim_{n \to \infty} \frac{n(n-1)(n-2)\cdots(n-k+1)}{n^k} = 1$$

Analyze the last factor as $n \to \infty$:

$$\left(1-\frac{\lambda}{n}\right)^{n-k} = \left(1-\frac{\lambda}{n}\right)^n \cdot \left(1-\frac{\lambda}{n}\right)^{-k}$$

As $n \to \infty$:

- $\left(1-\frac{\lambda}{n}\right)^n \to e^{-\lambda}$ (this is the definition of $e$)

- $\left(1-\frac{\lambda}{n}\right)^{-k} \to 1^{-k} = 1$

Combine all limits:

$$\lim_{n \to \infty} P(X = k) = \frac{\lambda^k}{k!} \cdot 1 \cdot e^{-\lambda} = \frac{e^{-\lambda} \lambda^k}{k!}$$

**Intuition**: When we have many rare events (large $n$, small $p$), the binomial distribution becomes approximately Poisson.

**Example:** A student types a 1000-word essay and makes an average of 2 typos per 1000 words. What's the probability of making exactly 3 typos?

**Solution**:

- **Rate**: $\lambda = 2$ typos per 1000 words

- **Random variable**: $X \sim \text{Poisson}(2)$

- **Probability**: $P(X = 3) = \frac{e^{-2} \cdot 2^3}{3!} = \frac{e^{-2} \cdot 8}{6} \approx 0.180$

**Example:** A coffee shop serves an average of 5 customers every 15 minutes. What's the probability of serving at least 2 customers in the next 15 minutes?

**Solution**:

- **Rate**: $\lambda = 5$ customers per 15 minutes

- **Random variable**: $X \sim \text{Poisson}(5)$

- **Probability**: $P(X \geq 2) = 1 - P(X = 0) - P(X = 1)$

First, calculate individual probabilities:

- $P(X = 0) = \frac{e^{-5} \cdot 5^0}{0!} = e^{-5} \approx 0.0067$

- $P(X = 1) = \frac{e^{-5} \cdot 5^1}{1!} = 5e^{-5} \approx 0.0337$

Therefore:

- $P(X \geq 2) = 1 - 0.0067 - 0.0337 = 0.9596$

Note: 

- **$\lambda$ is the average rate** of events occurring

- **Expectation represents the long-run average** of the random variable

- **In a Poisson process**, we expect $\lambda$ events per unit time or space on average

Let's see if  $E[X] = \lambda$ is the case.

Start with the definition of expectation for a discrete random variable:

$$E[X] = \sum_{k=0}^{\infty} k \cdot P(X = k) = \sum_{k=0}^{\infty} k \cdot \frac{e^{-\lambda} \lambda^k}{k!}$$

Notice that the first term ($k = 0$) is 0, so we can start from $k = 1$:

$$E[X] = \sum_{k=1}^{\infty} k \cdot \frac{e^{-\lambda} \lambda^k}{k!}$$

Cancel the $k$:

$$E[X] = \sum_{k=1}^{\infty} \frac{e^{-\lambda} \lambda^k}{(k-1)!}$$

Factor out $e^{-\lambda}$ and $\lambda$:

$$E[X] = e^{-\lambda} \lambda \sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!}$$

Make a change of variable: let $j = k-1$. Then $k = j+1$ and when $k = 1$, $j = 0$:

$$E[X] = e^{-\lambda} \lambda \sum_{j=0}^{\infty} \frac{\lambda^j}{j!}$$

Recognize the series as the Taylor series for $e^{\lambda}$:

$$\sum_{j=0}^{\infty} \frac{\lambda^j}{j!} = e^{\lambda}$$

Substitute and simplify:

$$E[X] = e^{-\lambda} \lambda \cdot e^{\lambda} = \lambda$$

**Final result**: $E[X] = \lambda$

Let's verify that the Poisson PMF satisfies all required properties:

**Property 1: Non-negativity**

$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!} \geq 0$ for all $k \geq 0$ since:

- $e^{-\lambda} > 0$

- $\lambda^k \geq 0$ for $\lambda > 0$

- $k! > 0$ for all $k \geq 0$

**Property 2: Sum to 1**

$$\sum_{k=0}^{\infty} P(X = k) = \sum_{k=0}^{\infty} \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = e^{-\lambda} \cdot e^{\lambda} = 1$$

The key insight is that $\sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = e^{\lambda}$ (the Taylor series for $e^{\lambda}$).

**Property 3: Probability bounds**

$0 \leq P(X = k) \leq 1$ for each $k$, which follows from Properties 1 and 2.