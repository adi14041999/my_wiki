# Expectation

Computing Averages: two approaches

Let's explore how to compute averages using two different methods, which will help build intuition for expectation.

Method 1: Arithmetic Mean (summation divided by n)

**Formula**: $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

**Example**: Consider the numbers $2, 5, 8, 8, 8, 11, 14$

**Calculation**:

$$\bar{x} = \frac{1}{7} \sum_{i=1}^7 x_i = \frac{1}{7}(2 + 5 + 8 + 8 + 8 + 11 + 14) = \frac{56}{7} = 8$$

Method 2: Weighted Sum

**Formula**: $\bar{x} = \sum_{i=1}^k w_i x_i$ where $\sum_{i=1}^k w_i = 1$ and $k$ is the number of unique values

**Same example**: Unique values $2, 5, 8, 11, 14$ with weights based on frequency:

- $w_1 = \frac{1}{7}$ (for value 2, appears 1 time)

- $w_2 = \frac{1}{7}$ (for value 5, appears 1 time)  

- $w_3 = \frac{3}{7}$ (for value 8, appears 3 times)

- $w_4 = \frac{1}{7}$ (for value 11, appears 1 time)

- $w_5 = \frac{1}{7}$ (for value 14, appears 1 time)

**Verification**: $\frac{1}{7} + \frac{1}{7} + \frac{3}{7} + \frac{1}{7} + \frac{1}{7} = \frac{7}{7} = 1$ ✓

**Calculation**:

$$\bar{x} = \sum_{i=1}^5 w_i x_i = \frac{1}{7}(2) + \frac{1}{7}(5) + \frac{3}{7}(8) + \frac{1}{7}(11) + \frac{1}{7}(14)$$

$$= \frac{2}{7} + \frac{5}{7} + \frac{24}{7} + \frac{11}{7} + \frac{14}{7} = \frac{56}{7} = 8$$

The weighted average equals the arithmetic mean because the weights reflect the actual frequency of each value in the data.

## Definition of Expectation

Now we're ready to define the **expectation** (or **expected value**) of a discrete random variable. The key insight is that the weights in our weighted sum become the **probabilities** of each value.

For a discrete random variable $X$ with possible values $x_1, x_2, \ldots, x_k$ and probability mass function $P(X = x_i) = p_i$, the **expectation** is defined as:

$$E[X] = \sum_{i=1}^k x_i \cdot P(X = x_i) = \sum_{i=1}^k x_i \cdot p_i$$

**Why use probabilities as weights?** Because we want to assign higher weights to values that are more likely to occur.

**Example**: Consider a random variable $X$ representing the outcome of a biased die:

- $P(X = 1) = 0.1$ (10% chance)

- $P(X = 2) = 0.1$ (10% chance)

- $P(X = 3) = 0.1$ (10% chance)

- $P(X = 4) = 0.1$ (10% chance)

- $P(X = 5) = 0.1$ (10% chance)

- $P(X = 6) = 0.5$ (50% chance)

**Expectation calculation**:

$$E[X] = 1(0.1) + 2(0.1) + 3(0.1) + 4(0.1) + 5(0.1) + 6(0.5)$$

$$= 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 3.0 = 4.5$$

Expectation is a weighted average where the weights are the probabilities of each possible value. Weights are probabilities $P(X = x_i)$ that sum to 1. The expectation gives us a single number that summarizes the "center" of a random variable's distribution.

## Expectation of a Bernoulli Random Variable

Let's compute the expectation of a **Bernoulli random variable** $X \sim \text{Bernoulli}(p)$.

A Bernoulli random variable $X$ takes only two values:

- $X = 1$ with probability $p$ (success)

- $X = 0$ with probability $1-p$ (failure)

Using the definition of expectation:

$$E[X] = \sum_{i=1}^k x_i \cdot P(X = x_i)$$

For Bernoulli, we have $k = 2$ possible values:

$$E[X] = 0 \cdot P(X = 0) + 1 \cdot P(X = 1)$$

$$E[X] = 0 \cdot (1-p) + 1 \cdot p$$

$$E[X] = 0 + p = p$$

**The expectation of a Bernoulli random variable is $p$**:

$$E[X] = p \quad \text{where } X \sim \text{Bernoulli}(p)$$

**Why does this make sense?**

- If $p = 0.8$ (80% chance of success), we expect to see 1 about 80% of the time

- The long-run average of many Bernoulli trials will be approximately $p$

- Since $X$ only takes values 0 and 1, the expectation represents the "proportion of successes"

## Expectation of a Binomial Random Variable

Now let's compute the expectation of a **binomial random variable** $X \sim \text{Binomial}(n, p)$.

A binomial random variable $X$ represents the number of successes in $n$ independent Bernoulli trials, each with success probability $p$.

**Possible values**: $X$ takes values in $\{0, 1, 2, \ldots, n\}$.
**PMF**: $P(X = k) = \binom{n}{k} p^k(1-p)^{n-k}$ for $k = 0, 1, 2, \ldots, n$

Using the definition of expectation:

$$E[X] = \sum_{k=0}^n k \cdot P(X = k) = \sum_{k=0}^n k \cdot \binom{n}{k} p^k(1-p)^{n-k}$$

$$E[X] = \sum_{k=0}^n k \cdot \binom{n}{k} p^k(1-p)^{n-k}$$

We can use this identity:

$$k \cdot \binom{n}{k} = k \cdot \frac{n!}{k!(n-k)!} = \frac{n!}{(k-1)!(n-k)!} = n \cdot \frac{(n-1)!}{(k-1)!(n-k)!} = n \cdot \binom{n-1}{k-1}$$

$$E[X] = \sum_{k=0}^n n \cdot \binom{n-1}{k-1} p^k(1-p)^{n-k}$$

$$E[X] = n \cdot p \cdot \sum_{k=0}^n \binom{n-1}{k-1} p^{k-1}(1-p)^{n-k}$$

Let $j = k-1$, so $k = j+1$. When $k = 0$, $j = -1$; when $k = n$, $j = n-1$.

**Note**: The term with $j = -1$ contributes 0 because $\binom{n-1}{-1} = 0$ (combinatorial coefficients are 0 for negative indices). So we can adjust the range to start from $j = 0$:

$$E[X] = n \cdot p \cdot \sum_{j=0}^{n-1} \binom{n-1}{j} p^j(1-p)^{(n-1)-j}$$

The sum $\sum_{j=0}^{n-1} \binom{n-1}{j} p^j(1-p)^{(n-1)-j}$ is exactly the binomial expansion of $(p + (1-p))^{n-1} = 1^{n-1} = 1$.

$$E[X] = n \cdot p \cdot 1 = np$$

**The expectation of a binomial random variable is $np$**:

$$E[X] = np \quad \text{where } X \sim \text{Binomial}(n, p)$$

**Why does this make sense?**

- **$n$ trials**: We perform $n$ independent Bernoulli trials

- **$p$ probability**: Each trial has success probability $p$

- **Expected successes**: We expect $p$ proportion of trials to succeed

- **Total expectation**: $n \cdot p$ total expected successes

**Example**: If we flip a fair coin ($p = 0.5$) 100 times ($n = 100$):

- $E[X] = 100 \cdot 0.5 = 50$

- Interpretation: We expect about 50 heads in 100 flips

## Linearity of Expectation

**Linearity of expectation** is one of the most powerful and useful properties in probability theory. It states that expectation is a linear operator, regardless of whether the random variables are independent or not.

For any random variables $X$ and $Y$ (discrete or continuous) and any constants $a$ and $b$:

$$E[aX + bY] = aE[X] + bE[Y]$$

**Key insight**: Linearity of expectation holds even when $X$ and $Y$ are **dependent**!

**Binomial Distribution Revisited**

If $X \sim \text{Binomial}(n, p)$, we can think of $X$ as the sum of $n$ independent Bernoulli$(p)$ random variables:

$$X = B_1 + B_2 + \cdots + B_n$$

where each $B_i \sim \text{Bernoulli}(p)$.

By linearity:

$$E[X] = E[B_1 + B_2 + \cdots + B_n] = E[B_1] + E[B_2] + \cdots + E[B_n] = p + p + \cdots + p = np$$

This gives us the same result as our direct calculation, but much more simply!

## Expectation of a Hypergeometric Random Variable

Now let's compute the expectation of a **hypergeometric random variable** $X \sim \text{Hypergeometric}(N, K, n)$.

A hypergeometric random variable $X$ represents the number of "success" items when drawing $n$ items without replacement from a population of $N$ items, where $K$ items are "successes".

**Possible values**: $X$ takes values in $\{0, 1, 2, \ldots, \min(K, n)\}$.
**PMF**: $P(X = k) = \frac{\binom{K}{k} \cdot \binom{N-K}{n-k}}{\binom{N}{n}}$

Using the definition of expectation:

$$E[X] = \sum_{k=0}^{\min(K,n)} k \cdot P(X = k) = \sum_{k=0}^{\min(K,n)} k \cdot \frac{\binom{K}{k} \cdot \binom{N-K}{n-k}}{\binom{N}{n}}$$

$$E[X] = \sum_{k=0}^{\min(K,n)} k \cdot \frac{\binom{K}{k} \cdot \binom{N-K}{n-k}}{\binom{N}{n}}$$

Use the identity $k \cdot \binom{K}{k} = K \cdot \binom{K-1}{k-1}$

This identity comes from:

$$k \cdot \binom{K}{k} = k \cdot \frac{K!}{k!(K-k)!} = \frac{K!}{(k-1)!(K-k)!} = K \cdot \frac{(K-1)!}{(k-1)!(K-k)!} = K \cdot \binom{K-1}{k-1}$$

$$E[X] = \sum_{k=0}^{\min(K,n)} K \cdot \binom{K-1}{k-1} \cdot \frac{\binom{N-K}{n-k}}{\binom{N}{n}}$$

$$E[X] = K \cdot \sum_{k=0}^{\min(K,n)} \binom{K-1}{k-1} \cdot \frac{\binom{N-K}{n-k}}{\binom{N}{n}}$$

Let $j = k-1$, so $k = j+1$. When $k = 0$, $j = -1$; when $k = \min(K,n)$, $j = \min(K,n)-1$.

**Note**: The term with $j = -1$ contributes 0 because $\binom{K-1}{-1} = 0$. So we can adjust the range to start from $j = 0$:

$$E[X] = K \cdot \sum_{j=0}^{\min(K-1,n-1)} \binom{K-1}{j} \cdot \frac{\binom{N-K}{n-(j+1)}}{\binom{N}{n}}$$

The sum $\sum_{j=0}^{\min(K-1,n-1)} \binom{K-1}{j} \cdot \binom{N-K}{n-(j+1)}$ represents the total number of ways to choose $n-1$ items from $N-1$ items (since we're choosing $j$ from $K-1$ and $n-1-j$ from $N-K$).

This equals $\binom{N-1}{n-1}$.

$$E[X] = K \cdot \frac{\binom{N-1}{n-1}}{\binom{N}{n}} = K \cdot \frac{n}{N} = n \cdot \frac{K}{N}$$

**The expectation of a hypergeometric random variable is $n \cdot \frac{K}{N}$**:

$$E[X] = n \cdot \frac{K}{N} \quad \text{where } X \sim \text{Hypergeometric}(N, K, n)$$

**Example**: If we have a population of 100 items with 30 successes, and we draw 20 items:

- $N = 100$, $K = 30$, $n = 20$

- $E[X] = 20 \cdot \frac{30}{100} = 20 \cdot 0.3 = 6$

- Interpretation: We expect about 6 successes in our sample of 20

This result connects beautifully to the binomial expectation:

- **Binomial**: $E[X] = np$ (with replacement, constant probability)

- **Hypergeometric**: $E[X] = n \cdot \frac{K}{N}$ (without replacement, changing probability)

- **Key difference**: $\frac{K}{N}$ vs. $p$ - the proportion of successes in the population

The hypergeometric expectation shows that even without replacement, the expected proportion of successes in our sample equals the proportion in the population, which is intuitively satisfying.