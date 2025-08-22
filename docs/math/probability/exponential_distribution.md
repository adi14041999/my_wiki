# Exponential Distribution

The **Exponential distribution** is a continuous probability distribution that describes the time between events in a Poisson process. It is characterized by its memoryless property and is fundamental in reliability theory, queuing theory, and survival analysis.

A continuous random variable $X$ has an **exponential distribution** with parameter $\lambda > 0$ if its PDF is:

$$f_X(x) = \begin{cases}
\lambda e^{-\lambda x} & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}$$

We write this as $X \sim \text{Exponential}(\lambda)$ or $X \sim \text{Exp}(\lambda)$.

**Parameters:**

- $\lambda$: **rate parameter** (events per unit time)

## Memory-less property

The exponential distribution is **memoryless**, meaning:

$$P(X > s + t \mid X > s) = P(X > t)$$

If you've already waited $s$ units of time, the probability of waiting an additional $t$ units is the same as if you were starting fresh.

**Proof:**

$$P(X > s + t \mid X > s) = \frac{P(X > s + t \text{ AND } X > s)}{P(X > s)}$$

Since $s + t > s$, if $X > s + t$, then automatically $X > s$. Therefore:

$$P(X > s + t \text{ AND } X > s) = P(X > s + t)$$

Using the CDF of the exponential distribution:

$$P(X > s + t) = 1 - P(X \leq s + t) = 1 - (1 - e^{-\lambda(s + t)}) = e^{-\lambda(s + t)}$$

$$P(X > s) = 1 - P(X \leq s) = 1 - (1 - e^{-\lambda s}) = e^{-\lambda s}$$

$$P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s + t)}}{e^{-\lambda s}}$$

$$\frac{e^{-\lambda(s + t)}}{e^{-\lambda s}} = e^{-\lambda(s + t) + \lambda s} = e^{-\lambda t} = P(X > t)$$

Therefore:

$$P(X > s + t \mid X > s) = P(X > t)$$

**This proves the memoryless property!**

**Example**: If a light bulb has been working for 100 hours, the probability it works for another 50 hours is the same as the probability a new bulb works for 50 hours.

## Connection to Poisson Process
The exponential distribution describes the **inter-arrival times** in a Poisson process:

- **Poisson process**: Events occur at a constant average rate $\lambda$

- **Exponential distribution**: Time between consecutive events

- **Relationship**: If events occur at rate $\lambda$, then inter-arrival times follow $\text{Exp}(\lambda)$

**Theorem**: In a Poisson process with rate $\lambda$, the inter-arrival times (time between consecutive events) are independent and identically distributed exponential random variables with parameter $\lambda$.

**Proof**: Let $T_1, T_2, T_3, \ldots$ be the inter-arrival times. We need to show that each $T_i \sim \text{Exp}(\lambda)$.

In a Poisson process, events occur continuously over time. We need multiple random variables because:

1. $T_1$: Time from start (time 0) until the **first event** occurs
2. $T_2$: Time from the **first event** until the **second event** occurs  
3. $T_3$: Time from the **second event** until the **third event** occurs
4. And so on...: Each $T_i$ represents the time between the $(i-1)$th and $i$th events

Each $T_i$ represents a **different time interval** between consecutive events. Since events occur randomly, each of these time intervals is itself a random variable.

The first event occurs at time $T_1$. The probability that no events occur in time interval $[0, t]$ is:

$$P(T_1 > t) = P(\text{No events in } [0, t])$$

Since the number of events in $[0, t]$ follows $\text{Poisson}(\lambda t)$:

$$P(T_1 > t) = P(\text{Poisson}(\lambda t) = 0) = \frac{(\lambda t)^0}{0!} e^{-\lambda t} = e^{-\lambda t}$$

Therefore:

$$P(T_1 \leq t) = 1 - P(T_1 > t) = 1 - e^{-\lambda t}$$

This is exactly the CDF of $\text{Exp}(\lambda)$, so $T_1 \sim \text{Exp}(\lambda)$.

**Key insight**: Each $T_i$ follows the same exponential distribution $\text{Exp}(\lambda)$ because:

1. **Stationary increments**: The Poisson process has the same behavior regardless of when we start observing

2. **Memoryless property**: The exponential distribution "forgets" how long we've been waiting

3. **Independent increments**: Each time interval is independent of previous intervals

**Example**: Consider a Poisson process with rate $\lambda = 2$ events per hour.

**Poisson aspect**: Number of events in 3 hours follows $\text{Poisson}(2 \times 3) = \text{Poisson}(6)$.

**Exponential aspect**: Time between consecutive events follows $\text{Exp}(2)$.

**Verification**: 

- **Expected events** in 3 hours: $E[\text{Poisson}(6)] = 6$

- **Expected time** between events: $E[\text{Exp}(2)] = \frac{1}{2}$ hour

- **Consistency**: $\frac{3 \text{ hours}}{6 \text{ events}} = \frac{1}{2} \text{ hour per event}$ ✓

## CDF of the Exponential Distribution

The CDF of $X \sim \text{Exp}(\lambda)$ is:

$$F_X(x) = \begin{cases}
1 - e^{-\lambda x} & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}$$

**Derivation**: 

$$F_X(x) = \int_0^x \lambda e^{-\lambda t} \, dt = \lambda \int_0^x e^{-\lambda t} \, dt = \lambda \left[-\frac{1}{\lambda} e^{-\lambda t}\right]_0^x = 1 - e^{-\lambda x}$$

## Expectation and Variance

**Expectation**

$$E[X] = \frac{1}{\lambda}$$

**Proof**:

$$E[X] = \int_0^{\infty} x \cdot \lambda e^{-\lambda x} \, dx$$

Using integration by parts with $u = x$ and $dv = \lambda e^{-\lambda x} \, dx$:

$$E[X] = \left[-x e^{-\lambda x}\right]_0^{\infty} - \int_0^{\infty} (-e^{-\lambda x}) \, dx$$

The boundary term evaluates to 0:

- At $x = 0$: $-0 \cdot e^0 = 0$

- At $x = \infty$: $-x \cdot e^{-\lambda x} \to 0$ (exponential decay dominates)

Therefore:

$$E[X] = \int_0^{\infty} e^{-\lambda x} \, dx = \left[-\frac{1}{\lambda} e^{-\lambda x}\right]_0^{\infty} = \frac{1}{\lambda}$$

**Variance**

$$\text{Var}(X) = \frac{1}{\lambda^2}$$

**Proof**:

$$\text{Var}(X) = E[X^2] - (E[X])^2 = E[X^2] - \frac{1}{\lambda^2}$$

We need to calculate $E[X^2]$:

$$E[X^2] = \int_0^{\infty} x^2 \cdot \lambda e^{-\lambda x} \, dx$$

Using integration by parts twice:

$$E[X^2] = \frac{2}{\lambda^2}$$

Therefore:

$$\text{Var}(X) = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}$$

**Standard Deviation**

$$\sigma_X = \frac{1}{\lambda}$$

**Note**: For the exponential distribution, the mean equals the standard deviation.