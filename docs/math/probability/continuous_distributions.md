# Continuous Distributions

Continuous distributions are probability distributions for random variables that can take on any value in a continuous range (typically an interval of real numbers). Unlike discrete distributions, continuous random variables have probability density functions (PDFs) rather than probability mass functions (PMFs).

For a continuous random variable $X$, the probability of $X$ taking any specific value is exactly 0:

$$P(X = x) = 0 \quad \text{for any specific value } x$$

Since $X$ can take uncountably many values in a continuous range, the probability of any single value must be 0. Otherwise, the total probability would exceed 1.

**Consequence**: We cannot use probability mass functions (PMFs) like we do for discrete random variables, because $P(X = x)$ is always 0. Instead, we need probability density functions (PDFs) to describe the distribution.

## Probability Density Function (PDF)

**Definition**: A function $f_X(x)$ such that $P(a \leq X \leq b) = \int_a^b f_X(x) dx$ for all $a$ and $b$.

When $a = b$, the interval $[a, b]$ becomes a single point, and we have:

$$P(a \leq X \leq a) = P(X = a) = \int_a^a f_X(x) dx = 0$$

This confirms our earlier statement that $P(X = x) = 0$ for any specific value $x$ in a continuous distribution. The integral over a single point (which has zero length) is always 0.

**Properties**: 

- $f_X(x) \geq 0$ for all $x$

- $\int_{-\infty}^{\infty} f_X(x) dx = 1$

**What does $f_X(x)$ actually mean?**

The PDF $f_X(x)$ represents the **probability density** at point $x$. To understand this, consider a small interval around $x$:

$$P(x - \frac{\epsilon}{2} \leq X \leq x + \frac{\epsilon}{2}) = \int_{x - \frac{\epsilon}{2}}^{x + \frac{\epsilon}{2}} f_X(t) dt$$

For very small $\epsilon$, this integral is approximately:

$$P(x - \frac{\epsilon}{2} \leq X \leq x + \frac{\epsilon}{2}) \approx f_X(x) \cdot \epsilon$$

$$f_X(x) \approx \frac{P(x - \frac{\epsilon}{2} \leq X \leq x + \frac{\epsilon}{2})}{\epsilon}$$

**Interpretation**: $f_X(x)$ tells us how much probability "mass" is concentrated around the point $x$. The probability of falling in a small interval around $x$ is approximately $f_X(x)$ times the length of that interval. Think of probability density like physical density:

- **Probability mass** = $P(x - \frac{\epsilon}{2} \leq X \leq x + \frac{\epsilon}{2})$ (the "amount" of probability)

- **Volume** = $\epsilon$ (the "size" of the interval)

- **Density** = $f_X(x)$ (how "concentrated" the probability is)

Just as $\text{density} = \frac{\text{mass}}{\text{volume}}$ in physics, we have:

$$\text{probability density} = \frac{\text{probability mass}}{\text{interval length}}$$

This explains why $f_X(x)$ can be greater than 1 - it's not a probability, but a density!

**Key insight**: While $P(X = x) = 0$, the density $f_X(x)$ tells us how likely $X$ is to fall near $x$ relative to other points.

## Cumulative Distribution Function (CDF)

The **Cumulative Distribution Function** (CDF) of a continuous random variable $X$ is defined as:

$$F_X(x) = P(X \leq x) = \int_{-\infty}^x f_X(t) dt$$

**What it represents**: $F_X(x)$ gives the probability that $X$ takes a value less than or equal to $x$. Here, $f_X(t)$ is the probability density function (PDF) of $X$.

**Properties**

1. **Non-decreasing**: $F_X(x_1) \leq F_X(x_2)$ whenever $x_1 \leq x_2$

2. **Limits**: $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to \infty} F_X(x) = 1$

3. **Right-continuous**: $F_X(x) = \lim_{h \to 0^+} F_X(x + h)$

4. **Probability interpretation**: $P(a < X \leq b) = F_X(b) - F_X(a)$

Since the CDF is the integral of the PDF, we can recover the PDF by differentiating the CDF:

$$f_X(x) = \frac{d}{dx} F_X(x) = F_X'(x)$$

**Example**: If $F_X(x) = 1 - e^{-x}$ for $x \geq 0$ (and $F_X(x) = 0$ for $x < 0$), then:

$$f_X(x) = \frac{d}{dx} F_X(x) = \frac{d}{dx}(1 - e^{-x}) = e^{-x}$$

This gives us the PDF: $f_X(x) = e^{-x}$ for $x \geq 0$ (and $f_X(x) = 0$ for $x < 0$).

**Key insight**: The PDF tells us where the CDF is changing rapidly (high density) versus slowly (low density).

**Why CDFs are useful**

1. **Probability calculations**: Easy to find $P(X \leq x)$ or $P(a < X \leq b)$

2. **Distribution comparison**: Can compare distributions by plotting CDFs

3. **Quantiles**: The $p$-th quantile $x_p$ satisfies $F_X(x_p) = p$

## Expectation of a Continuous Random Variable

The **expectation** (or **expected value**) of a continuous random variable $X$ is defined as:

$$E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

**What it represents**: $E[X]$ is the "center of mass" or "average value" of the distribution, representing the long-run average if we were to sample from this distribution many times. Here, $f_X(x)$ is the probability density function (PDF) of $X$.

Think of the PDF as a "weight distribution" along the real line:

- **$f_X(x)$**: How much "weight" (probability density) is at point $x$

- **$x \cdot f_X(x)$**: The "weighted position" at point $x$

- **$\int_{-\infty}^{\infty} x \cdot f_X(x) dx$**: The total "center of mass" of all the weight

## Variance

The **variance** of a random variable measures how spread out the distribution is around its mean. It's defined as the expected squared deviation from the mean.

**For any random variable $X$ (discrete or continuous)**:

$$\text{Var}(X) = E[(X - E[X])^2]$$

**Why not other measures of deviation?**

**Problem 1: $E[X - E[X]]$**

This would always equal 0 because:

$$E[X - E[X]] = E[X] - E[E[X]] = E[X] - E[X] = 0$$

The average deviation from the mean is always 0, so this tells us nothing about spread.

**Problem 2: $E[|X - E[X]|]$ (Mean Absolute Deviation)**

While this measures spread, it has mathematical disadvantages:

- **Non-differentiable**: The absolute value function isn't smooth, making calculus difficult

- **Harder to work with**: Properties like additivity are more complex

**Why $E[(X - E[X])^2]$ is perfect:**

1. **Always positive**: $(X - E[X])^2 \geq 0$ for all $X$, so variance is always non-negative

2. **Mathematically tractable**: Squaring gives smooth, differentiable functions

3. **Additivity**: Variance of sum of independent variables equals sum of variances

4. **Theoretical elegance**: Leads to beautiful results in probability theory

5. **Statistical properties**: Optimal for many statistical procedures

**Alternative formula** (often easier to compute):

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

**Discrete Case**

For a discrete random variable $X$ with PMF $p_X(x)$:

$$\text{Var}(X) = \sum_x (x - E[X])^2 \cdot p_X(x) = \sum_x x^2 \cdot p_X(x) - (E[X])^2$$

**Example**: For a Bernoulli random variable $X \sim \text{Bernoulli}(p)$:

- $E[X] = p$

- $E[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p$

- $\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$

**Continuous Case**

For a continuous random variable $X$ with PDF $f_X(x)$:

$$\text{Var}(X) = \int_{-\infty}^{\infty} (x - E[X])^2 \cdot f_X(x) dx = \int_{-\infty}^{\infty} x^2 \cdot f_X(x) dx - (E[X])^2$$

**Example**: For an exponential random variable $X \sim \text{Exponential}(\lambda)$:

- $E[X] = \frac{1}{\lambda}$

- $E[X^2] = \int_0^{\infty} x^2 \cdot \lambda e^{-\lambda x} dx = \frac{2}{\lambda^2}$ (using integration by parts)

- $\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}$

## Standard Deviation

The **standard deviation** of a random variable $X$ is the square root of its variance:

$$\sigma_X = \sqrt{\text{Var}(X)} = \sqrt{E[(X - E[X])^2]}$$

**What it represents**: Standard deviation measures spread in the same units as the original random variable, making it more interpretable than variance.

Variance has units that are the square of the original units. For example, if $X$ measures height in meters, $\text{Var}(X)$ is in square meters. If $X$ measures time in seconds, $\text{Var}(X)$ is in square seconds. Standard deviation has the same units as $X$.

## Uniform Distribution

The **uniform distribution** is the simplest continuous distribution, where every value in an interval has equal probability density.

A random variable $X$ follows a **uniform distribution** on the interval $[a, b]$ (denoted $X \sim \text{Uniform}(a, b)$) if its PDF is:

$$f_X(x) = \begin{cases} 
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

**Parameters**:

- **$a$**: Lower bound of the interval

- **$b$**: Upper bound of the interval ($b > a$)

- **Support**: $X$ takes values in $[a, b]$

Every point in $[a, b]$ has the same probability density. The PDF is a horizontal line (rectangle). If you randomly pick a point from $[a, b]$, every point is equally likely

**Examples**:

- Random number generation between 0 and 1

- Random angle selection (0 to 2π)

- Random time selection within an hour

- Random position selection along a line segment

The cumulative distribution function is:

$$F_X(x) = \begin{cases}
0 & \text{if } x < a \\
\frac{x-a}{b-a} & \text{if } a \leq x \leq b \\
1 & \text{if } x > b
\end{cases}$$

$F_X(x)$ increases linearly from 0 to 1 as $x$ goes from $a$ to $b$.

**Expectation**:

$$E[X] = \int_a^b x \cdot \frac{1}{b-a} dx = \frac{1}{b-a} \int_a^b x dx = \frac{1}{b-a} \cdot \frac{b^2 - a^2}{2} = \frac{a + b}{2}$$

**Variance**:

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

First, calculate $E[X^2]$:

$$E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a} dx = \frac{1}{b-a} \cdot \frac{b^3 - a^3}{3} = \frac{b^3 - a^3}{3(b-a)}$$

Then:

$$\text{Var}(X) = \frac{b^3 - a^3}{3(b-a)} - \left(\frac{a + b}{2}\right)^2 = \frac{(b-a)^2}{12}$$

**Standard deviation**:

$$\sigma_X = \frac{b-a}{2\sqrt{3}}$$