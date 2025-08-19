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
