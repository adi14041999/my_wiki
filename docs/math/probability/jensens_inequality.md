# Jensen's inequality

Jensen's inequality is one of the most fundamental and widely-used inequalities in probability theory and analysis. It provides a powerful tool for relating the expectation of a convex (or concave) function to the function of the expectation.

## Convex and Concave functions

### Convex functions

Before stating Jensen's inequality, we need to understand what convex and concave functions are.

A function $f: \mathbb{R} \to \mathbb{R}$ is **convex** if for any two points $x_1, x_2$ in its domain and any $\lambda \in [0,1]$:

$$f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)$$

**Geometric interpretation**: A function is convex if the line segment connecting any two points on its graph lies above or on the graph itself. In other words, the graph "curves upward" or is "bowl-shaped."

Let's see how the mathematical definition translates to the geometric property. Consider two points $(x_1, f(x_1))$ and $(x_2, f(x_2))$ on the graph of a convex function $f$.

The line segment connecting these points can be parameterized as:

$$L(\lambda) = (1-\lambda)(x_1, f(x_1)) + \lambda(x_2, f(x_2))$$

This gives us:

- **x-coordinate**: $(1-\lambda)x_1 + \lambda x_2 = x_1 + \lambda(x_2 - x_1)$

- **y-coordinate**: $(1-\lambda)f(x_1) + \lambda f(x_2)$

The convexity condition states that for any point on this line segment (i.e., for any $\lambda \in [0,1]$), the y-coordinate of the line is greater than or equal to the function value at the corresponding x-coordinate:

$$(1-\lambda)f(x_1) + \lambda f(x_2) \geq f((1-\lambda)x_1 + \lambda x_2)$$

**Visual interpretation**: This means that if you draw a straight line between any two points on the graph of a convex function, the entire line segment lies above or on the graph. The function "holds water" - it forms a bowl shape.

### Concave functions

A function $f: \mathbb{R} \to \mathbb{R}$ is **concave** if for any two points $x_1, x_2$ in its domain and any $\lambda \in [0,1]$:

$$f(\lambda x_1 + (1-\lambda) x_2) \geq \lambda f(x_1) + (1-\lambda) f(x_2)$$

**Geometric interpretation**: A function is concave if the line segment connecting any two points on its graph lies below or on the graph itself. In other words, the graph "curves downward" or is "cave-shaped."

Think of convex functions as "smiling" curves (like a bowl) and concave functions as "frowning" curves (like a cave). 

For twice-differentiable functions, we have a convenient test:

- **$f$ is convex** if and only if $f''(x) \geq 0$ for all $x$ in the domain

- **$f$ is concave** if and only if $f''(x) \leq 0$ for all $x$ in the domain

**Examples**:

- $f(x) = x^2$: $f''(x) = 2 > 0$ → convex

- $f(x) = \log(x)$: $f''(x) = -\frac{1}{x^2} < 0$ → concave

- $f(x) = e^x$: $f''(x) = e^x > 0$ → convex

## Theorem (Jensen's Inequality) 

Let $X$ be a random variable and let $\phi$ be a convex function. Then:

$$\phi(E[X]) \leq E[\phi(X)]$$

If $\phi$ is strictly convex, then equality holds if and only if $X$ is constant (i.e., $X = E[X]$ with probability 1).

**For concave functions**: If $\phi$ is concave, then the inequality is reversed:

$$\phi(E[X]) \geq E[\phi(X)]$$

Jensen's inequality captures a fundamental geometric insight: **the function of the average is less than or equal to the average of the function** (for convex functions).

Think of it this way: if you have a convex function (like $f(x) = x^2$), the graph "curves upward." If you take two points on this curve and draw a line between them, the line lies above the curve. This means that the average of the function values at two points is greater than the function value at the average of those points.

**Example 1: Quadratic function** Let $X$ be any random variable with finite variance, and let $\phi(x) = x^2$. Since $x^2$ is convex:

$$(E[X])^2 \leq E[X^2]$$

This immediately gives us the relationship between mean and variance:

$$\text{Var}(X) = E[X^2] - (E[X])^2 \geq 0$$

**Example 2: Logarithm function** Let $X$ be a positive random variable, and let $\phi(x) = \log(x)$. Since $\log(x)$ is concave:

$$\log(E[X]) \geq E[\log(X)]$$

**Example 3: Exponential function** Let $X$ be any random variable, and let $\phi(x) = e^x$. Since $e^x$ is convex:

$$e^{E[X]} \leq E[e^X]$$

This inequality is crucial in proving concentration inequalities like Hoeffding's inequality.

## Multivariate version

Jensen's inequality also extends to multivariate functions. If $\phi: \mathbb{R}^n \to \mathbb{R}$ is convex and $\mathbf{X}$ is a random vector, then:

$$\phi(E[\mathbf{X}]) \leq E[\phi(\mathbf{X})]$$

This multivariate version is particularly useful in machine learning and optimization contexts where we deal with vector-valued random variables.