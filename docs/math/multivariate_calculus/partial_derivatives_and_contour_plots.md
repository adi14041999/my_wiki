# Partial derivatives and contour plots

## Single-variable derivative review

The derivative of a function $f(x)$ at a point $x = c$ is written in one of the equivalent forms:

$$f'(c), \quad \frac{df}{dx}(c), \quad \left.\frac{df}{dx}\right|_{x=c}$$

It is convenient to think of the derivative as a function too. Thus we write $f'(x)$ or $\frac{df}{dx}$ for the function which assigns to every point $x$ the value $f'(x)$, the derivative of $f$ at $x$. This assumes that $f$ has a derivative at every point $x$.

We can have two different ways of thinking about derivatives: (i) they describe the sensitivity in the values of $f$ to small changes in the independent variable ($f(c + h) \approx f(c) + f'(c)h$ for $h$ near 0), and (ii) they are purely geometric quantities (slopes of tangent lines). Like so many ideas in mathematics, derivatives can be interpreted in many different ways.

For a real-valued function $f(x)$, the derivative at a point $x$ (denoted $f'(x)$ or $\frac{df}{dx}$) is defined as:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

### Local minima and local maxima

**Local minimum:** A point $c$ in the domain of $f$ is a **local minimum** of $f$ if there exists an open interval around $c$ such that $f(c) \leq f(x)$ for all $x$ in that interval. In other words, $f(c)$ is the smallest value of $f$ in some region containing $c$.

**Local maximum:** A point $c$ in the domain of $f$ is a **local maximum** of $f$ if there exists an open interval around $c$ such that $f(c) \geq f(x)$ for all $x$ in that interval. In other words, $f(c)$ is the largest value of $f$ in some region containing $c$.

![img](minmax.png)

Collectively, local minima and local maxima are called **local extrema**. If $f(x)$ is differentiable at every $x$, and if $x = c$ is a local maximum or local minimum for $f$, then $f'(c) = 0$. In other words, at such points the tangent line is horizontal, or alternatively (in terms of our other interpretation of the derivative) the value of $f(x)$ is insensitive (to first order) to small changes in $x$.

Keep in mind that the converse is false: $f'(c) = 0$ does **not** imply that $c$ is a local maximum or local minimum. For example, consider $f(x) = x^3$. We have $f'(x) = 3x^2$, so $f'(0) = 0$. But $x = 0$ is neither a local maximum nor a local minimum: for any small neighborhood around $0$, we have $f(x) < f(0) = 0$ when $x < 0$ and $f(x) > f(0) = 0$ when $x > 0$, so $f$ takes values both above and below $f(0)$ arbitrarily close to $0$. Such a point is called an **inflection point** (or, in the language of critical points, a *saddle point*). Thus $f'(c) = 0$ is a necessary condition for a local extremum at $c$, but not a sufficient one.

The figure below shows the graph of $f(x) = x^3$ over the interval $[-1, 1]$.

![img](x3.png)

## Partial derivatives, a first look