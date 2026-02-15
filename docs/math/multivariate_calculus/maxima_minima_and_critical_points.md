# Maxima, minima, and critical points

## Single-variable recap: first derivative test

A useful result in single-variable calculus is the **first derivative test**. If $f \colon (a, b) \to \mathbb{R}$ is differentiable and attains a local maximum or local minimum at some point $x = c$ inside this open interval (i.e. $a < c < b$), then $f'(c) = 0$. Any point where $f'(c) = 0$ is called a **critical point** for $f$, so local maxima and local minima of differentiable functions on an open interval are always critical points.

It is worth emphasizing that the **differentiability** of $f$ is necessary here. For example, the function $f(x) = |x|$ attains its minimum value of $0$ at $x = 0$, but since $f$ is not differentiable at that point we cannot say that $x = 0$ is a critical point for $f$. The first derivative test does not apply when the function fails to be differentiable at the point in question.

## Motivation

Optimization for functions of $n$ variables is one of the important applications of multivariable calculus.

**Example:** Suppose we look at the graph of the data points $(x_i, y_i)$, and notice that it looks approximately "periodic" (like the graph of sine or cosine) as in the figure below. This might suggest that it is more reasonable to try to fit the data with a function of the form $A\sin(Bx + C)$. 

![img](sin.png)

Here $B$ describes the **frequency** of the oscillation, $A$ its **amplitude**, and $C$ its **phase**. Our fitting task amounts to finding $(A, B, C)$ that minimizes the sum of squared errors:

$$E(A, B, C) = \sum_{i=1}^{N} \bigl( y_i - A\sin(Bx_i + C) \bigr)^2$$

This minimization problem can no longer be carried out using linear algebra alone; it requires multivariable calculus.

**Example:** In the physical sciences, many problems can be studied by means of energy minimization. For example, predicting how a protein folds is a fundamental problem in molecular biology, and a useful clue for understanding the folding that occurs in nature is that it minimizes the energy. The energy of a protein configuration corresponds to a function of hundreds or thousands of variables (that keep track of the configuration), and numerically solving the associated minimization problem is a valuable tool in work on protein folding.

## Testing for critical points

A function $f(x, y)$ achieves a **local maximum** at $(a, b)$ if $f(a, b) \geq f(x, y)$ for all $(x, y)$ sufficiently close to $(a, b)$. In other words, if we move in any direction from $(a, b)$, then as long as we stay nearby, $f(x, y)$ decreases or stays the same.

A function $f(x, y)$ achieves a **local minimum** at $(a, b)$ if $f(a, b) \leq f(x, y)$ for all $(x, y)$ sufficiently close to $(a, b)$. That is, moving in any direction from $(a, b)$ causes $f(x, y)$ to increase or stay the same (as long as we stay near $(a, b)$).

![img](minmax.png)

Let us try to understand what is going on by reducing to functions of one variable. Suppose that $f(x, y)$ has a local maximum at $(a, b)$. If we keep $x$ close to $a$ and keep $y$ fixed at the value $b$, then we have

$$f(a + h, b) \approx f(a, b) + f_x(a, b)\,h$$

for values of $h$ near $0$.

Thus, if $f_x(a, b) > 0$ and we consider small $h > 0$, then this shows that $f(a + h, b)$ should be larger than $f(a, b)$, so $(a, b)$ cannot be a local maximum. Similarly, if $f_x(a, b) < 0$ and we consider small $h < 0$, then $f(a + h, b)$ will again be larger than $f(a, b)$. This is analogous to the fact that if the derivative of a single-variable function is positive somewhere then the graph is sloping upward nearby, while if the derivative is negative there then the graph is sloping downward nearby. This shows that if $(a, b)$ is a local maximum then the options $f_x(a, b) > 0$ and $f_x(a, b) < 0$ are ruled out, so necessarily $f_x(a, b) = 0$. Using precisely the same reasoning by wiggling $y$ while keeping $x = a$, when $(a, b)$ is a local maximum for $f$ we also conclude that $f_y(a, b) = 0$.

We summarize what we have learned as follows: if $f \colon \mathbb{R}^2 \to \mathbb{R}$ has a local maximum at $(a, b)$, then

$$\frac{\partial f}{\partial x}(a, b) = 0 \quad \text{and} \quad \frac{\partial f}{\partial y}(a, b) = 0$$

We can reason the same way for local minima, and also for functions of more than two variables.

Let $f \colon \mathbb{R}^n \to \mathbb{R}$ be a function.

**Theorem:** Suppose that a point $\mathbf{a} \in \mathbb{R}^n$ is either a local maximum or a local minimum of $f$. Then all partial derivatives of $f$ vanish at $\mathbf{x} = \mathbf{a}$; i.e.,

$$\frac{\partial f}{\partial x_i}(\mathbf{a}) = 0 \quad \text{for } 1 \leq i \leq n$$

If $\frac{\partial f}{\partial x_i}(\mathbf{a}) = 0$ for all $1 \leq i \leq n$, then we say $\mathbf{a}$ is a **critical point** for $f$. In particular, every local maximum and every local minimum of $f \colon \mathbb{R}^n \to \mathbb{R}$ is a critical point.

Strictly speaking, in the preceding theorem and definition we should assume $f$ is "differentiable" in an appropriate $n$-variable sense that recovers the notion from single-variable calculus when $n = 1$.

**Example:** Consider $f(x, y) = x^2 + y^2$. We compute that $f_x(x, y) = 2x$, $f_y(x, y) = 2y$, so the simultaneous vanishing of partial derivatives happens only at the origin $(x, y) = (0, 0)$. Now $f(0, 0) = 0$, and if $(x, y)$ is any other point (it does not even need to be near the origin), then $f(x, y) > 0$ by inspection. This means that $(0, 0)$ is a local minimum. Since the condition $f(x, y) \geq f(0, 0)$ even holds without requiring $(x, y)$ to be close to $(0, 0)$, we say that $(0, 0)$ is a "global minimum" for $f$. Similarly, if we consider $g(x, y) = -x^2 - y^2$ then the partial derivatives are $g_x = -2x$, $g_y = -2y$, and these simultaneously vanish only when $(x, y) = (0, 0)$. Furthermore, $g(0, 0) = 0$ while $g(x, y) < 0$ for any other point, so the origin is a global maximum for $g$.

![img](minmax2.png)

**Example:** Let $f(x, y) = 4x^2 - 2xy + y^2 + 8x - 2y + 5$. We compute that $f_x = 8x - 2y + 8$, $f_y = -2x + 2y - 2$. Solving $f_x = f_y = 0$ amounts to solving a system of 2 linear equations in 2 unknowns. Using substitution of one equation into the other, the unique solution is $(x, y) = (-1, 0)$. This is the only critical point of $f$, at which the value is $f(-1, 0) = 1$. 

![img](minmax3.png)

The surface is the graph of $f(x, y) = 4x^2 - 2xy + y^2 + 8x - 2y + 5$. The green line is determined by the equation $f_y = -2x + 2y - 2 = 0$. The red line is determined by the equation $f_x = 8x - 2y + 8 = 0$. Their point of intersection is $(x, y) = (-1, 0)$.

How can we tell whether this is a local maximum or local minimum, or perhaps neither? That is, how do the values of $f(x, y)$ for $(x, y)$ near $(-1, 0)$ compare to the value $1$ at $(-1, 0)$? One method is: plot the graph on a computer (as shown above) and just stare at it! By visual inspection, it looks like it is probably a local minimum, though to be more convinced one might want to "zoom in" and rotate the graph a bit to be sure. This method has a couple of defects, the most serious one being that it doesn't adapt well to problems in more unknowns (we can't literally "see" the graph of $f(x_1, \ldots, x_n)$ in $\mathbb{R}^{n+1}$ when $n > 2$). We want to use methods that can adapt to problems with any number of variables, so we seek another technique.

**Example:** Turning to the surface graph $z = x^2 - y^2$ in $\mathbb{R}^3$ as shown in the figure below, we see that it has a "mountain pass" at the origin precisely because of the dichotomy in the behavior: $x^2 - y^2$ has a local minimum at the origin when restricted to certain lines through the origin in the $xy$-plane (such as the $x$-axis $y = 0$) and a local maximum at the origin when restricted to certain other lines through the origin in the $xy$-plane (such as the $y$-axis $x = 0$). The picture of the surface graph near $(0, 0, f(0, 0)) = (0, 0, 0)$ makes clear why we call $(0, 0)$ a **saddle point** for $x^2 - y^2$.

![img](saddle.png)

The graph of $f(x, y) = x^2 - y^2$. Notice the saddle point at $(x, y) = (0, 0)$.

We will soon define the concept of saddle point in general, and analyze it systematically (for any number of variables). It will be seen to represent a genuinely new phenomenon of the multivariable world with no counterpart in single-variable calculus, and has rather practical significance too.

We emphasize that if $\mathbf{a}$ is a critical point of $f \colon \mathbb{R}^n \to \mathbb{R}$, then it may be neither a local maximum nor a local minimum. In fact, for functions of two or more variables there is a new phenomenon.

**Definition:** A critical point $\mathbf{a} \in \mathbb{R}^n$ of $f \colon \mathbb{R}^n \to \mathbb{R}$ is a **saddle point** if (i) as we move away from $\mathbf{a}$ along some line (e.g., parallel to a coordinate axis) then $f$ increases nearby, so $\mathbf{a}$ is a local minimum along that line, and (ii) as we move away from $\mathbf{a}$ along some other line then $f$ decreases, so $\mathbf{a}$ is a local maximum along that line.

Such behavior can happen with $n$ variables when $n > 1$ because then there are "more lines" in $\mathbb{R}^n$ through a point along which we can move away from the point: in $\mathbb{R}$ there is only one line through a point, but in $\mathbb{R}^2$ there are many lines through a point, let alone in $\mathbb{R}^n$ for $n > 2$. This is a genuinely multivariable phenomenon.

We do not yet have any general methods for determining when a critical point is a local maximum or local minimum or a saddle point (or perhaps even more exotic possibilities?). The method for this involves second partial derivatives, which we will discuss sometime later.

**Example:** To illustrate why the signs of the second partials are insufficient when $n > 1$, we give a function $f(x, y)$ with a critical point $\mathbf{a}$ at which all three values $f_{xx}(\mathbf{a})$, $f_{yy}(\mathbf{a})$, $f_{xy}(\mathbf{a})$ are positive yet $f$ has a saddle point (rather than a local minimum) at $\mathbf{a}$! Consider $f(x, y) = x e^{x+3y} - x + \sin^2(y)$. This has partials $f_x = (1 + x)e^{x+3y} - 1$ and $f_y = 3x e^{x+3y} + 2\sin(y)\cos(y) = 3x e^{x+3y} + \sin(2y)$ that both vanish at $\mathbf{0}$, so the origin is a critical point of $f$. We compute $f_{xx} = (2 + x)e^{x+3y}$, $f_{yy} = 9x e^{x+3y} + 2\cos(2y)$, $f_{xy} = 3(1 + x)e^{x+3y}$, so $f_{xx}(0) = 2$, $f_{yy}(0) = 2$, $f_{xy}(0) = 3$ are all positive. On the coordinate axes we have $f(x, 0) = x e^{x} - x = x(e^{x} - 1)$ and $f(0, y) = \sin^2(y)$, each of which is checked (via single-variable calculus) to have a local minimum at the origin.

But the behavior of $f(x, y)$ on the line $y = -x$ is encoded in the function $g(x) = f(x, -x) = x e^{-2x} - x + \sin^2(-x) = x(e^{-2x} - 1) + \sin^2(x)$ that has a critical point at $x = 0$ (because $g'(x) = (1 - 2x)e^{-2x} - 1 + \sin(2x)$ satisfies $g'(0) = 0$) yet that is a local maximum for $g(x)$ since $g''(x) = 4(x - 1)e^{-2x} + \sin(4x)$ satisfies $g''(0) = -4 < 0$. Hence, $f$ has a saddle point at the origin.

