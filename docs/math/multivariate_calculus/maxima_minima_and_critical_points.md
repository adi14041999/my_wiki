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

