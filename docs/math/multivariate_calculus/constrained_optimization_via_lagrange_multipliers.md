# Constrained optimization via Lagrange multipliers

A new feature in the multivariable case is that it is often necessary to maximize or minimize subject to a **constraint**. For example,

$$\min_{(x,\,y)} \; x^3 + 2xy^2 \quad \text{subject to} \quad 3x^2 + 4y^2 = 1.$$

A basic two-variable version of the problem — which already contains many of the essential mathematical issues of the general multivariable case — is:

> Find the maximum of $f(x, y)$ subject to the constraint $g(x, y) = c$, for a specific $c \in \mathbb{R}$,

where $g$ is some auxiliary function of prior interest (i.e. the constraint $g(x, y) = c$ expresses some real-world condition on the points $(x, y)$ of interest for the optimization problem). Geometrically, we are trying to maximize $f$ while restricting $(x, y)$ to a certain curve — a **level curve** of $g$. For example:

$$\text{find the extrema of } \; x^2 - y \quad \text{subject to} \quad x^3 + x + y^3 + y = 1.$$

Alternatively, we might want to work with a constraint of the form

> Find the maximum of $f(x, y)$ subject to the constraint $g(x, y) \le c$, for a specific $c \in \mathbb{R}$.

There are many real-world reasons to want this. As one example, such inequality constraints show up in machine learning algorithms used to build classifiers — in particular, in **support vector machines**.

One approach is to try to solve for $y$ in terms of $x$ under the constraint: starting from $g(x, y) = c$, try to solve for $y$ in terms of $x$, substitute it into $f(x, y)$, and apply single-variable calculus to $f(x, y(x))$. This does not work well when $g$ is complicated. It is often impossible to explicitly solve for $y$ in terms of $x$ on the constraint curve $g(x, y) = c$, and even when it is possible, $f(x, y(x))$ is typically a mess. A better method is called **Lagrange multipliers**.