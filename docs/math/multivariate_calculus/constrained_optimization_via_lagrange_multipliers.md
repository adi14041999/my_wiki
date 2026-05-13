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

## Intuition

Let's start off with a couple of examples.

**Example:** Consider the unit sphere $S$ centered at the origin $\mathbf{0}$ in $\mathbb{R}^3$, and let $f(x, y, z) = z$. The sphere $S$ is defined by $x^2 + y^2 + z^2 = 1$, so for $g(x, y, z) = x^2 + y^2 + z^2$, finding the extrema of $z$ on $S$ is the same as finding the points in the region $g = 1$ at which $f$ attains maximal or minimal values.

By inspection, $f$ attains its maximal value on $S$ at the north pole $\mathbf{p} = (0, 0, 1)$ and its minimal value at the south pole $-\mathbf{p} = (0, 0, -1)$.

These two points on $S$ are **not** local extrema for $f$ on all of $\mathbb{R}^3$: for tiny $t > 0$, at $(0, 0, 1 + t)$ near $\mathbf{p}$ the value $f = 1 + t$ is larger than the maximal value $f(\mathbf{p}) = 1$ of $f = z$ **on** $S$, and similarly at $(0, 0, -1 - t)$ near $-\mathbf{p}$ the value $f = -1 - t$ is less than the minimal value $f(-\mathbf{p}) = -1$ on $S$. The points $(0, 0, \pm(1 + t))$ lie **outside** $S$, so they are irrelevant when optimizing $f$ subject to the constraint $g = 1$ that defines $S$.

We have just observed an important point. A solution to a constrained optimization problem on $\mathbb{R}^n$ of the form

> optimize $F(\mathbf{x})$ subject to the condition $G(\mathbf{x}) = c$

is typically **not** a local extremum for $F$ on the ambient $\mathbb{R}^n$. Hence there is no reason for it to be a critical point of $F$; i.e. **no reason for $\nabla F$ to vanish there**. Indeed, in our sphere example with $f(x, y, z) = z$, we have

$$(\nabla f)(\mathbf{x}) = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} \ne \mathbf{0} \quad \text{for all } \mathbf{x} \in \mathbb{R}^3,$$

yet the constrained extrema $\pm \mathbf{p}$ on $S$ still exist.

But there **is** nevertheless something special about the behavior of $\nabla f$ at the points $\pm \mathbf{p}$ where $f$ attains its extreme values on the region $S$ defined by $g = 1$. To explain it, we will work with the gradient $\nabla g$ at points of the constraint region $S$. For any point $\mathbf{x} = (x, y, z) \in \mathbb{R}^3$ (which may or may not lie in $S$),

$$(\nabla g)(\mathbf{x}) = \begin{pmatrix} 2x \\ 2y \\ 2z \end{pmatrix} = 2\mathbf{x}.$$

In the figure below we show $S = \{\mathbf{x} \in \mathbb{R}^3 : g(\mathbf{x}) = 1\}$ with the corresponding gradient vector $(\nabla g)(\mathbf{x}) = 2\mathbf{x}$ drawn at every point $\mathbf{x} \in S$. This is a vector **perpendicular to the tangent plane of $S$ at $\mathbf{x}$**, pointing outward from the sphere with length $2$ (the lengths don't all look the same in the figure due to the effect of perspective). The non-negative coordinate axes are also drawn in light blue, but they are irrelevant in what follows.

![img](sphere_gradients.png)

Here is the **key observation**: for each $\mathbf{x} \in S$, compare the "radial" line through $\mathbf{x}$ along the direction of $(\nabla g)(\mathbf{x})$ with the "vertical" line through $\mathbf{x}$ along the direction of $(\nabla f)(\mathbf{x}) = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$. The points $\mathbf{a} \in S$ for which these two lines through $\mathbf{a}$ coincide are exactly the north pole $\mathbf{p}$ and the south pole $-\mathbf{p}$ (indicated by the red radial vectors in the figure).

Equality of these two lines at a point $\mathbf{a} \in S$ (equality of the line spanned by $(\nabla f)(\mathbf{a})$ and the line spanned by $(\nabla g)(\mathbf{a})$) says exactly that $(\nabla f)(\mathbf{a})$ is a **scalar multiple** of the nonzero gradient vector $(\nabla g)(\mathbf{a})$:

$$(\nabla f)(\mathbf{a}) = \lambda \, (\nabla g)(\mathbf{a})$$

for some $\lambda \in \mathbb{R}$ (here $\lambda = 1/2$, but the specific scalar isn't essential). This special property at the constrained extrema — that $(\nabla f)(\mathbf{a})$ is a scalar multiple of $(\nabla g)(\mathbf{a})$ — will turn out to be a general feature of solutions to all constrained extremum problems.

**Example:** For $g(x, y) = x^4 + x^3 y + y^2$, let's find the local extrema of

$$f(x, y) = x^3 + xy^2$$

subject to the constraint $g(x, y) = 1$. This amounts to studying the behavior of $f$ on the curve $C$ defined by $g(x, y) = 1$, shown in the figure below.

By "local" extrema of $f$ on $C$ we mean points $P \in C$ such that all $(x, y) \in C$ near $P$ satisfy

$$f(P) \ge f(x, y) \quad (\text{constrained local maximum at } P)$$

or

$$f(P) \le f(x, y) \quad (\text{constrained local minimum at } P).$$

![img](redblue.png)

To visualize the task of finding local extrema of $f$ on $C$, consider the **level curves** $f(x, y) = m$ and let $m$ vary. For some values of $m$ the curve $f = m$ does not touch $C$, so $f$ never attains the value $m$ on $C$. For other values of $m$, the curve $f = m$ does touch $C$; the points where this happens are precisely the points where $f$ takes the value $m$ on $C$.

The utility of this is illustrated in the figure below, which shows a blue level curve $f(x, y) = m_0$ for a specific $m_0 \approx 2.056$ together with black level curves $f(x, y) = m$ for $m$ near $m_0$. For $m$ near $m_0$:

- if $m > m_0$, the curve $f = m$ does **not** touch $C$;
- if $m < m_0$, the curve $f = m$ touches $C$ at a few points;
- the curve $f = m_0$ touches $C$ at exactly one point $P_0$ (black dot), and does so **tangentially** — i.e. the tangent lines at $P_0$ to the blue curve $f = m_0$ and the red curve $C$ coincide.

![img](zoomedin.png)

To summarize:

1. for $Q \in C$ near $P_0$ we have $f(Q) \le m_0 = f(P_0)$, so $f$ on $C$ has a local maximum at $P_0$;
2. the curve $f = m_0$ through the constrained local extremum $P_0$ is **tangent** to $C$ at $P_0$;
3. for nearby $m < m_0$, the curve $f = m$ is **not** tangent to $C$ at the points $Q$ near $P_0$ where this level curve of $f$ meets $C$.

The key is to reinterpret (2) and (3) in terms of **gradient vectors**. At any point, the perpendicular direction to a level curve is the direction of the gradient at that point. Equality of two lines through a point in $\mathbb{R}^2$ (such as the two tangent lines at a common point above) is the same as equality of their perpendicular directions through that point. So we can restate (2) and (3) as:

- **(2′)** $(\nabla f)(P_0)$ and $(\nabla g)(P_0)$ span the same line;
- **(3′)** for $Q \in C$ near $P_0$ but distinct from $P_0$, the vectors $(\nabla f)(Q)$ and $(\nabla g)(Q)$ do **not** point along the same line.

Hence, for $(a, b) \in C$ near $P_0$, the condition

$$(\nabla f)(a, b) \text{ is a scalar multiple of } (\nabla g)(a, b)$$

holds **at $P_0$ and nowhere else nearby on $C$**. We have not yet said what the scalar multiplier is, but the mere fact that such a relationship holds between the two gradients at a point of $C$ — regardless of the value of the scalar multiplier — is a very restrictive condition that picks out $P_0$ from all other nearby points of $C$.

Equivalently, we restricted attention to the **level curve** $g(x, y) = 1$ and asked where it met the level curves $f(x, y) = m$ as $m$ varied. Each intersection point lay on $C$ and had $f = m$ there; a constrained local maximum (minimum) was a point of $C$ where no nearby point on $C$ had a **larger** (respectively **smaller**) value of $m$, i.e. we were trying to push $m$ as high or as low as possible while staying on $g = 1$.

And it turned out that such constrained maxima were precisely the points $(a, b) \in C$ at which $(\nabla f)(a, b)$ was a **scalar multiple** of $(\nabla g)(a, b)$.

**Example:** Consider

$$f(x, y) = xy + 1,$$

which has a **saddle point** at $(0, 0)$. Let the constraint function be

$$g(x, y) = x^2 + y^2.$$

![img](sad0.png)

We seek the **local extrema** of $f(x, y)$ subject to the constraint $g(x, y) = 1$, i.e. extrema of $f$ on the unit circle $x^2 + y^2 = 1$.

![img](sad1.png)
![img](sad2.png)

Let's try to visualize the constrained maxima and the contours.

![img](sad3.png)
![img](sad4.png)
![img](sad5.png)
![img](sad6.png)

Below is a top-down we're more familiar with where we have the $g(x, y) = 1$ level set as well as the level sets of $f(x, y)$.

![img](sad7.png)

Let's isolate just the constrained maxima contour line with $g(x, y) = 1$.

![img](sad8.png)
![img](sad9.png)

The tangent to $g(x, y) = 1$ and $f(x, y)$ are the same at the constrained maxima.

![img](sad10.png)

In comparison, the tangents drawn to $f(x, y)$ at points that are **not** the constrained extrema, are not parallel to the tangents drawn to $g(x, y) = 1$.

![img](sad11.png)
![img](sad12.png)

Following the principle from above, we look for points $(a, b)$ on the unit circle at which $(\nabla f)(a, b)$ is a scalar multiple of $(\nabla g)(a, b)$.

![img](sad13.png)

$$(\nabla f)(x, y) = \begin{pmatrix} y \\ x \end{pmatrix}, \qquad (\nabla g)(x, y) = \begin{pmatrix} 2x \\ 2y \end{pmatrix},$$

so the condition $(\nabla f)(a, b) = \lambda \, (\nabla g)(a, b)$ becomes the system

$$b = 2\lambda a, \qquad a = 2\lambda b, \qquad a^2 + b^2 = 1.$$

Substituting the first into the second gives $a = 2\lambda(2\lambda a) = 4\lambda^2 a$, i.e. $a(1 - 4\lambda^2) = 0$. So either $a = 0$ (which forces $b = 0$ from $b = 2\lambda a$, contradicting $a^2 + b^2 = 1$) or

$$\lambda^2 = \tfrac{1}{4}, \qquad \lambda = \pm \tfrac{1}{2}.$$

**Case $\lambda = \tfrac{1}{2}$.** Then $b = a$, and $a^2 + b^2 = 1$ gives $2a^2 = 1$, so $a = b = \pm \tfrac{1}{\sqrt{2}}$. The candidate points are

$$P_{\pm} = \left( \pm \tfrac{1}{\sqrt{2}}, \, \pm \tfrac{1}{\sqrt{2}} \right), \qquad f(P_{\pm}) = ab + 1 = \tfrac{1}{2} + 1 = \tfrac{3}{2}.$$

**Case $\lambda = -\tfrac{1}{2}$.** Then $b = -a$, and $2a^2 = 1$ again, so

$$Q_{\pm} = \left( \pm \tfrac{1}{\sqrt{2}}, \, \mp \tfrac{1}{\sqrt{2}} \right), \qquad f(Q_{\pm}) = ab + 1 = -\tfrac{1}{2} + 1 = \tfrac{1}{2}.$$

Since the unit circle is closed and bounded and $f$ is continuous, the constrained max and min are attained. Comparing values:

- **Constrained maxima** at $P_{\pm} = \left( \pm \tfrac{1}{\sqrt{2}}, \, \pm \tfrac{1}{\sqrt{2}} \right)$ with $f = \tfrac{3}{2}$.

![img](sad14.png)
![img](sad15.png)

- **Constrained minima** at $Q_{\pm} = \left( \pm \tfrac{1}{\sqrt{2}}, \, \mp \tfrac{1}{\sqrt{2}} \right)$ with $f = \tfrac{1}{2}$.

![img](sad16.png)
![img](sad17.png)

Note that none of these are critical points of $f$ in the ambient $\mathbb{R}^2$ (the only critical point of $f$ is the saddle $(0, 0)$, which does not even lie on the constraint circle). The Lagrange condition picked out the right points anyway.

**Drive home the point:** Imagine trying to find the highest point on a mountain — described by the elevation $f(x, y)$ — while restricted to walking along a specific trail $g(x, y) = c$. You are at the highest point on the trail exactly when the trail is **perfectly tangent** to the mountain's elevation contour lines at your location. If the trail were not tangent there, it would cut **across** a contour line, meaning a step in one direction along the trail would take you higher — so you could not have been at the highest point. The Lagrange condition $\nabla f = \lambda\, \nabla g$ is just the algebraic statement of this tangency.