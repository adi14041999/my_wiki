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

In comparison, the tangents drawn to $f(x, y)$ at points that are **not** the constrained extrema, are not coinciding to the tangents drawn to $g(x, y) = 1$.

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

## The method of Lagrange multipliers

!!! note "Theorem (Lagrange multipliers, one constraint)"
    Suppose $f : \mathbb{R}^n \to \mathbb{R}$ and $g : \mathbb{R}^n \to \mathbb{R}$ are functions, and consider the problem of finding a local maximum (or local minimum) of $f$ on the region where $g(\mathbf{x}) = c$. If a local extremum of $f$ on the constraint region (i.e. the set of $\mathbf{x}$ satisfying $g(\mathbf{x}) = c$) occurs at $\mathbf{a}$, then **either**

    $$(\nabla g)(\mathbf{a}) = \mathbf{0} \qquad \text{or} \qquad (\nabla f)(\mathbf{a}) = \lambda \, (\nabla g)(\mathbf{a})$$

    for some scalar $\lambda$ (called the **Lagrange multiplier**) that may depend on $\mathbf{a}$.

!!! warning "We do not know $\lambda$!"
    We do not know $\lambda$ in advance — we (often) need to solve for it.

    The equations $\nabla f = \lambda \, \nabla g$ in $\mathbb{R}^n$ and $g(\mathbf{x}) = c$ in $\mathbb{R}$ amount to $n + 1$ scalar equations in the $n + 1$ unknowns $x_1, \ldots, x_n, \lambda$, and we typically need to solve for all of them.

To summarize: any solution to an optimization problem must satisfy an **auxiliary equation** — vanishing of the gradient in the **unconstrained** case, or one of the options in the theorem above in the **constrained** setting. Solving this equation gives a list of **candidate** points for solutions of the original optimization problem. We then compare the values of $f$ at those candidates to see which are largest and smallest, much as we do with the first-derivative test in single-variable calculus.

**Remark:** As with the first-derivative test, the theorem identifies **candidates** for constrained local extrema. It can happen that there are **no** constrained global extrema.

**Example:** Find the point(s) on the curve

$$\{(x, y) \in \mathbb{R}^2 : 8y^2 - 4x^3 + x^4 = 0\}$$

closest to $P = (3, 0)$, and compute that minimal distance.

The quantity to minimize is the distance from $(x, y)$ to $P$:

$$\sqrt{(x - 3)^2 + y^2}.$$

It is the same as minimizing the quantity **inside** the square root, so we focus on minimizing

$$f(x, y) = (x - 3)^2 + y^2$$

subject to the constraint $g(x, y) = 0$, where

$$g(x, y) = 8y^2 - 4x^3 + x^4.$$

By the Lagrange multiplier method, any point $(x, y)$ at which $f$ attains a local extremum on $g = 0$ (either a local maximum or a local minimum — though we are seeking the **global** minimum) either makes $\nabla g$ vanish or makes $\nabla f$ a scalar multiple of $\nabla g$.

Let's first figure out where (if anywhere) $\nabla g$ vanishes on the constraint curve $g = 0$, so such **bad** points (corresponding to the first option in the Theorem) can be treated separately. We calculate

$$(\nabla g)(x, y) = \begin{pmatrix} -12x^2 + 4x^3 \\ 16y \end{pmatrix}.$$

For this to equal $\mathbf{0}$ we need $-12x^2 + 4x^3 = 0$ and $16y = 0$, so $y = 0$ and

$$0 = -12x^2 + 4x^3 = 4x^2(-3 + x).$$

This happens at the points $(0, 0)$ and $(3, 0) = P$. But $P$ is **not** on the constraint curve $g = 0$:

$$g(P) = g(3, 0) = -4 \cdot 27 + 81 = -108 + 81 = -27 \ne 0,$$

so in this case really only $(0, 0)$ occurs (and $g(0, 0) = 0$).

At any other point on the constraint curve we must be in the second option of the Theorem:

$$(\nabla f)(x, y) = \lambda \, (\nabla g)(x, y)$$

for some unknown scalar $\lambda$. We have computed $(\nabla g)(x, y)$ above, and

$$(\nabla f)(x, y) = \begin{pmatrix} 2(x - 3) \\ 2y \end{pmatrix},$$

so the Lagrange multiplier condition becomes

$$\begin{pmatrix} 2(x - 3) \\ 2y \end{pmatrix} = \lambda \begin{pmatrix} -12x^2 + 4x^3 \\ 16y \end{pmatrix} = \lambda \begin{pmatrix} 4x^2(-3 + x) \\ 16y \end{pmatrix}.$$

Equating corresponding vector entries and remembering the constraint equation, we want to find solutions to the combined system

$$
\begin{aligned}
2x - 6 &= \lambda \cdot 4x^2(-3 + x), \\
2y &= \lambda \cdot 16y, \\
8y^2 - 4x^3 + x^4 &= 0.
\end{aligned}
$$

At this point we carry out a very useful general technique: use each of the conditions **other than** the constraint equation to obtain different expressions for $\lambda$, which we then equate to get new conditions on the variables **without** involving $\lambda$. An extremely important point that one must always keep in mind to be systematic about this step is that one must always be careful about **division by zero** (i.e. avoid it!). To be more specific, the first and second equations in our combined system give expressions

$$
\frac{2x - 6}{4x^2(x - 3)} = \lambda = \frac{2y}{16y} = \frac{1}{8}
$$

assuming the denominators $4x^2(x - 3)$ and $16y$ are both nonzero. The left side is $1/(2x^2)$ upon cancelling $x - 3$, provided that $x \ne 3$, as is necessary for the original fraction to make sense.

It is always good to first determine when one of those denominators vanishes. In some such cases there might not be a $\lambda$ satisfying all conditions, but don't worry about it. We handle that now:

**Case 1.** The equation $4x^2(x - 3) = 0$ holds when $x = 0$ or $x = 3$. If $x = 3$, the first equation becomes $2x - 6 = 0$, which is consistent. The constraint $g(x, y) = 0$ becomes $g(3, y) = 0$, which says $8y^2 - 27 = 0$. Hence we obtain the problematic points $(3, \pm \sqrt{27/8})$.

If instead $x = 0$, the first equation reads $2x - 6 = -6 = \lambda \cdot 4x^2(x - 3) = \lambda \cdot 0$, which is impossible for any finite $\lambda$. So no new Lagrange candidates arise from the $x = 0$ branch of $4x^2(x - 3) = 0$. The point $(0, 0)$ on $g = 0$ already appeared where $(\nabla g)(0, 0) = \mathbf{0}$, not from forcing the first equation in this way.

**Case 2.** The case $16y = 0$, or equivalently $y = 0$, makes the second equation tell us nothing, but the constraint $g(x, y) = 0$ says $g(x, 0) = 0$, or in other words $-4x^3 + x^4 = 0$. Since $-4x^3 + x^4 = x^3(x - 4)$, this makes $x = 0$ or $x = 4$, yielding $(0, 0)$ and $(4, 0)$.

To summarize, so far we have obtained the points $(0, 0)$, $(4, 0)$, and $(3, \pm \sqrt{27/8})$ that merit separate treatment, and otherwise we have the two expressions $1/(2x^2)$ and $1/8$ for $\lambda$ (assuming the non-vanishing of the denominators). Equating these two fractional expressions for $\lambda$ gives

$$\frac{1}{2x^2} = \frac{1}{8},$$

which is to say $x^2 = 4$, i.e. $x = \pm 2$. The constraint curve $g(x, y) = 0$ says $g(2, y) = 0$ when $x = 2$ and $g(-2, y) = 0$ when $x = -2$. Since

$$g(2, y) = 8y^2 - 4(2)^3 + 2^4 = 8y^2 - 16$$

and

$$g(-2, y) = 8y^2 - 4(-2)^3 + (-2)^4 = 8y^2 + 48,$$

the latter never vanishes (so the case $x = -2$ does not occur!) and the former vanishes when $y^2 = 2$, which is to say $y = \pm \sqrt{2}$. Hence, we obtain the additional candidate points $(2, \pm \sqrt{2})$ for local extrema of $f$ on the constraint curve.

Putting it all together, we have six points to examine: $(2, \pm \sqrt{2})$, $(3, \pm \sqrt{27/8})$, $(0, 0)$, $(4, 0)$. Evaluating $f$ at these points, we get

$$f(2, \pm \sqrt{2}) = (2 - 3)^2 + (\pm \sqrt{2})^2 = 1 + 2 = 3,$$

$$f(3, \pm \sqrt{27/8}) = 0^2 + \left(\pm \sqrt{27/8}\right)^2 = \frac{27}{8}, \qquad f(0, 0) = 9, \qquad f(4, 0) = 1.$$

The smallest of these values is $1$, attained at $(4, 0)$, and the largest is $9$, attained at $(0, 0)$. So the closest point to $P = (3, 0)$ on the constraint curve $g = 0$ is $(4, 0)$ with distance $\sqrt{1} = 1$, and the farthest point to $P$ on the constraint curve is $(0, 0)$ with distance $\sqrt{9} = 3$.

![img](pinch.png)

The figure above shows what the constraint curve $g = 0$ looks like, and this makes rather visible that $(4, 0)$ is the point on this curve nearest to $(3, 0)$, and that $(0, 0)$ is the point on this curve farthest from $(3, 0)$ (the pinching of the curve $g = 0$ at $(0, 0)$ is related to the fact that $\nabla g$ vanishes there).

!!! warning "Remark"
    In the preceding example, although we were seeking point(s) on a curve $g(x, y) = 0$ closest to a given point $P$ not on the curve, the method also yielded a unique point on the curve at the **largest** distance from $P$. The notable feature is that this point at maximal distance was the point $(0, 0)$, which emerged in our analysis in the situation $(\nabla g)(0, 0) = \mathbf{0}$, and at this point $(\nabla f)(0, 0)$ is equal to

    $$\begin{pmatrix} -6 \\ 0 \end{pmatrix} \ne \mathbf{0},$$

    so there is **no** scalar $\lambda$ making $(\nabla f)(0, 0) = \lambda \, (\nabla g)(0, 0)$.

    This illustrates that the first case really **can** occur at a solution to a constrained optimization problem (such as a point at maximal distance): in the Lagrange multiplier theorem, the solution might be completely missed by the multiplier condition “$\nabla f = \lambda \, \nabla g$” and only captured by the other possibility $(\nabla g) = \mathbf{0}$ in the Theorem. Hence, the first option in the Theorem really must **never** be disregarded; it could be the only part of the method that actually finds the solution!

The multiplier $\lambda$ often has an **interpretation** in specific applications. In an economics application it can mean **marginal cost**; in a chemistry problem with **three** constraints and hence three multipliers, two of the multipliers might encode **temperature** and **pressure** (depending on how the model is set up).

When seeking local extrema for $f : \mathbb{R}^n \to \mathbb{R}$, we have introduced the concept of a **critical point**. We will later talk about a multivariable **second derivative test** that gives a way to check whether a critical point is a local maximum or a local minimum. Since the Lagrange multiplier equation is the substitute for the notion of critical point in the context of constrained optimization, it is natural to wonder whether there is a version of the multivariable second derivative test that can be used to check whether a solution to the Lagrange multiplier equation is a local maximum or a local minimum on the constraint region. There is indeed such a result.

## Why does the method of Lagrange multipliers work?

To explain the key idea behind the Theorem, we first note that if $(\nabla g)(\mathbf{a}) = \mathbf{0}$ then there is nothing to do. Hence, we may and do now focus on the case $(\nabla g)(\mathbf{a}) \ne \mathbf{0}$. We also focus on the case of local maxima; the case of local minima goes the same way (or apply the case of local maxima to $-f$ in place of $f$).

To convey the main idea with a minimum of fuss, we shall consider the special case $n = 2$: functions $f$ and $g$ on $\mathbb{R}^2$, so we may visualize the level set $g(\mathbf{x}) = c$ near $\mathbf{a}$ as a curve in $\mathbb{R}^2$. The reasoning for functions on $\mathbb{R}^n$ goes similarly when $n > 2$, but the geometry then becomes a bit more involved, since the level set $g(\mathbf{x}) = c$ in $\mathbb{R}^n$ near $\mathbf{a}$ is not a curve but rather an $(n - 1)$-dimensional **hypersurface**; e.g. for $n = 3$ it is a surface in $\mathbb{R}^3$.

By assumption, the point $\mathbf{a}$ on the level set $g(\mathbf{x}) = c$ is one at which $f$ attains a **local maximum**. For instance, if we think of $f(\mathbf{x})$ as the temperature at the point $\mathbf{x}$, then $\mathbf{a}$ is a point on the curve $g(\mathbf{x}) = c$ where the temperature is at least as hot as at all nearby points on the curve.

Imagine an insect crawling along the curve $g(\mathbf{x}) = c$ with **nonzero** velocity, and suppose that at time $t = 0$ it is at the point $\mathbf{a}$. If we let

$$\mathbf{p}(t) = \begin{pmatrix} x(t) \\ y(t) \end{pmatrix} \in \mathbb{R}^2$$

be the position of the insect at time $t$, then $\mathbf{p}(0) = \mathbf{a}$. Now $f(\mathbf{p}(t))$ is the temperature of the insect's location at time $t$, so at time $0$ the insect is at a point at least as hot as all nearby points (because that is when it is at $\mathbf{a}$). Since the function $f(\mathbf{p}(t))$ therefore has a local maximum at $t = 0$, by single-variable calculus we know that

$$\left.\frac{d}{dt}\, f(\mathbf{p}(t))\right|_{t = 0} = 0.$$

The composite $f(\mathbf{p}(t))$ is a composition of functions $\mathbf{p} : \mathbb{R} \to \mathbb{R}^2$ and $f : \mathbb{R}^2 \to \mathbb{R}$, where the intermediate step lies in $\mathbb{R}^2$ rather than $\mathbb{R}$, so it does not fit the usual scalar-composition paradigm: the inner function $\mathbf{p}$ is **vector-valued**, not scalar-valued. This is a new situation that we have never encountered before!

In the special case of composing functions $\mathbf{p} : \mathbb{R} \to \mathbb{R}^n$ and $f : \mathbb{R}^n \to \mathbb{R}$, this yields a formula for the calculus derivative in terms of a **dot product** of the **gradient** of $f$ against the **velocity** of $\mathbf{p}$ (we'll see the concepts and derivations of these later):

$$\frac{d}{dt}\, f(\mathbf{p}(t)) = (\nabla f)(\mathbf{p}(t)) \cdot \mathbf{p}'(t),$$

where $\mathbf{p}'(t)$ denotes the velocity

$$\mathbf{p}'(t) = \begin{pmatrix} x'(t) \\ y'(t) \end{pmatrix}$$

of the insect at time $t$ (in the plane, $n = 2$; the same pattern holds in $\mathbb{R}^n$ with $n$ components). Setting $t = 0$ in the chain-rule identity gives the reformulation

$$(\nabla f)(\mathbf{a}) \cdot \mathbf{p}'(0) = 0.$$

The nonzero velocity $\mathbf{p}'(0)$ at time $t = 0$ is **tangent** to the level curve $g(\mathbf{x}) = c$ at $\mathbf{p}(0) = \mathbf{a}$, since the insect is crawling along this level curve (and the velocity of a moving particle or insect is always tangent to the path of motion).

Consequently, the equation $(\nabla f)(\mathbf{a}) \cdot \mathbf{p}'(0) = 0$ means that $(\nabla f)(\mathbf{a})$ is **perpendicular** to the tangent line to the curve $g(\mathbf{x}) = c$ at the point $\mathbf{a}$.

The gradient vector $(\nabla g)(\mathbf{a})$ is also **normal** (perpendicular) to the level curve $g(\mathbf{x}) = c$ at $\mathbf{a}$. In the plane, the tangent line to that curve at $\mathbf{a}$ is one-dimensional, so the vectors in $\mathbb{R}^2$ perpendicular to the tangent line form a **one-dimensional** subspace: the **normal directions** at $\mathbf{a}$. Both $(\nabla f)(\mathbf{a})$ and $(\nabla g)(\mathbf{a})$ lie in that subspace. Since $(\nabla g)(\mathbf{a}) \ne \mathbf{0}$, it **spans** that entire subspace. Hence $(\nabla f)(\mathbf{a})$ must be a scalar multiple of $(\nabla g)(\mathbf{a})$:

$$(\nabla f)(\mathbf{a}) = \lambda \, (\nabla g)(\mathbf{a})$$

for some scalar $\lambda$.

**Intuition:** At a constrained local maximum of $f$ on $g(\mathbf{x}) = c$, you cannot improve $f$ by **sliding** along the constraint. Along a smooth **curve** (the usual plane picture $g(x,y)=c$), there is only **one tangent line** at a point. The way $f$ changes as you start to move along the curve must be **zero** at a maximum. If it were nonzero, a short step one way or the other along the trail would increase $f$. So $\nabla f$ has **no component along the tangent**. Hence $\nabla f$ is **orthogonal to that tangent line**, i.e. it lies in the normal line. The gradient $\nabla g$ is also normal to the level set $g = c$. In the plane the normal line is one-dimensional, so $\nabla f$ and $\nabla g$ are **parallel**: $\nabla f = \lambda\, \nabla g$. The proof above makes this precise by following an insect along the curve and applying single-variable calculus plus the chain rule.

## Exercises

**1.** Let $c \in \mathbb{R}$ be a constant, and let

$$h(x, y, z) = \sqrt{\dfrac{x^2 + y^2 + z^2}{3}}.$$

**(a)** Use Lagrange multipliers to generate a list of candidate extrema of $h$ on the plane

$$x + y + z = 3c.$$

**Solution:** Since $h(x, y, z) \geq 0$ on all of $\mathbb{R}^3$ and $u \mapsto u^2$ is strictly increasing on $[0, \infty)$, the inequalities $h(\mathbf{p}) \leq h(\mathbf{q})$ and $h(\mathbf{p})^2 \leq h(\mathbf{q})^2$ are equivalent. So the maxima and minima of $h$ on the plane occur at the same points as the maxima and minima of

$$\phi(x, y, z) := h(x, y, z)^2 = \dfrac{x^2 + y^2 + z^2}{3},$$

and $\phi$ is smooth everywhere (no square root to worry about at the origin).

Set up the Lagrange equation for $\phi$ with constraint $g(x, y, z) := x + y + z - 3c = 0$. We have

$$\nabla \phi = \begin{pmatrix} 2x/3 \\ 2y/3 \\ 2z/3 \end{pmatrix}, \qquad \nabla g = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} \ne \mathbf{0}.$$

Since $\nabla g \ne \mathbf{0}$ everywhere, the only case to consider is $\nabla \phi = \lambda \, \nabla g$ for some scalar $\lambda$:

$$\dfrac{2x}{3} = \lambda, \qquad \dfrac{2y}{3} = \lambda, \qquad \dfrac{2z}{3} = \lambda.$$

These force $x = y = z$. Combined with $x + y + z = 3c$, we get $3x = 3c$, so

$$x = y = z = c.$$

The list of candidates therefore consists of the **single point** $(c, c, c)$.

**(b)** Use the fact that $h$ is non-negative (bounded below) and $h \to \infty$ as $\|(x, y, z)\| \to \infty$, plus your result from (a), to find the minimum value of $h$ on the plane $x + y + z = 3c$ when $c \geq 0$.

**Solution:** The hypotheses $h \geq 0$ and $h \to \infty$ at infinity guarantee that $h$ attains a global minimum on the closed plane $x + y + z = 3c$. By Lagrange's Theorem, that global minimum must occur at a candidate point from (a). The only candidate is $(c, c, c)$, so the minimum is achieved there. Compute:

$$h(c, c, c) = \sqrt{\dfrac{c^2 + c^2 + c^2}{3}} = \sqrt{c^2} = |c| = c \quad (\text{since } c \geq 0).$$

So the **minimum value** of $h$ on the plane $x + y + z = 3c$ is $c$, attained at $(c, c, c)$.

**(c)** Explain why your result from (b) implies the inequality

$$\sqrt{\dfrac{x^2 + y^2 + z^2}{3}} \;\geq\; \dfrac{x + y + z}{3}$$

for all $x, y, z \in \mathbb{R}$.

**Solution:** Fix any $\mathbf{v} = (x, y, z) \in \mathbb{R}^3$, and define

$$c := \dfrac{x + y + z}{3}.$$

By construction, $\mathbf{v}$ lies on the plane $x + y + z = 3c$.

- **Case $c \geq 0$.** By (b), the minimum of $h$ on this plane is $c$, so

    $$h(\mathbf{v}) \geq c = \dfrac{x + y + z}{3},$$

    which is exactly the desired inequality.

- **Case $c < 0$.** Then $\dfrac{x + y + z}{3} = c < 0$, while $h(\mathbf{v}) \geq 0$, so

    $$h(\mathbf{v}) \geq 0 > c = \dfrac{x + y + z}{3}.$$

Either way,

$$\sqrt{\dfrac{x^2 + y^2 + z^2}{3}} \;\geq\; \dfrac{x + y + z}{3}.$$

The left side is the **root mean square** (or **quadratic mean**) of $x, y, z$; the right side is the **arithmetic mean**. This is the QM–AM inequality for three real numbers.

**2.** Let $f(x, y, z) = xz + yz$.

**(a)** Using Lagrange multipliers, show that any global extrema of $f$ on the surface

$$x^2 + y^2 - 4z^2 = 1$$

(a **hyperboloid**, since cutting it by any plane $y = c$ gives a hyperbola) must be among **two** specific points. Be careful about division by $0$, and account for the possibility that $\nabla g = \mathbf{0}$.

**Solution:** Write the constraint as $g(x, y, z) := x^2 + y^2 - 4z^2 - 1 = 0$. Compute

$$\nabla f = \begin{pmatrix} z \\ z \\ x + y \end{pmatrix}, \qquad \nabla g = \begin{pmatrix} 2x \\ 2y \\ -8z \end{pmatrix}.$$

**Check $\nabla g = \mathbf{0}$ first.** $\nabla g = \mathbf{0}$ requires $x = y = z = 0$, but then $g(0, 0, 0) = -1 \ne 0$. So $\nabla g \ne \mathbf{0}$ at every point of the surface, and Lagrange's Theorem requires $\nabla f = \lambda \, \nabla g$:

$$z = 2\lambda x, \qquad z = 2\lambda y, \qquad x + y = -8\lambda z. \qquad (\ast)$$

Subtracting the first two equations gives $2\lambda (x - y) = 0$, so either $\lambda = 0$ or $x = y$.

**Case 1: $\lambda = 0$.** The first two equations of $(\ast)$ give $z = 0$, and the third gives $x + y = 0$, so $y = -x$. The constraint becomes

$$x^2 + (-x)^2 - 0 = 2x^2 = 1 \quad \Longrightarrow \quad x = \pm \tfrac{1}{\sqrt{2}}.$$

This yields two candidate points

$$P_1 = \left(\tfrac{1}{\sqrt{2}},\, -\tfrac{1}{\sqrt{2}},\, 0\right), \qquad P_2 = \left(-\tfrac{1}{\sqrt{2}},\, \tfrac{1}{\sqrt{2}},\, 0\right),$$

both with $f = z(x + y) = 0$.

**Case 2: $\lambda \ne 0$ and $x = y$.** The third equation in $(\ast)$ becomes $2x = -8\lambda z$, i.e. $x = -4\lambda z$. Substituting into the first equation,

$$z = 2\lambda x = 2\lambda \cdot (-4\lambda z) = -8\lambda^2 z \quad \Longrightarrow \quad z\,(1 + 8\lambda^2) = 0.$$

Since $1 + 8\lambda^2 > 0$, we must have $z = 0$. Then the first equation gives $0 = 2\lambda x$, and since $\lambda \ne 0$, we get $x = 0$, hence $y = 0$. But $(0, 0, 0)$ is not on the constraint surface ($g(0,0,0) = -1$), so this case yields **no** candidates.

So the only candidates from Lagrange's method are the two points $P_1$ and $P_2$, and any global extremum of $f$ on the hyperboloid must be one of them.

**(b)** Evaluate $f$ at both points from (a) and exhibit a point on the hyperboloid where $f$ is **larger** than those values, and another where $f$ is **smaller**. Why does this imply $f$ has no global extrema on the hyperboloid?

**Solution:** From (a), $f(P_1) = f(P_2) = 0$.

Try $x = y = 1$. The constraint becomes $1 + 1 - 4z^2 = 1$, so $z^2 = \tfrac{1}{4}$, giving $z = \pm \tfrac{1}{2}$. Both $(1, 1, \tfrac{1}{2})$ and $(1, 1, -\tfrac{1}{2})$ lie on the hyperboloid, and

$$f\!\left(1, 1, \tfrac{1}{2}\right) = \tfrac{1}{2} \cdot (1 + 1) = 1 > 0, \qquad f\!\left(1, 1, -\tfrac{1}{2}\right) = -\tfrac{1}{2} \cdot 2 = -1 < 0.$$

So we have a point on the surface where $f = 1$ and another where $f = -1$, both **outside** the candidate set $\{P_1, P_2\}$ (where $f = 0$).

If $f$ had a global maximum on the hyperboloid, Lagrange's Theorem (combined with $\nabla g \ne \mathbf{0}$) would force the maximizer to be among $\{P_1, P_2\}$, with maximum value $0$. But $f(1, 1, \tfrac{1}{2}) = 1 > 0$ contradicts that. So $f$ has **no global maximum**. By the same reasoning with $f(1, 1, -\tfrac{1}{2}) = -1 < 0$, $f$ has **no global minimum** either.

!!! note "Are $P_1$ and $P_2$ local extrema?"
    **No**. They are **saddle points** of $f$ on the surface. Lagrange's Theorem only says that any global (or local) extremum must satisfy the multiplier equation; it does **not** say every solution of that equation is even a local extremum.

**3.** It is a fact that the function

$$f(x, y) = e^{-x - 2y}$$

restricted to the level set

$$x^2 - y^2 = -3$$

in the region $y < 0$ attains a minimal value. Moreover, the value is attained at exactly one point. Find that unique point and the value of $f$ there by using Lagrange multipliers in two ways:

**(a)** Work with $f(x, y) = e^{-x - 2y}$.

**Solution:** Write the constraint as

$$g(x, y) := x^2 - y^2 + 3 = 0,$$

together with the branch condition $y < 0$. We have

$$\nabla f = e^{-x - 2y}\begin{pmatrix} -1 \\ -2 \end{pmatrix}, \qquad \nabla g = \begin{pmatrix} 2x \\ -2y \end{pmatrix}.$$

First check whether $\nabla g = \mathbf{0}$ on the constraint. This would require $x = 0$ and $y = 0$, but

$$g(0, 0) = 3 \ne 0.$$

So $\nabla g \ne \mathbf{0}$ everywhere on the constraint, and Lagrange's Theorem requires

$$\nabla f = \lambda \, \nabla g.$$

Thus

$$
\begin{aligned}
-e^{-x - 2y} &= 2\lambda x, \\
-2e^{-x - 2y} &= -2\lambda y, \\
x^2 - y^2 &= -3.
\end{aligned}
$$

Since $e^{-x - 2y} > 0$, the first equation implies $x \ne 0$ and $\lambda \ne 0$. Therefore we may divide the second equation by the first equation:

$$\dfrac{-2e^{-x - 2y}}{-e^{-x - 2y}} = \dfrac{-2\lambda y}{2\lambda x}.$$

This gives

$$2 = -\dfrac{y}{x},$$

so

$$y = -2x.$$

Substitute this into the constraint:

$$x^2 - (-2x)^2 = -3 \quad \Longrightarrow \quad x^2 - 4x^2 = -3.$$

Hence

$$-3x^2 = -3, \qquad x^2 = 1.$$

So $x = \pm 1$. Since $y = -2x$ and we are restricted to the branch $y < 0$, we must choose $x = 1$, giving

$$y = -2.$$

Thus, the only Lagrange candidate on the branch $y < 0$ is

$$P = (1, -2).$$

By the hypothesis in the problem, the minimum is attained at exactly one point. Therefore this candidate is the unique minimizer. The minimum value is

$$f(1, -2) = e^{-1 - 2(-2)} = e^3.$$

So the unique point is $(1, -2)$, and the minimum value is $e^3$.

**(b)** Work with $h(x, y) = -x - 2y$ and use the fact that an inequality $e^a \leq e^b$ holds for numbers $a$ and $b$ precisely when $a \leq b$.

**Solution:** Since the exponential function is increasing, minimizing

$$f(x, y) = e^{-x - 2y}$$

on the constraint is equivalent to minimizing

$$h(x, y) = -x - 2y$$

on the same constraint. Again use

$$g(x, y) := x^2 - y^2 + 3 = 0.$$

We have

$$\nabla h = \begin{pmatrix} -1 \\ -2 \end{pmatrix}, \qquad \nabla g = \begin{pmatrix} 2x \\ -2y \end{pmatrix}.$$

As in part (a), $\nabla g = \mathbf{0}$ does not occur on the constraint. Hence any constrained extremum must satisfy

$$\nabla h = \lambda \, \nabla g,$$

so

$$
\begin{aligned}
-1 &= 2\lambda x, \\
-2 &= -2\lambda y, \\
x^2 - y^2 &= -3.
\end{aligned}
$$

The first equation implies $x \ne 0$ and $\lambda \ne 0$. Dividing the second equation by the first equation gives

$$\dfrac{-2}{-1} = \dfrac{-2\lambda y}{2\lambda x},$$

so

$$2 = -\dfrac{y}{x}, \qquad \text{hence} \qquad y = -2x.$$

The constraint then gives

$$x^2 - (-2x)^2 = -3,$$

so again

$$x^2 = 1.$$

The branch condition $y < 0$ rules out $x = -1$ (which would give $y = 2$), so the unique candidate on the desired branch is

$$P = (1, -2).$$

At this point,

$$h(1, -2) = -1 - 2(-2) = 3.$$

Since $f = e^h$, the corresponding value of the original function is

$$f(1, -2) = e^{h(1, -2)} = e^3.$$

So this second method gives the same answer: the unique minimizing point is $(1, -2)$, and the minimum value of $f$ is $e^3$.

**4.** Consider the equation

$$126z + 9yz^3 - y^2 + xy - 7x^2 = 0.$$

Solving for $z$ is tantamount to expressing $z$ as a function of $x$ and $y$, but it is not clear how one would go about doing this (since terms with $z^3$ and $z$ both appear). Thus, one says that the equation gives $z$ as an implicit (as opposed to explicit) function of $x, y$. Use Lagrange multipliers to find all local extrema of $z$ within the set of points satisfying the equation.

!!! hint
    It is a fact, which you may accept, that the candidate extrema produced by the Lagrange multiplier method are indeed local extrema in this case. Apply this with $f(x, y, z) = z$.

**Solution:** Let

$$G(x, y, z) := 126z + 9yz^3 - y^2 + xy - 7x^2.$$

We want to optimize

$$f(x, y, z) = z$$

subject to the constraint $G(x, y, z) = 0$. Compute

$$\nabla f = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix},$$

and

$$\nabla G = \begin{pmatrix} y - 14x \\ 9z^3 - 2y + x \\ 126 + 27yz^2 \end{pmatrix}.$$

First check the possible exceptional case $\nabla G = \mathbf{0}$ on the constraint. The equations

$$y - 14x = 0, \qquad 9z^3 - 2y + x = 0, \qquad 126 + 27yz^2 = 0$$

give

$$y = 14x, \qquad x = \dfrac{z^3}{3}, \qquad y = \dfrac{14z^3}{3}.$$

Substituting $y = \dfrac{14z^3}{3}$ into the third equation gives

$$126 + 27 \cdot \dfrac{14z^3}{3} \cdot z^2 = 126 + 126z^5 = 126(1 + z^5) = 0,$$

so $z = -1$. This would give

$$x = -\dfrac{1}{3}, \qquad y = -\dfrac{14}{3}.$$

But this point does **not** satisfy the constraint:

$$G\!\left(-\dfrac{1}{3}, -\dfrac{14}{3}, -1\right) = -105 \ne 0.$$

So there is no point on the constraint where $\nabla G = \mathbf{0}$, and the extrema must satisfy the Lagrange multiplier equation

$$\nabla f = \lambda \, \nabla G.$$

This gives the system

$$
\begin{aligned}
0 &= \lambda(y - 14x), \\
0 &= \lambda(9z^3 - 2y + x), \\
1 &= \lambda(126 + 27yz^2), \\
126z + 9yz^3 - y^2 + xy - 7x^2 &= 0.
\end{aligned}
$$

The third equation implies $\lambda \ne 0$. Hence the first two equations force

$$y - 14x = 0, \qquad 9z^3 - 2y + x = 0.$$

From the first equation,

$$y = 14x.$$

Substitute this into the second equation:

$$9z^3 - 2(14x) + x = 0,$$

so

$$9z^3 - 27x = 0, \qquad x = \dfrac{z^3}{3}.$$

Therefore

$$y = 14x = \dfrac{14z^3}{3}.$$

Now substitute

$$x = \dfrac{z^3}{3}, \qquad y = \dfrac{14z^3}{3}$$

into the constraint:

$$
\begin{aligned}
0
&= 126z + 9\left(\dfrac{14z^3}{3}\right)z^3 - \left(\dfrac{14z^3}{3}\right)^2
    + \left(\dfrac{z^3}{3}\right)\left(\dfrac{14z^3}{3}\right)
    - 7\left(\dfrac{z^3}{3}\right)^2 \\
&= 126z + 42z^6 - \dfrac{196z^6}{9} + \dfrac{14z^6}{9} - \dfrac{7z^6}{9} \\
&= 126z + 21z^6 \\
&= 21z(z^5 + 6).
\end{aligned}
$$

Thus

$$z = 0 \qquad \text{or} \qquad z^5 = -6.$$

So

$$z = 0 \qquad \text{or} \qquad z = -\sqrt[5]{6}.$$

Using $x = z^3/3$ and $y = 14z^3/3$, we get the two candidate points

$$P_0 = (0, 0, 0)$$

and

$$P_1 = \left(-\dfrac{6^{3/5}}{3},\, -\dfrac{14 \cdot 6^{3/5}}{3},\, -6^{1/5}\right).$$

**5.** Maximize

$$f(x, y, z) = 2x^2 + 6xy + y^2 + 2z$$

on the solid region $R$ defined by

$$x, y, z \ge 0, \qquad x + y + z \le 1.$$

$R$ is a pyramid with four triangular sides. One side is the triangle $T$ in the plane $x + y + z = 1$ with $x, y, z \ge 0$; the other three lie in the coordinate planes. It is a fact that $f$ attains a maximum on $R$.

![img](pyramid.png)

**(a)** Show $f$ has no critical points on the interior of $R$ (where $x, y, z > 0$ and $x + y + z < 1$).

**Solution:** The gradient is

$$\nabla f = \begin{pmatrix} \partial f/\partial x \\ \partial f/\partial y \\ \partial f/\partial z \end{pmatrix} = \begin{pmatrix} 4x + 6y \\ 6x + 2y \\ 2 \end{pmatrix}.$$

The third component is $2 \ne 0$, so $\nabla f \ne \mathbf{0}$ at **every** point of $\mathbb{R}^3$. Hence $f$ has no critical points anywhere, in particular none on the interior of $R$. **Since the maximum exists and cannot occur at an interior critical point, it must be attained on one of the four sides of $R$.**

**(b)** Use Lagrange multipliers to find all candidate extrema on the interior of $T$ (where $x + y + z = 1$ and $x, y, z > 0$).

**Solution:** With constraint $g(x, y, z) := x + y + z - 1 = 0$ we have $\nabla g = (1, 1, 1) \ne \mathbf{0}$, so the only case is $\nabla f = \lambda \, \nabla g$:

$$4x + 6y = \lambda, \qquad 6x + 2y = \lambda, \qquad 2 = \lambda.$$

The third equation gives $\lambda = 2$, so

$$4x + 6y = 2 \;\Rightarrow\; 2x + 3y = 1, \qquad 6x + 2y = 2 \;\Rightarrow\; 3x + y = 1.$$

From the second, $y = 1 - 3x$; substituting into the first,

$$2x + 3(1 - 3x) = 1 \;\Rightarrow\; -7x = -2 \;\Rightarrow\; x = \tfrac{2}{7}, \quad y = \tfrac{1}{7}, \quad z = 1 - x - y = \tfrac{4}{7}.$$

All coordinates are positive, so the single candidate on the interior of $T$ is

$$P = \left(\tfrac{2}{7}, \tfrac{1}{7}, \tfrac{4}{7}\right), \qquad f(P) = \dfrac{8 + 12 + 1 + 56}{49} = \dfrac{77}{49} = \dfrac{11}{7}.$$

**(c)** Express the restriction of $f$ to the edge $L = \{(t, 1 - t, 0) : 0 \le t \le 1\}$ as a function of $t$ and find its maximum.

**Solution:** Substituting $x = t$, $y = 1 - t$, $z = 0$,

$$
\begin{aligned}
f(t) &= 2t^2 + 6t(1 - t) + (1 - t)^2 + 0 \\
&= 2t^2 + 6t - 6t^2 + 1 - 2t + t^2 \\
&= -3t^2 + 4t + 1.
\end{aligned}
$$

Then $f'(t) = -6t + 4 = 0$ gives $t = \tfrac{2}{3}$, an interior critical point of the segment. Comparing with the endpoints,

$$f(0) = 1, \qquad f(1) = 2, \qquad f\!\left(\tfrac{2}{3}\right) = -3 \cdot \tfrac{4}{9} + \tfrac{8}{3} + 1 = \dfrac{7}{3}.$$

So the maximum of $f$ on $L$ is $\dfrac{7}{3}$, attained at $\left(\tfrac{2}{3}, \tfrac{1}{3}, 0\right)$.

**(d)** Assuming the maximum of $f$ on $R$ is attained either at the interior candidate of $T$ from (b) or on the edge $L$ from (c), find the maximum value.

**Solution:** Comparing the two candidate values,

$$f(P) = \dfrac{11}{7} \approx 1.571, \qquad \max_L f = \dfrac{7}{3} \approx 2.333,$$

the larger is $\dfrac{7}{3}$. Hence

$$\max_R f = \dfrac{7}{3}, \quad \text{attained at } \left(\tfrac{2}{3}, \tfrac{1}{3}, 0\right).$$

