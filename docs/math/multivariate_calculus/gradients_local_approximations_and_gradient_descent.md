# Gradients, local approximations, and gradient descent

## The Gradient

For a scalar-valued function $f : \mathbb{R}^n \to \mathbb{R}$, we now package all of its partial derivatives into a single vector-valued function denoted $\nabla f : \mathbb{R}^n \to \mathbb{R}^n$ called the **gradient** of $f$.

Consider $f : \mathbb{R}^n \to \mathbb{R}$. We will sometimes write the inputs to $f$ as vectors. For example, when $n = 2$ we may think of $f$ as a function of a column vector $\begin{pmatrix} x \\ y \end{pmatrix}$ (often written $f(x,y)$ with the same meaning once we identify $(x,y)$ with that vector).

The **gradient** of $f$ is defined to be the column vector of partial derivatives:

$$
\nabla f =
\begin{pmatrix}
\dfrac{\partial f}{\partial x_1} \\[0.35em]
\dfrac{\partial f}{\partial x_2} \\[0.35em]
\vdots \\[0.2em]
\dfrac{\partial f}{\partial x_n}
\end{pmatrix}.
$$

**Example:** Let $f(x,y) = x^2 + y^2$. Then $\dfrac{\partial f}{\partial x} = 2x$ and $\dfrac{\partial f}{\partial y} = 2y$, so

$$
\nabla f(x,y) =
\begin{pmatrix}
2x \\[0.25em]
2y
\end{pmatrix}.
$$

For instance $\nabla f(1, -2) = \begin{pmatrix} 2 \\ -4 \end{pmatrix}$.

**Example:** Let $f(x,y,z) = x^2 + xy + yz^2$. Compute

$$
\frac{\partial f}{\partial x} = 2x + y, \qquad
\frac{\partial f}{\partial y} = x + z^2, \qquad
\frac{\partial f}{\partial z} = 2yz.
$$

Hence

$$
\nabla f(x,y,z) =
\begin{pmatrix}
2x + y \\[0.25em]
x + z^2 \\[0.25em]
2yz
\end{pmatrix}.
$$

## The linear approximation for a scalar-valued function

For a function $f$ of a single variable $x$, we know that a small change $h$ in the value of $x$ near $x = a$ causes an approximate change of $f'(a)\,h$ in the value of $f(x)$ near $x = a$:

$$f(a + h) \approx f(a) + f'(a)\,h$$

for all $h$ near $0$. We can write this in another way: for $x$ very close to $x = a$,

$$f(x) \approx f(a) + f'(a)\,(x - a),$$

which is the same statement of approximation except with $x$ in place of $a + h$ (so $x - a = h$ is small).

Note that the gradient of $f : \mathbb{R}^n \to \mathbb{R}$ is a vector-valued function $\nabla f : \mathbb{R}^n \to \mathbb{R}^n$: its value $(\nabla f)(\mathbf{a})$ at $\mathbf{a} \in \mathbb{R}^n$ is an $n$-vector. For $\mathbf{x}$ near $\mathbf{a} \in \mathbb{R}^n$, the **linear approximation** to $f$ is

$$f(\mathbf{x}) \approx f(\mathbf{a}) + (\nabla f)(\mathbf{a}) \cdot (\mathbf{x} - \mathbf{a}).$$

Observe that this looks just like the single-variable case except that now there are vectors and a dot product involved. When $n = 1$ this recovers exactly the familiar single-variable case, with $(\nabla f)(a)$ equal to the $1$-vector $[f'(a)] \in \mathbb{R}^1 = \mathbb{R}$.

Let us write it out explicitly in the case of a two-variable function (i.e., $n = 2$) without the vector notation:

$$
f(x, y) \approx f(a, b) + \underbrace{f_x(a, b)(x - a) + f_y(a, b)(y - b)}_{\displaystyle (\nabla f)(a,b) \cdot \begin{pmatrix} x - a \\ y - b \end{pmatrix}}
$$

for $(x, y)$ near $(a, b)$. Here, the underbraced term is exactly $(\nabla f)(\mathbf{a}) \cdot (\mathbf{x} - \mathbf{a})$ with $\mathbf{a} = (a,b)$ and $\mathbf{x} = (x,y)$.

We refer to these equations as the **local approximation** for $f$ at $\mathbf{a}$ or as the **linear approximation** for $f$ at $\mathbf{a}$.

**Example:** Consider $f(x, y) = x^2 + 2y^2 + xy$ near $\mathbf{a} = (1, 1)$. Working at $\mathbf{x} = (1.3, 0.9)$ near $\mathbf{a}$, let us estimate $f(1.3, 0.9)$ using the linear approximation with base point $\mathbf{a} = (1, 1)$.

First compute the partial derivatives:

$$\frac{\partial f}{\partial x} = 2x + y, \qquad \frac{\partial f}{\partial y} = 4y + x,$$

so

$$(\nabla f)(x, y) =
\begin{pmatrix}
2x + y \\[0.25em]
4y + x
\end{pmatrix},
\qquad
(\nabla f)(1, 1) =
\begin{pmatrix}
3 \\[0.25em]
5
\end{pmatrix}.$$

Also $f(1, 1) = 1 + 2 + 1 = 4$, and $\mathbf{x} - \mathbf{a} = \begin{pmatrix} 0.3 \\ -0.1 \end{pmatrix}$. Thus

$$
f(1.3, 0.9) \approx f(1, 1) + (\nabla f)(1, 1) \cdot \begin{pmatrix} 0.3 \\ -0.1 \end{pmatrix}
= 4 + \begin{pmatrix} 3 \\ 5 \end{pmatrix} \cdot \begin{pmatrix} 0.3 \\ -0.1 \end{pmatrix}
= 4 + (0.9 - 0.5) = 4.4.
$$

The exact value is $f(1.3, 0.9) = 4.48$.

## The gradient as normal to contours

**Theorem:**

Let $f : \mathbb{R}^2 \to \mathbb{R}$ be a scalar-valued function, and suppose $(\nabla f)(a, b) \neq 0$.

**(i)** The gradient $(\nabla f)(a, b)$ is **perpendicular** to the level set of $f$ that goes through $(a, b)$ (equivalently: it is perpendicular to the **tangent line** to that level set at $(a, b)$). It points in the **direction of maximal increase** of $f(x, y)$ as $(x, y)$ moves away from $(a, b)$. The figure below illustrates this for $f(x, y) = xy - x$.

![img](gradient_contour.png)

*A contour plot of $f(x, y) = xy - x$. The gradient $\nabla f$ is drawn at three points: $\mathbf{a} = (1, 3)$, $\mathbf{b} = (2, 2)$, $\mathbf{c} = (4, 3/2)$. Observe that $\nabla f$ is perpendicular to the level curve at each point.*

**(ii)** The equation

$$(\nabla f)(a, b) \cdot \begin{pmatrix} x - a \\ y - b \end{pmatrix} = 0$$

in the $(x, y)$-plane is the line **tangent** to the level curve of $f$ through $(x, y) = (a, b)$. Equivalently, writing the dot product out,

$$f_x(a, b)(x - a) + f_y(a, b)(y - b) = 0.$$

**Theorem:**

For a scalar-valued function $f : \mathbb{R}^3 \to \mathbb{R}$ and a point $\mathbf{a} = (a_1, a_2, a_3)$ for which $(\nabla f)(\mathbf{a}) \neq 0$, the **gradient vector** $(\nabla f)(\mathbf{a})$ is perpendicular to the **plane tangent** to the level set of $f$ through $\mathbf{a}$. In particular, that tangent plane is given by

$$(\nabla f)(a_1, a_2, a_3) \cdot \begin{pmatrix} x - a_1 \\ y - a_2 \\ z - a_3 \end{pmatrix} = 0.$$

The same picture carries over to functions $f : \mathbb{R}^n \to \mathbb{R}$ for $n > 3$, except that the tangent **plane** to a level set in $\mathbb{R}^3$ is replaced by the appropriate **tangent hyperplane** to a level set $\{f = c\}$ in $\mathbb{R}^n$.

**Example:** Consider the circle defined by $x^2 + y^2 = 25$. Let us find a **normal vector** to this circle at $(x, y) = (3, 4)$, as well as the equation of the **tangent line** at that point. This case can be handled with single-variable calculus (worth trying if you are interested), but we use **gradients** here to illustrate a method that extends to three or more variables.

Define $h(x, y) = x^2 + y^2$. Then $h(3, 4) = 25$ and

$$\nabla h(x, y) = \begin{pmatrix} 2x \\ 2y \end{pmatrix}.$$

A normal vector to the level set $h = 25$ (our circle) through $(3, 4)$ is the gradient

$$(\nabla h)(3, 4) = \begin{pmatrix} 6 \\ 8 \end{pmatrix}.$$

If $\begin{pmatrix} x \\ y \end{pmatrix}$ lies on the tangent line, the displacement from $(3, 4)$ to $\begin{pmatrix} x \\ y \end{pmatrix}$ must be **perpendicular** to that normal. Thus

$$\begin{pmatrix} x - 3 \\ y - 4 \end{pmatrix} \cdot \begin{pmatrix} 6 \\ 8 \end{pmatrix} = 0,$$

i.e.

$$6(x - 3) + 8(y - 4) = 0.$$

After simplifying, this becomes $3x + 4y = 25$, as illustrated below.

![img](circle_tangent.png)

**Example:** Consider the sphere $S$ given by $x^2 + y^2 + z^2 = 6$. Let us find an equation for the **tangent plane** to $S$ at $(2, 1, 1)$.

$S$ is the level set $f = 0$ of

$f(x, y, z) = x^2 + y^2 + z^2 - 6$ (since $f = 0$ is equivalent to $x^2 + y^2 + z^2 = 6$). Then

$$\nabla f(x, y, z) = \begin{pmatrix} 2x \\ 2y \\ 2z \end{pmatrix}.$$

As above, this tangent plane is perpendicular to $(\nabla f)(2, 1, 1) = \begin{pmatrix} 4 \\ 2 \\ 2 \end{pmatrix}$, so its equation is

$$\begin{pmatrix} 4 \\ 2 \\ 2 \end{pmatrix} \cdot \begin{pmatrix} x - 2 \\ y - 1 \\ z - 1 \end{pmatrix} = 0.$$

After simplifying, this becomes $2x + y + z = 6$.

**Example:** Consider the surface $S$ defined by $z = x^2 + y^2$. Let us find the equation of the **tangent plane** to $S$ at $(1, 2, 5)$.

We can rewrite the equation as $x^2 + y^2 - z = 0$. Thus $S$ is the level set $f = 0$ of

$$f(x, y, z) = z - x^2 - y^2$$

(since $f = 0$ is equivalent to $z = x^2 + y^2$). The gradient is

$$\nabla f(x, y, z) = \begin{pmatrix} -2x \\ -2y \\ 1 \end{pmatrix},$$

Next,

$$(\nabla f)(1, 2, 5) = \begin{pmatrix} -2 \\ -4 \\ 1 \end{pmatrix}$$

is normal to $S$ at $(1, 2, 5)$. The tangent plane through that point is therefore given in **point–normal** form by

$$\begin{pmatrix} -2 \\ -4 \\ 1 \end{pmatrix} \cdot \left( \begin{pmatrix} x \\ y \\ z \end{pmatrix} - \begin{pmatrix} 1 \\ 2 \\ 5 \end{pmatrix} \right) = 0.$$

This simplifies to $-2(x - 1) - 4(y - 2) + (z - 5) = 0$, or equivalently

$$z = 2x + 4y - 5.$$

The figure below compares the graph of $z = x^2 + y^2$ and the tangent plane $z = 2x + 4y - 5$ at $(1, 2, 5)$.

![img](paraboloid_tangent_plane.png)

## Gradient descent