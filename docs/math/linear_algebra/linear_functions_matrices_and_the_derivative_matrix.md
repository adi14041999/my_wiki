# Linear functions, matrices, and the derivative matrix

## The linear approximation to a vector-valued function

For a function of a single variable $f : \mathbb{R} \to \mathbb{R}$ that is differentiable at $a$, a small change $h$ in the input produces an approximate change of $f'(a)\,h$ in the output (see [Single-variable derivative review](../multivariate_calculus/partial_derivatives_and_contour_plots.md#single-variable-derivative-review)):

$$f(a + h) \approx f(a) + f'(a)\,h \qquad \text{for } h \text{ near } 0,$$

or equivalently, writing $x = a + h$ so that $h = x - a$,

$$f(x) \approx f(a) + f'(a)\,(x - a) \qquad \text{for } x \text{ near } a.$$

For $f : \mathbb{R}^n \to \mathbb{R}$ the single number $f'(a)$ is replaced by the **gradient** $(\nabla f)(\mathbf{a})$, the column vector of partial derivatives evaluated at $\mathbf{a}$ (see [The Gradient](../multivariate_calculus/gradients_local_approximations_and_gradient_descent.md#the-gradient)):

$$
(\nabla f)(\mathbf{a}) =
\begin{pmatrix}
\dfrac{\partial f}{\partial x_1}(\mathbf{a}) \\[0.35em]
\dfrac{\partial f}{\partial x_2}(\mathbf{a}) \\[0.35em]
\vdots \\[0.2em]
\dfrac{\partial f}{\partial x_n}(\mathbf{a})
\end{pmatrix} \in \mathbb{R}^n ,
$$

and the product $f'(a)\,h$ is replaced by a dot product. The **linear approximation** to $f$ at $\mathbf{a}$ (see [The linear approximation for a scalar-valued function](../multivariate_calculus/gradients_local_approximations_and_gradient_descent.md#the-linear-approximation-for-a-scalar-valued-function)) is

$$f(\mathbf{x}) \approx f(\mathbf{a}) + (\nabla f)(\mathbf{a}) \cdot (\mathbf{x} - \mathbf{a}) \qquad \text{for } \mathbf{x} \text{ near } \mathbf{a}.$$

Written out for $n = 2$, with $\mathbf{a} = (a,b)$ and $\mathbf{x} = (x,y)$, this is

$$
f(x, y) \approx f(a, b) + \underbrace{f_x(a, b)(x - a) + f_y(a, b)(y - b)}_{\displaystyle (\nabla f)(a,b)\,\cdot\, \begin{pmatrix} x - a \\ y - b \end{pmatrix}} .
$$

Suppose that $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ is a vector-valued function, with component functions $f_1, f_2, \ldots, f_m$:

$$
\mathbf{f}(\mathbf{x}) =
\begin{pmatrix}
f_1(\mathbf{x}) \\[0.25em]
f_2(\mathbf{x}) \\[0.25em]
\vdots \\[0.2em]
f_m(\mathbf{x})
\end{pmatrix}
\qquad \text{for } \mathbf{x} \in \mathbb{R}^n .
$$

For an $n$-vector $\mathbf{a}$, how can we approximate the $m$-vector $\mathbf{f}(\mathbf{x})$ when $\mathbf{x}$ is near $\mathbf{a}$?

There is an obvious thing to try, and it needs no new machinery. Each component function $f_i : \mathbb{R}^n \to \mathbb{R}$ is **scalar-valued**, and we know how to linearly approximate one of those: $f_i(\mathbf{x}) \approx f_i(\mathbf{a}) + (\nabla f_i)(\mathbf{a}) \cdot (\mathbf{x} - \mathbf{a})$. So approximate each of the $m$ components separately, then stack the $m$ results back up into a column. To make things easier to imagine, suppose $m = n = 3$. Let's consider a specific example.

**Example:** Define $g : \mathbb{R}^3 \to \mathbb{R}^3$ by

$$g(x, y, z) = \bigl(e^{z}(x - y)^2,\ \ 2yz + x^3,\ \ x^2 - y^3 + z\bigr),$$

and let $\mathbf{a} = (2, 1, 0)$. The respective component functions are

$$g_1(x,y,z) = e^{z}(x-y)^2, \qquad g_2(x,y,z) = 2yz + x^3, \qquad g_3(x,y,z) = x^2 - y^3 + z,$$

each a scalar-valued function $\mathbb{R}^3 \to \mathbb{R}$. Their gradients are

$$
\nabla g_1 = \begin{pmatrix} 2e^{z}(x-y) \\ -2e^{z}(x-y) \\ e^{z}(x-y)^2 \end{pmatrix}, \qquad
\nabla g_2 = \begin{pmatrix} 3x^2 \\ 2z \\ 2y \end{pmatrix}, \qquad
\nabla g_3 = \begin{pmatrix} 2x \\ -3y^2 \\ 1 \end{pmatrix},
$$

so that at $\mathbf{a} = (2,1,0)$ we get $g_1(\mathbf{a}) = 1$, $g_2(\mathbf{a}) = 8$, $g_3(\mathbf{a}) = 3$ and

$$(\nabla g_1)(\mathbf{a}) = (2, -2, 1), \qquad (\nabla g_2)(\mathbf{a}) = (12, 0, 2), \qquad (\nabla g_3)(\mathbf{a}) = (4, -3, 1).$$

Hence if $(x, y, z)$ is near $\mathbf{a}$ then we have

$$
\begin{aligned}
g_1(x,y,z) &\approx g_1(\mathbf{a}) + \bigl((\nabla g_1)(\mathbf{a})\bigr) \cdot (x - 2,\, y - 1,\, z) = 1 + (2, -2, 1) \cdot (x - 2,\, y - 1,\, z), \\[0.35em]
g_2(x,y,z) &\approx g_2(\mathbf{a}) + \bigl((\nabla g_2)(\mathbf{a})\bigr) \cdot (x - 2,\, y - 1,\, z) = 8 + (12, 0, 2) \cdot (x - 2,\, y - 1,\, z), \\[0.35em]
g_3(x,y,z) &\approx g_3(\mathbf{a}) + \bigl((\nabla g_3)(\mathbf{a})\bigr) \cdot (x - 2,\, y - 1,\, z) = 3 + (4, -3, 1) \cdot (x - 2,\, y - 1,\, z).
\end{aligned}
$$

Putting these scalar approximations together yields a vector approximation

$$
g(x, y, z) \approx
\begin{pmatrix}
1 + (2, -2, 1) \cdot (x - 2,\, y - 1,\, z) \\[0.35em]
8 + (12, 0, 2) \cdot (x - 2,\, y - 1,\, z) \\[0.35em]
3 + (4, -3, 1) \cdot (x - 2,\, y - 1,\, z)
\end{pmatrix}
=
\begin{pmatrix}
-1 + 2x - 2y + z \\[0.35em]
-16 + 12x + 2z \\[0.35em]
-2 + 4x - 3y + z
\end{pmatrix} .
$$

More generally, still with $m = n = 3$, we want to approximate the vector

$$
\mathbf{f}(x, y, z) =
\begin{pmatrix}
f_1(x, y, z) \\[0.25em]
f_2(x, y, z) \\[0.25em]
f_3(x, y, z)
\end{pmatrix}
$$

for $(x, y, z)$ near a point $(a, b, c)$, for **any** scalar-valued functions $f_1, f_2, f_3$. We have

$$
\begin{aligned}
f_i(x, y, z)
&\approx f_i(a, b, c) + \bigl((\nabla f_i)(a, b, c)\bigr) \cdot (x - a,\, y - b,\, z - c) \\[0.4em]
&= f_i(a, b, c)
+ \left( \frac{\partial f_i}{\partial x}(a, b, c) \right)(x - a)
+ \left( \frac{\partial f_i}{\partial y}(a, b, c) \right)(y - b)
+ \left( \frac{\partial f_i}{\partial z}(a, b, c) \right)(z - c),
\end{aligned}
$$

so combining these for $f_1$, $f_2$, and $f_3$ yields

$$
\begin{pmatrix}
f_1(x, y, z) \\[0.25em]
f_2(x, y, z) \\[0.25em]
f_3(x, y, z)
\end{pmatrix}
\approx
\mathbf{f}(a, b, c)
+
\begin{pmatrix}
\dfrac{\partial f_1}{\partial x}(a, b, c)(x - a) + \dfrac{\partial f_1}{\partial y}(a, b, c)(y - b) + \dfrac{\partial f_1}{\partial z}(a, b, c)(z - c) \\[0.9em]
\dfrac{\partial f_2}{\partial x}(a, b, c)(x - a) + \dfrac{\partial f_2}{\partial y}(a, b, c)(y - b) + \dfrac{\partial f_2}{\partial z}(a, b, c)(z - c) \\[0.9em]
\dfrac{\partial f_3}{\partial x}(a, b, c)(x - a) + \dfrac{\partial f_3}{\partial y}(a, b, c)(y - b) + \dfrac{\partial f_3}{\partial z}(a, b, c)(z - c)
\end{pmatrix} .
$$

Alas, the right side of this last approximation is horrifying. We need a more compact and efficient way to work with it. The way to handle it is provided by the language of **matrices**, to which we now turn.

## Linear and affine functions

**Definition:** A scalar-valued function $f : \mathbb{R}^n \to \mathbb{R}$ is called

- **affine** if it has the form $f(x_1, \ldots, x_n) = a_1x_1 + a_2x_2 + \cdots + a_nx_n + b$ for some numbers $a_1, \ldots, a_n, b$ (so $b = f(\mathbf{0})$).
- **linear** if it has the form $f(x_1, \ldots, x_n) = a_1x_1 + a_2x_2 + \cdots + a_nx_n$ for some numbers $a_1, \ldots, a_n$; i.e., it is affine with $b = 0$, or equivalently with $f(\mathbf{0}) = 0$.

A vector-valued function $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ (so $\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_m(\mathbf{x}))$) is called

- **affine** if each of its component functions $f_i : \mathbb{R}^n \to \mathbb{R}$ is affine.
- **linear** if each of its component functions $f_i : \mathbb{R}^n \to \mathbb{R}$ is linear.

**Example:** The function

$$
f(x, y, z) =
\begin{pmatrix}
x - y + z + 3 \\[0.25em]
z - x \\[0.25em]
y + x + 1
\end{pmatrix}
=
\begin{pmatrix}
x - y + z + 3 \\[0.25em]
-x + 0y + z \\[0.25em]
x + y + 0z + 1
\end{pmatrix}
$$

from $\mathbb{R}^3$ to $\mathbb{R}^3$ is affine. The function

$$
g(x, y, z) =
\begin{pmatrix}
x - y + z \\[0.25em]
z - x \\[0.25em]
y + x
\end{pmatrix}
$$

from $\mathbb{R}^3$ to $\mathbb{R}^3$ is linear, and

$$
h(x, y, z) =
\begin{pmatrix}
x - y + z \\[0.25em]
z - x \\[0.25em]
y + x + z^2
\end{pmatrix}
$$

is neither affine nor linear (due to $z^2$ in the third component function $h_3 = y + x + z^2$).

## Matrices: a shorthand for linear functions

**Definition:** An $m \times n$ **matrix** is a rectangular array $A$ of numbers presented like this:

$$
\begin{pmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\[0.25em]
a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\[0.25em]
\vdots & \vdots & \ddots & \vdots \\[0.25em]
a_{m,1} & a_{m,2} & \cdots & a_{m,n}
\end{pmatrix} .
$$

The collection of entries $\begin{pmatrix} a_{i,1} & a_{i,2} & \cdots & a_{i,n} \end{pmatrix}$ along the $i$th horizontal layer (with $i = 1$ along the top side) is called the $i$th **row**, and the collection of entries

$$
\begin{pmatrix}
a_{1,j} \\[0.25em]
a_{2,j} \\[0.25em]
\vdots \\[0.25em]
a_{m,j}
\end{pmatrix}
$$

along the $j$th vertical layer (with $j = 1$ along the left side) is called the $j$th **column**.

The entry at the crossing of the $i$th row from the top and $j$th column from the left is denoted $a_{ij}$ (or sometimes $a_{i,j}$); it is called the **$ij$-entry** or **$(i,j)$-entry**.

**Definition:** If $A$ is an $m \times n$ matrix, and $\mathbf{x} \in \mathbb{R}^n$, the **matrix-vector product** $A\mathbf{x} \in \mathbb{R}^m$ is defined as

$$
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\[0.25em]
a_{21} & a_{22} & \cdots & a_{2n} \\[0.25em]
\vdots & \vdots & \ddots & \vdots \\[0.25em]
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}
\begin{pmatrix}
x_1 \\[0.25em]
x_2 \\[0.25em]
\vdots \\[0.25em]
x_n
\end{pmatrix}
=
\begin{pmatrix}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n \\[0.25em]
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n \\[0.25em]
\vdots \\[0.25em]
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n
\end{pmatrix} .
$$

In other words, if we write $\mathbf{r}_1, \ldots, \mathbf{r}_m$ for the rows of $A$ (so these are $n$-vectors), then

$$
A\mathbf{x} =
\begin{pmatrix}
\mathbf{r}_1 \cdot \mathbf{x} \\[0.25em]
\mathbf{r}_2 \cdot \mathbf{x} \\[0.25em]
\vdots \\[0.25em]
\mathbf{r}_m \cdot \mathbf{x}
\end{pmatrix} .
$$

!!! warning
    If $A$ is an $m \times n$ matrix and $\mathbf{x}$ is a $d$-vector with $d \neq n$ then the matrix-vector product $A\mathbf{x}$ is not defined.

**Example:** We saw earlier that the function

$$
g(x, y, z) =
\begin{pmatrix}
x - y + z \\[0.25em]
z - x \\[0.25em]
y + x
\end{pmatrix}
$$

from $\mathbb{R}^3$ to $\mathbb{R}^3$ is linear. Let us find the matrix that represents it. Write each component function as a combination of $x$, $y$, $z$ in that order:

$$
g(x, y, z) =
\begin{pmatrix}
1x - 1y + 1z \\[0.25em]
-1x + 0y + 1z \\[0.25em]
1x + 1y + 0z
\end{pmatrix} .
$$

The $i$th row of the matrix is the list of coefficients of the $i$th component function, so

$$
A =
\begin{pmatrix}
1 & -1 & 1 \\[0.25em]
-1 & 0 & 1 \\[0.25em]
1 & 1 & 0
\end{pmatrix},
\qquad
g(x, y, z) = A \begin{pmatrix} x \\[0.25em] y \\[0.25em] z \end{pmatrix} .
$$

Checking against the row description $A\mathbf{x} = (\mathbf{r}_1 \cdot \mathbf{x},\, \mathbf{r}_2 \cdot \mathbf{x},\, \mathbf{r}_3 \cdot \mathbf{x})$ with $\mathbf{x} = (x, y, z)$:

$$
\mathbf{r}_1 \cdot \mathbf{x} = x - y + z, \qquad
\mathbf{r}_2 \cdot \mathbf{x} = z - x, \qquad
\mathbf{r}_3 \cdot \mathbf{x} = x + y ,
$$

as desired. Note that $a_{22} = 0$ because $y$ is absent from the second component function, and $a_{33} = 0$ because $z$ is absent from the third.

**Definition:** A function $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ is **linear** precisely when $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ for an $m \times n$ matrix $A$. This just rephrases the definition of "linear function". In this way, an $m \times n$ matrix $A$ is a shorthand way of encoding a linear function from $\mathbb{R}^n$ to $\mathbb{R}^m$.

We can also use matrices to give a shorthand for affine functions.

**Example:** The affine function

$$
f(x, y, z) =
\begin{pmatrix}
x - y + z + 3 \\[0.25em]
z - x \\[0.25em]
y + x + 1
\end{pmatrix}
$$

from the earlier example can be expressed as

$$
f\begin{pmatrix} x \\[0.25em] y \\[0.25em] z \end{pmatrix}
=
\begin{pmatrix}
1 & -1 & 1 \\[0.25em]
-1 & 0 & 1 \\[0.25em]
1 & 1 & 0
\end{pmatrix}
\begin{pmatrix} x \\[0.25em] y \\[0.25em] z \end{pmatrix}
+
\begin{pmatrix} 3 \\[0.25em] 0 \\[0.25em] 1 \end{pmatrix} .
$$

!!! note
    Much as inspection of definitions showed that linear functions $\mathbb{R}^n \to \mathbb{R}^m$ are exactly those of the form $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ for an $m \times n$ matrix $A$, affine functions $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ are exactly those of the form $\mathbf{f}(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$, where $A$ is an $m \times n$ matrix and $\mathbf{b} \in \mathbb{R}^m$ is a vector.

## Further viewpoints on matrix-vector products

!!! note "Theorem"
    If $\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_n$ are the columns of $A$ (so viewed as vectors in $\mathbb{R}^m$), which is to say

    $$
    A =
    \begin{pmatrix}
    | & | & & | \\
    \mathbf{c}_1 & \mathbf{c}_2 & \cdots & \mathbf{c}_n \\
    | & | & & |
    \end{pmatrix},
    $$

    then

    $$
    A
    \begin{pmatrix}
    x_1 \\[0.25em]
    x_2 \\[0.25em]
    \vdots \\[0.25em]
    x_n
    \end{pmatrix}
    = x_1\mathbf{c}_1 + x_2\mathbf{c}_2 + \cdots + x_n\mathbf{c}_n \in \mathbb{R}^m .
    $$

    In particular, the matrix-vector product is a specific linear combination of the columns of the matrix.
