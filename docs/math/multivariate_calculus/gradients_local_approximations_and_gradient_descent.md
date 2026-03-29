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
