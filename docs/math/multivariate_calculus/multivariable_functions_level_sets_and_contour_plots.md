# Multivariable functions, level sets, and contour plots

## Basic terminology

Functions of more than one variable are called multivariable functions.

In particular a function from $\mathbb{R}^n$ to $\mathbb{R}^m$, typically denoted as

$$f : \mathbb{R}^n \rightarrow \mathbb{R}^m$$

takes vectors in $\mathbb{R}^n$ as input and gives vectors in $\mathbb{R}^m$ as output. Keep in mind that a function assigns to each input a single output, but it is fine if two inputs yield the same output (e.g., $f(\mathbf{x}) = \|\mathbf{x}\| = f(-\mathbf{x})$). 

A **scalar-valued function** is a function $\mathbb{R}^n \rightarrow \mathbb{R}$ (that is to say, with $m = 1$). In other words, a scalar-valued function gives real number outputs.

**Example:** The problem of finding a best-fit line involves minimizing a scalar-valued function $E : \mathbb{R}^2 \rightarrow \mathbb{R}$ of the vector $(m, b) \in \mathbb{R}^2$ (or more concretely, $E$ is an $\mathbb{R}$-valued function of two variables $m$ and $b$). We choose $(m, b)$ to minimize the sum of the squares of the errors; i.e., choose $(m, b)$ to minimize the scalar-valued function

$$E(m, b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

**Example:** Addition and multiplication are scalar-valued functions $\mathbb{R}^2 \rightarrow \mathbb{R}$:

$$A(x_1, x_2) = x_1 + x_2, \quad M(x_1, x_2) = x_1x_2$$

**Definition:** A **vector-valued function** is a function $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ with general $m \geq 1$. In other words, a vector-valued function gives output considered as vectors in some $\mathbb{R}^m$.

A vector-valued function $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ can be expressed in terms of $m$ scalar-valued **component functions** or **coordinate functions** $f_1, \ldots, f_m : \mathbb{R}^n \rightarrow \mathbb{R}$, defined by the expressions

$$f(\mathbf{x}) = \begin{pmatrix} f_1(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x}) \end{pmatrix} = (f_1(\mathbf{x}), \ldots, f_m(\mathbf{x}))$$

(depending on whether we consider the output to be a "vector" or a "point"), with each $f_j$ a scalar-valued function.

We can write the output of $f$ on the input $\mathbf{x} \in \mathbb{R}^n$ in (at least) three ways:

$$f(\mathbf{x}) = f\left(\begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}\right) = f(x_1, \ldots, x_n),$$

depending on whether we want to keep things compact, emphasize that the input to $f$ is considered as a vector in $\mathbb{R}^n$, or emphasize that the output of $f$ depends on $n$ real-number inputs (the coordinates of the point or vector $\mathbf{x}$).

**Example:** Consider a small object flying through the air. At any given time $t$, its position in space is a point $\mathbf{x}(t) = (x(t), y(t), z(t)) \in \mathbb{R}^3$ and its velocity (a vector pointing in the direction of motion with magnitude equal to the speed) is some

$$\mathbf{v}(t) = \begin{pmatrix} v_1(t) \\ v_2(t) \\ v_3(t) \end{pmatrix} \in \mathbb{R}^3,$$

so both position and velocity are $\mathbb{R}^3$-valued functions of time $t \in \mathbb{R}$. In other words, we have vector-valued functions

$$\mathbf{x} : \mathbb{R} \rightarrow \mathbb{R}^3, \quad \mathbf{v} : \mathbb{R} \rightarrow \mathbb{R}^3.$$

**Example:** Vector-valued functions can encode ways to manipulate vectors geometrically. For example, the function $T : \mathbb{R}^2 \rightarrow \mathbb{R}^2$ given by

$$T\left(\begin{pmatrix} x \\ y \end{pmatrix}\right) = \begin{pmatrix} -y \\ x \end{pmatrix}$$

is a rotation.