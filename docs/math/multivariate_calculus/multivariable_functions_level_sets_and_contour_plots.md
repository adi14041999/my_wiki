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

## Composition

For functions $f : \mathbb{R} \rightarrow \mathbb{R}$ and $g : \mathbb{R} \rightarrow \mathbb{R}$, the composition $f \circ g : \mathbb{R} \rightarrow \mathbb{R}$ is defined by $(f \circ g)(x) = f(g(x))$. As an illustration, the function $h(x) = \sin(x^2)$ is the composition $f \circ g$ for $g(x) = x^2$ and $f(u) = \sin(u)$.

Just as with functions $\mathbb{R} \rightarrow \mathbb{R}$, we can form the composition of vector-valued functions.

**Definition.** If $g : \mathbb{R}^n \rightarrow \mathbb{R}^p$ and $f : \mathbb{R}^p \rightarrow \mathbb{R}^m$ are multivariable functions (note that $g$ has output belonging to $\mathbb{R}^p$ on which $f$ is applied), we can form a new composite function: take an input in $\mathbb{R}^n$; first apply $g$ to it, and then apply $f$:

$$\text{input } \mathbf{x} \in \mathbb{R}^n \xrightarrow{g} \mathbb{R}^p \xrightarrow{f} \mathbb{R}^m$$

As a shorthand, we write this new function as $f \circ g$; the symbol $\circ$ is read as "composed with." In symbols, the new function is given by

$$(f \circ g)(\mathbf{x}) = (f \text{ applied to } g(\mathbf{x})) = f(g(\mathbf{x}))$$

**Example:** Consider the functions $f : \mathbb{R}^2 \rightarrow \mathbb{R}^2$ and $g : \mathbb{R}^2 \rightarrow \mathbb{R}^2$ defined by

$$f(u, v) = (uv, u + v), \quad g(x, y) = (e^{xy}, x - y)$$

So $f_1(u, v) = uv$, $f_2(u, v) = u + v$, $g_1(x, y) = e^{xy}$, $g_2(x, y) = x - y$. Then $f \circ g : \mathbb{R}^2 \rightarrow \mathbb{R}^2$ evaluated at $(x, y) \in \mathbb{R}^2$ equals

$$(f \circ g)(x, y) = f(g(x, y)) = f(e^{xy}, x - y) = (e^{xy}(x - y), e^{xy} + x - y).$$

In this example, the composite function $g \circ f : \mathbb{R}^2 \rightarrow \mathbb{R}^2$ also makes sense. Its value on input $(u, v) \in \mathbb{R}^2$ is

$$(g \circ f)(u, v) = g(f(u, v)) = g(uv, u + v) = (e^{uv(u+v)}, uv - (u + v)) = (e^{u^2v+uv^2}, uv - u - v).$$

Observe that in this case, $f \circ g$ and $g \circ f$ are very different functions (just look at the formulas we have computed for each). The order of composition matters (familiar for scalar functions: $\sin(x^2) \neq \sin(x)^2$ as functions).

**Example:** For the functions $g : \mathbb{R} \rightarrow \mathbb{R}^3$ and $f : \mathbb{R}^3 \rightarrow \mathbb{R}^2$ defined by

$$g(t) = (t, \cos(t), \sin(t)), \quad f(x, y, z) = (y, z),$$

$g$ can be visualized as the path of a particle moving on a helix on the cylinder $y^2 + z^2 = 1$ of radius 1 around the $x$-axis, and $f$ is the projection onto the $yz$-plane. Then $f \circ g : \mathbb{R} \rightarrow \mathbb{R}^2$ is given by

$$(f \circ g)(t) = f(t, \cos(t), \sin(t)) = (\cos(t), \sin(t)),$$

the path of the particle's "shadow" in the $yz$-plane moving counterclockwise around a circle of radius 1 in the $yz$-plane.

![img](helix.png)

## Graphs, level sets, and contour plots

In our experience with functions $f(x)$ of one variable, it can be quite helpful to visualize the function graphically.

Much as the graph of a 1-variable function $f(x)$ is the subset of $\mathbb{R}^2$ defined as

$$\text{Graph}(f) = \{(x, y) \in \mathbb{R}^2 : y = f(x)\},$$

for an $n$-variable function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ its graph is a subset of $\mathbb{R}^{n+1}$ defined as follows:

**Definition:** The graph of $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is the subset of $\mathbb{R}^{n+1}$ (not $\mathbb{R}^n$!) defined as

$$\text{Graph}(f) = \{(x_1, \cdots, x_n, z) \in \mathbb{R}^{n+1} : z = f(x_1, \cdots, x_n)\}.$$

![img](fxy.png)

**Example:** Let's work out the graph of the function $f(x, y) = \sqrt{1 - x^2 - y^2}$.

We can only take the square root of a nonnegative number, so we require $1 - x^2 - y^2 \geq 0$, or equivalently $x^2 + y^2 \leq 1$.

This is the disk $D$ centered at the origin in $\mathbb{R}^2$ with radius 1. The graph of $f$ is therefore

$$\text{Graph}(f) = \{(x, y, z) \in \mathbb{R}^3 : (x, y) \in D, z = f(x, y)\} = \{(x, y, \sqrt{1 - x^2 - y^2}) : (x, y) \in D\}$$

Notice that since $z = \sqrt{1 - x^2 - y^2}$, we then have $x^2 + y^2 + z^2 = 1$ with $z \geq 0$. This graph is the upper hemisphere of the sphere in $\mathbb{R}^3$ with radius 1 centered at $(0, 0, 0)$.

![img](hemi.png)

If you are hiking in a park and you get a “contour map” (or “contour plot”) of the terrain you are about to hike in, it may look something like below.

![img](cont.png)

The curves on the left in the figure above indicate where the terrain is at a fixed level. For example, the curve that is labeled by 400 represents where the terrain has an altitude of 400 feet. The mathematical way to think about this is to consider the altitude function,

$$z = A(x, y).$$

The set of points $(x, y)$ where $A(x, y) = 400$ is the curve on the contour map labeled 400. The set of points $(x, y)$ where $A(x, y) = 600$ is the curve on the contour map labeled 600. In general, the set of points $(x, y)$ where $A(x, y) = c$ is called the **level curve** of the function $A$ at level $c$ (and is also called a **level set**, or sometimes even a **contour line** even though it generally looks nothing at all like a line).

The contour map consisting of a collection of level curves is very helpful in visualizing the altitude function $z = A(x, y)$, and gives us a good understanding of the terrain. Of course, the contour map doesn't show the level curves $A(x, y) = c$ for every $c$; that would be impossible. Rather, the contour map shows these level curves for "enough" values of $c$ that one can get a sense of the hilliness of the terrain for practical purposes. Between level curves drawn for values $c_1 < c_2$ are level curves for intermediate values of $c$ that are omitted for clarity.
