# Linear transformations and matrix multiplication

## Rotations and linear functions

The overall effect of applying a counterclockwise rotation by an angle $\theta_1$ and then a counterclockwise rotation by an angle $\theta_2$ is such a rotation: counterclockwise by the angle $\theta_1 + \theta_2$. Note in particular that the order in which we apply the $2$-dimensional rotations doesn't matter for the final outcome since the angle sums $\theta_1 + \theta_2$ and $\theta_2 + \theta_1$ are equal.

When we consider an analogue in $3$ dimensions, the situation is more subtle (and so more interesting). Think of the following operations you can do to the surface of a ball:

- **operation $R_1$**: rotate the ball by $-45$ degrees around the vertical axis through its center;
- **operation $R_2$**: rotate the ball by $45$ degrees around a specified horizontal axis through its center.

![img](2r.png)

What if you do operation $R_1$ first and then operation $R_2$? What if you do operation $R_2$ first and then operation $R_1$? The overall effect of these is not the same!

![img](assym.png)

**Definition:** A function $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ is called

- a **linear function**, or a **linear transformation**, if there is an $m \times n$ matrix $A$ for which $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ for all vectors $\mathbf{x} \in \mathbb{R}^n$ (so the $j$th column of $A$ is $A\mathbf{e}_j = \mathbf{f}(\mathbf{e}_j)$),
- an **affine function**, or an **affine transformation**, if there is an $m \times n$ matrix $A$ and a vector $\mathbf{b} \in \mathbb{R}^m$ for which $\mathbf{f}(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ for all vectors $\mathbf{x} \in \mathbb{R}^n$.

This is not a new class of functions. In [Linear functions, matrices, and the derivative matrix](linear_functions_matrices_and_the_derivative_matrix.md#linear-and-affine-functions) we defined these notions one component at a time: $\mathbf{f}$ was called linear when each component function $f_i$ has the form $a_{i1}x_1 + \cdots + a_{in}x_n$, and affine when each $f_i$ has that form plus a constant. We then recorded, as a consequence of inspecting the definitions, that linear functions are exactly those of the form $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ and affine functions exactly those of the form $\mathbf{f}(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$.

What was a *consequence* there is taken as the *definition* here. The two formulations describe the same functions, and it is worth seeing exactly how each turns into the other. Starting from the componentwise description, suppose $f_i(\mathbf{x}) = a_{i1}x_1 + \cdots + a_{in}x_n$ for each $i$. Collect the coefficients of $f_i$ as the $i$th row of a matrix $A$. Then $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$, because the $i$th entry of $A\mathbf{x}$ is precisely the $i$th row of $A$ dotted with $\mathbf{x}$, which is $f_i(\mathbf{x})$. Starting instead from $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$, reading off the $i$th row of $A$ hands back the coefficients of $f_i$. So the matrix is nothing but the table of coefficients of the component functions, one row per component. The advantage of starting from $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ is that it names the matrix up front.

We want to discuss how to visualize the effect of the linear transformation $\mathbf{f} : \mathbb{R}^2 \to \mathbb{R}^2$ associated with a matrix

$$
A = \begin{pmatrix} a & b \\[0.25em] c & d \end{pmatrix}
$$

whose columns span different lines through $\mathbf{0}$. Let $S$ be the "unit square"

$$S = \{(x, y) \in \mathbb{R}^2 : 0 \leq x, y \leq 1\}$$

(on the left in the figure below), so $\mathbb{R}^2$ is tiled by copies of $S$ laid out in all directions (like a bathroom floor). Let $\mathbf{f}(S)$ denote the output of $\mathbf{f}$ on $S$ (i.e., the collection of points $\mathbf{f}(\mathbf{s})$ for $\mathbf{s} \in S$), also called the **image** of $S$ under $\mathbf{f}$. The image $\mathbf{f}(S)$ is a parallelogram (on the right in the figure below). 

![img](tiling.png)

This helps to visualize the effect of $\mathbf{f}$ on $\mathbb{R}^2$ due to two facts:

- **Linearity Principle:** for $c_1, c_2 \in \mathbb{R}$ and $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^2$ we have $\mathbf{f}(c_1\mathbf{v}_1 + c_2\mathbf{v}_2) = c_1\mathbf{f}(\mathbf{v}_1) + c_2\mathbf{f}(\mathbf{v}_2)$.
- **Tiling Principle:** $\mathbf{f}$ transforms the tiling of $\mathbb{R}^2$ by copies of $S$ into a tiling of $\mathbb{R}^2$ by copies of $\mathbf{f}(S)$.

**Example:** Consider $\mathbf{f} : \mathbb{R}^2 \to \mathbb{R}^2$ given by the matrix

$$
A = \begin{pmatrix} 2 & 1 \\[0.25em] 1 & 2 \end{pmatrix}.
$$

$\mathbf{f}(S)$ is the parallelogram with edges

$$
\mathbf{f}\begin{pmatrix} 1 \\[0.25em] 0 \end{pmatrix} = \begin{pmatrix} 2 \\[0.25em] 1 \end{pmatrix}
\quad (\text{first column of } A)
\qquad \text{and} \qquad
\mathbf{f}\begin{pmatrix} 0 \\[0.25em] 1 \end{pmatrix} = \begin{pmatrix} 1 \\[0.25em] 2 \end{pmatrix}
\quad (\text{second column of } A),
$$

as shown in the figure below. 

![img](tiling2.png)

By the Tiling Principle, $\mathbf{f}$ transforms the usual tiling of $\mathbb{R}^2$ by unit squares into the tiling by copies of $\mathbf{f}(S)$ laid out in all directions parallel to the edges of $\mathbf{f}(S)$.

!!! note
    The tiling on the right in the figure above expresses the parametric form of $\mathbb{R}^2$ using the vectors

    $$
    \mathbf{e} = \mathbf{f}\begin{pmatrix} 1 \\[0.25em] 0 \end{pmatrix} = \begin{pmatrix} 2 \\[0.25em] 1 \end{pmatrix}
    \quad (\text{first column of } A),
    \qquad
    \mathbf{e}' = \mathbf{f}\begin{pmatrix} 0 \\[0.25em] 1 \end{pmatrix} = \begin{pmatrix} 1 \\[0.25em] 2 \end{pmatrix}
    \quad (\text{second column of } A).
    $$

    In particular, the Linearity Principle

    $$
    t\mathbf{e} + t'\mathbf{e}'
    = t\,\mathbf{f}\begin{pmatrix} 1 \\[0.25em] 0 \end{pmatrix} + t'\,\mathbf{f}\begin{pmatrix} 0 \\[0.25em] 1 \end{pmatrix}
    = \mathbf{f}\left( t\begin{pmatrix} 1 \\[0.25em] 0 \end{pmatrix} + t'\begin{pmatrix} 0 \\[0.25em] 1 \end{pmatrix} \right)
    = \mathbf{f}\begin{pmatrix} t \\[0.25em] t' \end{pmatrix}
    $$

    says that the point in parametric form $t\mathbf{e} + t'\mathbf{e}'$ is the output of $\mathbf{f}$ on $\begin{pmatrix} t \\ t' \end{pmatrix}$.
