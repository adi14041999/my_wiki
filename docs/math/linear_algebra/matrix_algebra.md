# Matrix algebra

We haven’t given an especially good reason for the terminology “matrix multiplication” (other than that we chose to denote it as AB). There is also an operation called **matrix addition**. In the historical development of matrices, multiplication came before addition! This may sound surprising, but the reason is very natural: composing rotations has visual meaning whereas matrix addition has only algebraic rather than geometric significance.

## Some examples with matrix multiplication

**Definition:** For an $n \times n$ matrix $A$, its **diagonal** consists of the entries $a_{ii}$ in position $(i, i)$ for all $1 \leq i \leq n$: this is the diagonal going from upper left to lower right, sometimes called the **main diagonal**. The other diagonal direction, from lower left to upper right, is sometimes called the anti-diagonal; it plays no significant role whatsoever in linear algebra. More generally, for an $m \times n$ matrix $B$ for any $m, n \geq 1$, its diagonal consists of its entries $b_{ii}$. An $m \times n$ matrix is called **diagonal** if all entries away from the diagonal vanish.

**Example:** Here are some examples of diagonal matrices:

$$
\begin{pmatrix}
2 & 0 & 0 \\[0.25em]
0 & 3 & 0 \\[0.25em]
0 & 0 & 4
\end{pmatrix},
\qquad
\begin{pmatrix}
-1 & 0 \\[0.25em]
0 & 5
\end{pmatrix},
\qquad
\begin{pmatrix}
0 & 0 & 0 & 0 \\[0.25em]
0 & 0 & 0 & 0 \\[0.25em]
0 & 0 & -1 & 0 \\[0.25em]
0 & 0 & 0 & 0
\end{pmatrix},
$$

$$
\begin{pmatrix}
-6 & 0 & 0 & 0 \\[0.25em]
0 & 4 & 0 & 0 \\[0.25em]
0 & 0 & 1 & 0
\end{pmatrix},
\qquad
\begin{pmatrix}
3 & 0 & 0 \\[0.25em]
0 & 0 & 0 \\[0.25em]
0 & 0 & -2 \\[0.25em]
0 & 0 & 0
\end{pmatrix}.
$$

It's very easy to multiply two diagonal matrices: the diagonal entries just multiply! For instance:

$$
\begin{pmatrix}
a & 0 & 0 \\[0.25em]
0 & b & 0 \\[0.25em]
0 & 0 & c
\end{pmatrix}
\begin{pmatrix}
a' & 0 & 0 \\[0.25em]
0 & b' & 0 \\[0.25em]
0 & 0 & c'
\end{pmatrix}
=
\begin{pmatrix}
aa' & 0 & 0 \\[0.25em]
0 & bb' & 0 \\[0.25em]
0 & 0 & cc'
\end{pmatrix},
$$

$$
\begin{pmatrix}
a & 0 & 0 \\[0.25em]
0 & b & 0 \\[0.25em]
0 & 0 & c \\[0.25em]
0 & 0 & 0
\end{pmatrix}
\begin{pmatrix}
a' & 0 & 0 & 0 & 0 \\[0.25em]
0 & b' & 0 & 0 & 0 \\[0.25em]
0 & 0 & c' & 0 & 0
\end{pmatrix}
=
\begin{pmatrix}
aa' & 0 & 0 & 0 & 0 \\[0.25em]
0 & bb' & 0 & 0 & 0 \\[0.25em]
0 & 0 & cc' & 0 & 0 \\[0.25em]
0 & 0 & 0 & 0 & 0
\end{pmatrix}.
$$

The special property of the number $1$ is that $1 \times a = a = a \times 1$ for any $a \in \mathbb{R}$. There is an $n \times n$ matrix that satisfies an analogous property for multiplication of $n \times n$ matrices, and (even though typically $AB \neq BA$ when $n > 1$) it works whether you multiply on the left or right.

**Definition:** The $n \times n$ **identity matrix**, denoted $I_n$, is defined to be the diagonal $n \times n$ matrix

$$
I_n =
\begin{pmatrix}
1 & 0 & 0 & \cdots & 0 \\[0.25em]
0 & 1 & 0 & \cdots & 0 \\[0.25em]
0 & 0 & 1 & \cdots & 0 \\[0.25em]
\vdots & \vdots & \vdots & \ddots & \vdots \\[0.25em]
0 & 0 & 0 & \cdots & 1
\end{pmatrix}
$$

whose diagonal entries are all equal to $1$.

- The corresponding linear transformation $T_{I_n} : \mathbb{R}^n \to \mathbb{R}^n$ gives the same output as input. In other words, $T_{I_n}(\mathbf{x}) = \mathbf{x}$ for every $\mathbf{x} \in \mathbb{R}^n$ (the notation $T_A$ was introduced in [Linear transformations and matrix multiplication](linear_transformations_and_matrix_multiplication.md#composing-linear-transformations-and-matrix-multiplication)).
- For any $m \times n$ matrix $A$ we have $I_mA = A = AI_n$.

It may seem that $I_n$ is hardly deserving of any attention. But for our later work with the fundamental concepts of matrix inversion (the matrix analogue of "$1/x$", essential for studying systems of linear equations) and eigenvectors, the matrix $I_n$ will be extremely convenient.

## Addition and scalar multiplication for matrices

Consider a scalar $c$ and two $m \times n$ matrices

$$
A =
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\[0.25em]
a_{21} & a_{22} & \cdots & a_{2n} \\[0.25em]
\vdots & \vdots & \ddots & \vdots \\[0.25em]
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix},
\qquad
B =
\begin{pmatrix}
b_{11} & b_{12} & \cdots & b_{1n} \\[0.25em]
b_{21} & b_{22} & \cdots & b_{2n} \\[0.25em]
\vdots & \vdots & \ddots & \vdots \\[0.25em]
b_{m1} & b_{m2} & \cdots & b_{mn}
\end{pmatrix}.
$$

**Definition:** The **matrix sum** $A + B$ is defined to be the $m \times n$ matrix

$$
A + B =
\begin{pmatrix}
a_{11} + b_{11} & a_{12} + b_{12} & \cdots & a_{1n} + b_{1n} \\[0.35em]
a_{21} + b_{21} & a_{22} + b_{22} & \cdots & a_{2n} + b_{2n} \\[0.35em]
\vdots & \vdots & \ddots & \vdots \\[0.35em]
a_{m1} + b_{m1} & a_{m2} + b_{m2} & \cdots & a_{mn} + b_{mn}
\end{pmatrix}
$$

and the **scalar multiple** $cA$ is defined to be the $m \times n$ matrix

$$
cA =
\begin{pmatrix}
ca_{11} & ca_{12} & \cdots & ca_{1n} \\[0.25em]
ca_{21} & ca_{22} & \cdots & ca_{2n} \\[0.25em]
\vdots & \vdots & \ddots & \vdots \\[0.25em]
ca_{m1} & ca_{m2} & \cdots & ca_{mn}
\end{pmatrix}.
$$

!!! note "Proposition"
    The linear transformations $T_{A+B}$ and $T_{cA}$ for the matrix sum and the scalar multiple respectively satisfy

    $$T_{A+B}(\mathbf{x}) = T_A(\mathbf{x}) + T_B(\mathbf{x}) \qquad \text{and} \qquad T_{cA}(\mathbf{x}) = c\,T_A(\mathbf{x})$$

    for all $\mathbf{x} \in \mathbb{R}^n$.

## First properties of matrix algebra

Important basic properties of matrix multiplication:

**(MM1)** It recovers matrix-vector multiplication: if $A$ is an $m \times n$ matrix, and $\mathbf{x} \in \mathbb{R}^n$ is thought of as an $n \times 1$ matrix, the matrix-matrix product $A\mathbf{x}$ is the same as the matrix-vector product.

**(MM2)** $A(B + C) = AB + AC$ and $(A' + B')C' = A'C' + B'C'$. These "distributive laws" are the reason we call it matrix multiplication. This settles the point left open in [Linear transformations and matrix multiplication](linear_transformations_and_matrix_multiplication.md#composing-linear-transformations-and-matrix-multiplication), where we noted that $AB \neq BA$ is only worth being troubled by once the name "multiplication" has been justified.

**(MM3)** $A(BC) = (AB)C$, and $A(cB) = (cA)B = c(AB)$ for any scalar $c$. In particular, taking $C$ to be an $m \times 1$ matrix that is a column vector $\mathbf{v}$ by another name,

$$A(B\mathbf{v}) = (AB)\mathbf{v}.$$

**(MM4)** If $A$ is an $m \times n$ matrix, then $I_mA = A = AI_n$, where $I_m$ is the $m \times m$ identity matrix and $I_n$ is the $n \times n$ identity matrix.

The distributive law for scalars explains "$(a + b)(c + d) = ac + ad + bc + bd$" (indeed, $(a+b)(c+d) = a(c+d) + b(c+d) = ac + ad + bc + bd$), and the same works for matrices for the same reason provided that we are careful about the order of matrix multiplication:

$$(A + B)(C + D) = A(C + D) + B(C + D) = AC + AD + BC + BD .$$

As a special case with $n \times n$ matrices,

$$(A + B)^2 = (A + B)(A + B) = A^2 + AB + BA + B^2 .$$

This is **not** $A^2 + 2AB + B^2$ except when $BA = AB$, so there is no "binomial theorem" for computing $(A + B)^m$ with general $n \times n$ matrices when $n > 1$.

Let's see why **MM3** (Associative property) holds ($A(BC) = (AB)C$).

Firstly, $BC$ is the matrix of the linear transformation $T_B \circ T_C$. Therefore $A(BC)$ is the matrix of the linear transformation $T_A \circ (T_B \circ T_C)$. This sends a vector $\mathbf{x}$ to

$$T_A\bigl((T_B \circ T_C)(\mathbf{x})\bigr) = T_A\bigl(T_B(T_C(\mathbf{x}))\bigr).$$

Going the other way, $AB$ is the matrix of the linear transformation $T_A \circ T_B$. Therefore $(AB)C$ is the matrix of the linear transformation $(T_A \circ T_B) \circ T_C$. This transformation sends an input $\mathbf{x}$ to

$$(T_A \circ T_B)\bigl(T_C(\mathbf{x})\bigr) = T_A\bigl(T_B(T_C(\mathbf{x}))\bigr).$$

So $(AB)C$ and $A(BC)$, considered as linear transformations, are the same: they take an input $\mathbf{x}$ to the output $T_A(T_B(T_C(\mathbf{x})))$. Thus $(AB)C = A(BC)$, since for any linear transformation $T(\mathbf{x}) = M\mathbf{x}$ with a matrix $M$, $T$ determines $M$ via the outputs $T(\mathbf{e}_j)$ (these are the columns of $M$).
