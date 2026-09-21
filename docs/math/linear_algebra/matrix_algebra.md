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

## Exercises

**1.** If a matrix $A$ is square (i.e. size $n \times n$), then we can multiply $A$ by itself, so the products $A^2 = AA$, $A^3 = AAA$, $\ldots$, and so on are all defined. The operations of matrix algebra defined then allow us to define how to "evaluate" a single-variable polynomial $f(t)$ at $A$, as follows. If $f(t) = c_2t^2 + c_1t + c_0$, then we define the $n \times n$ matrix $f(A)$ to be

$$f(A) = c_2A^2 + c_1A + c_0I_n .$$

A similar recipe is used if $f$ is a higher-degree polynomial (but we won't need it for this exercise). Perhaps surprisingly, this concept is very broadly useful, such as in the more advanced study of algebraic properties of matrices and for differential equations.

Note the role of $I_n$: the constant term $c_0$ is a scalar, and a scalar cannot be added to a matrix, so it must be converted into the matrix $c_0I_n$ first. This is one of the conveniences promised when $I_n$ was introduced.

**(a).** Let $f(t) = 2t^2 + 3t - 1$, and

$$
A = \begin{pmatrix} 1 & 2 \\[0.25em] -5 & 2 \end{pmatrix},
\qquad
B = \begin{pmatrix} 2 & 1 & 0 \\[0.25em] 0 & -1 & 0 \\[0.25em] 0 & 0 & 3 \end{pmatrix}.
$$

Compute $f(A)$ and $f(B)$.

**Solution:** In each case compute the square first, then assemble the three pieces.

*For $A$.* Multiplying out,

$$
A^2 = \begin{pmatrix} 1 & 2 \\[0.25em] -5 & 2 \end{pmatrix}\begin{pmatrix} 1 & 2 \\[0.25em] -5 & 2 \end{pmatrix}
= \begin{pmatrix} 1 - 10 & 2 + 4 \\[0.25em] -5 - 10 & -10 + 4 \end{pmatrix}
= \begin{pmatrix} -9 & 6 \\[0.25em] -15 & -6 \end{pmatrix}.
$$

Hence

$$
f(A) = 2A^2 + 3A - I_2
= \begin{pmatrix} -18 & 12 \\[0.25em] -30 & -12 \end{pmatrix}
+ \begin{pmatrix} 3 & 6 \\[0.25em] -15 & 6 \end{pmatrix}
+ \begin{pmatrix} -1 & 0 \\[0.25em] 0 & -1 \end{pmatrix}
= \begin{pmatrix} -16 & 18 \\[0.25em] -45 & -7 \end{pmatrix}.
$$

*For $B$.* Multiplying out,

$$
B^2 = \begin{pmatrix} 2 & 1 & 0 \\[0.25em] 0 & -1 & 0 \\[0.25em] 0 & 0 & 3 \end{pmatrix}\begin{pmatrix} 2 & 1 & 0 \\[0.25em] 0 & -1 & 0 \\[0.25em] 0 & 0 & 3 \end{pmatrix}
= \begin{pmatrix} 4 & 1 & 0 \\[0.25em] 0 & 1 & 0 \\[0.25em] 0 & 0 & 9 \end{pmatrix}.
$$

Hence

$$
f(B) = 2B^2 + 3B - I_3
= \begin{pmatrix} 8 & 2 & 0 \\[0.25em] 0 & 2 & 0 \\[0.25em] 0 & 0 & 18 \end{pmatrix}
+ \begin{pmatrix} 6 & 3 & 0 \\[0.25em] 0 & -3 & 0 \\[0.25em] 0 & 0 & 9 \end{pmatrix}
+ \begin{pmatrix} -1 & 0 & 0 \\[0.25em] 0 & -1 & 0 \\[0.25em] 0 & 0 & -1 \end{pmatrix}
= \begin{pmatrix} 13 & 5 & 0 \\[0.25em] 0 & -2 & 0 \\[0.25em] 0 & 0 & 26 \end{pmatrix}.
$$

There is a useful check available for $B$. It is triangular, with all entries below the diagonal equal to $0$, and such matrices keep that shape under multiplication with diagonal entries multiplying separately. So the diagonal of $f(B)$ should just be $f$ applied to each diagonal entry of $B$:

$$f(2) = 8 + 6 - 1 = 13, \qquad f(-1) = 2 - 3 - 1 = -2, \qquad f(3) = 18 + 9 - 1 = 26,$$

matching the diagonal computed above.

**(b).** Let $g(t) = t^2 - 3t + 12$. For $A$ as in part (a), check that $g(A) = 0$ (the $2 \times 2$ zero matrix).

**Solution:** We already have $A^2$, so

$$
g(A) = A^2 - 3A + 12I_2
= \begin{pmatrix} -9 & 6 \\[0.25em] -15 & -6 \end{pmatrix}
+ \begin{pmatrix} -3 & -6 \\[0.25em] 15 & -6 \end{pmatrix}
+ \begin{pmatrix} 12 & 0 \\[0.25em] 0 & 12 \end{pmatrix}
= \begin{pmatrix} 0 & 0 \\[0.25em] 0 & 0 \end{pmatrix}.
$$

**(c).** Let $h(t) = t^2 + 2t + 1$ and $C = \begin{pmatrix} 3 & -1 \\ 0 & 1 \end{pmatrix}$. Notice that $h(t) = (t + 1)^2$. Verify (by computing both sides separately) that $h(C) = (C + I_2)(C + I_2)$ as $2 \times 2$ matrices.

**Solution:** *Left side.* First

$$
C^2 = \begin{pmatrix} 3 & -1 \\[0.25em] 0 & 1 \end{pmatrix}\begin{pmatrix} 3 & -1 \\[0.25em] 0 & 1 \end{pmatrix}
= \begin{pmatrix} 9 & -3 - 1 \\[0.25em] 0 & 1 \end{pmatrix}
= \begin{pmatrix} 9 & -4 \\[0.25em] 0 & 1 \end{pmatrix},
$$

so

$$
h(C) = C^2 + 2C + I_2
= \begin{pmatrix} 9 & -4 \\[0.25em] 0 & 1 \end{pmatrix}
+ \begin{pmatrix} 6 & -2 \\[0.25em] 0 & 2 \end{pmatrix}
+ \begin{pmatrix} 1 & 0 \\[0.25em] 0 & 1 \end{pmatrix}
= \begin{pmatrix} 16 & -6 \\[0.25em] 0 & 4 \end{pmatrix}.
$$

*Right side.* Separately,

$$
C + I_2 = \begin{pmatrix} 4 & -1 \\[0.25em] 0 & 2 \end{pmatrix},
\qquad
(C + I_2)(C + I_2) = \begin{pmatrix} 4 & -1 \\[0.25em] 0 & 2 \end{pmatrix}\begin{pmatrix} 4 & -1 \\[0.25em] 0 & 2 \end{pmatrix}
= \begin{pmatrix} 16 & -4 - 2 \\[0.25em] 0 & 4 \end{pmatrix}
= \begin{pmatrix} 16 & -6 \\[0.25em] 0 & 4 \end{pmatrix}.
$$

The two sides agree, as required.

It is worth asking why the factorization $h(t) = (t+1)^2$ survived the passage to matrices, given the earlier warning that $(X + Y)^2 = X^2 + XY + YX + Y^2$ is not $X^2 + 2XY + Y^2$ in general. The expansion here is

$$(C + I_2)^2 = C^2 + CI_2 + I_2C + I_2^2 ,$$

and the middle two terms are *both* equal to $C$ by the identity property, so they combine into $2C$ with no commutativity issue. That is the special feature of $I_2$: it commutes with every matrix.

**2.** Let

$$
A = \begin{pmatrix}
2 & 3 & -4 \\[0.25em]
4 & -1 & 6 \\[0.25em]
5 & 2 & 1
\end{pmatrix}.
$$

Find a non-zero $3 \times 3$ matrix $M$ for which $AM$ is the $3 \times 3$ zero matrix (there are many valid solutions).

**Solution:** The key is to remember how the columns of a product behave: the $j$th column of $AM$ is $A$ applied to the $j$th column of $M$. So $AM$ is the zero matrix exactly when **every column of $M$ is a vector that $A$ sends to $\mathbf{0}$.** The problem therefore reduces to finding a single nonzero $3$-vector $\mathbf{v}$ with $A\mathbf{v} = \mathbf{0}$, after which we may use $\mathbf{v}$ to build the columns of $M$.

*Finding such a $\mathbf{v}$.* Writing $\mathbf{v} = (x, y, z)$, the condition $A\mathbf{v} = \mathbf{0}$ says

$$
2x + 3y - 4z = 0, \qquad
4x - y + 6z = 0, \qquad
5x + 2y + z = 0 .
$$

The solutions are exactly the multiples of $(-1, 2, 1)$, obtained by taking $z = 1$. As a check,

$$
A\begin{pmatrix} -1 \\[0.25em] 2 \\[0.25em] 1 \end{pmatrix}
= \begin{pmatrix} -2 + 6 - 4 \\[0.25em] -4 - 2 + 6 \\[0.25em] -5 + 4 + 1 \end{pmatrix}
= \begin{pmatrix} 0 \\[0.25em] 0 \\[0.25em] 0 \end{pmatrix}.
$$

*Building $M$:* Put $\mathbf{v}$ in the first column and fill the rest with zeros:

$$
M = \begin{pmatrix}
-1 & 0 & 0 \\[0.25em]
2 & 0 & 0 \\[0.25em]
1 & 0 & 0
\end{pmatrix}.
$$

This $M$ is nonzero, and $AM$ has first column $A\mathbf{v} = \mathbf{0}$ and second and third columns $A\mathbf{0} = \mathbf{0}$, so $AM$ is the zero matrix.

**3.** Let $T : \mathbb{R}^2 \to \mathbb{R}^2$ be the linear transformation that, on an input $\mathbf{x}$, outputs the sum of the rotation counterclockwise by $45°$ of $\mathbf{x}$ and twice $\mathbf{x}$. That is, $T(\mathbf{x}) = R_{\pi/4}(\mathbf{x}) + 2\mathbf{x}$. Find the matrix $A$ for $T$.

**Solution:** Recall that $T_{A+B}(\mathbf{x}) = T_A(\mathbf{x}) + T_B(\mathbf{x})$ and $T_{cA}(\mathbf{x}) = c\,T_A(\mathbf{x})$. Reading those from right to left lets us assemble the matrix of $T$ out of the matrices of its two pieces.

The first piece is rotation counterclockwise by $\pi/4$, whose matrix is

$$
A_{\pi/4} = \begin{pmatrix} \cos(\pi/4) & -\sin(\pi/4) \\[0.35em] \sin(\pi/4) & \cos(\pi/4) \end{pmatrix}
= \begin{pmatrix} 1/\sqrt{2} & -1/\sqrt{2} \\[0.35em] 1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}.
$$

The second piece is the map $\mathbf{x} \mapsto 2\mathbf{x}$, which is $2$ times the identity transformation, so its matrix is $2I_2$. Therefore

$$
A = A_{\pi/4} + 2I_2
= \begin{pmatrix} 1/\sqrt{2} & -1/\sqrt{2} \\[0.35em] 1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}
+ \begin{pmatrix} 2 & 0 \\[0.35em] 0 & 2 \end{pmatrix}
= \begin{pmatrix} 2 + \dfrac{1}{\sqrt{2}} & -\dfrac{1}{\sqrt{2}} \\[0.9em] \dfrac{1}{\sqrt{2}} & 2 + \dfrac{1}{\sqrt{2}} \end{pmatrix},
$$

where $1/\sqrt{2} = \sqrt{2}/2 \approx 0.707$ if a decimal form is wanted.

*Check by columns:* The columns of the matrix of a linear transformation are its values on $\mathbf{e}_1$ and $\mathbf{e}_2$, so we can confirm the answer without using matrix addition at all. Rotating $\mathbf{e}_1$ by $45°$ gives $(1/\sqrt{2},\ 1/\sqrt{2})$, and doubling $\mathbf{e}_1$ gives $(2, 0)$, so

$$T(\mathbf{e}_1) = \begin{pmatrix} 1/\sqrt{2} \\[0.35em] 1/\sqrt{2} \end{pmatrix} + \begin{pmatrix} 2 \\[0.35em] 0 \end{pmatrix} = \begin{pmatrix} 2 + 1/\sqrt{2} \\[0.35em] 1/\sqrt{2} \end{pmatrix},$$

which is the first column above. Rotating $\mathbf{e}_2$ gives $(-1/\sqrt{2},\ 1/\sqrt{2})$, and doubling $\mathbf{e}_2$ gives $(0, 2)$, so

$$T(\mathbf{e}_2) = \begin{pmatrix} -1/\sqrt{2} \\[0.35em] 1/\sqrt{2} \end{pmatrix} + \begin{pmatrix} 0 \\[0.35em] 2 \end{pmatrix} = \begin{pmatrix} -1/\sqrt{2} \\[0.35em] 2 + 1/\sqrt{2} \end{pmatrix},$$

the second column. The two methods agree.

**4.** Let $D$ be a $5 \times 5$ diagonal matrix with diagonal entries $d_1, d_2, \ldots, d_5 \in \mathbb{R}$ (from top left to bottom right). Suppose $A$ is a $5 \times 5$ matrix whose second and fourth columns, respectively, are

$$
\begin{pmatrix} 2 \\[0.25em] -1 \\[0.25em] 0 \\[0.25em] 3 \\[0.25em] 4 \end{pmatrix}
\qquad \text{and} \qquad
\begin{pmatrix} 1 \\[0.25em] 8 \\[0.25em] -9 \\[0.25em] 3 \\[0.25em] 1 \end{pmatrix}.
$$

For each of the following, compute it in terms of the $d_i$'s or explain why there is not enough information given to do so.

**(a).** The second column of $DA$.

**Solution:** The second column of $DA$ is $D$ applied to the second column of $A$, which we are given. Multiplying by a diagonal matrix scales the $i$th entry of a vector by $d_i$, so

$$
D\begin{pmatrix} 2 \\[0.25em] -1 \\[0.25em] 0 \\[0.25em] 3 \\[0.25em] 4 \end{pmatrix}
= \begin{pmatrix} 2d_1 \\[0.25em] -d_2 \\[0.25em] 0 \\[0.25em] 3d_4 \\[0.25em] 4d_5 \end{pmatrix}.
$$

There is enough information. Note that $d_3$ does not appear, since the entry it scales is $0$.

**(b).** The fourth column of $AD$.

**Solution:** Again the fourth column of $AD$ is $A$ applied to the fourth column of $D$. Since $D$ is diagonal, its fourth column has $d_4$ in the fourth slot and $0$ everywhere else, so it equals $d_4\mathbf{e}_4$. Here $d_4$ is a single real number (a scalar) and $\mathbf{e}_4$ is the $5 \times 1$ column vector with a $1$ in the fourth entry and $0$ elsewhere, so $d_4\mathbf{e}_4$ is a scalar multiple of a vector, not a product of two matrices. Pulling that scalar out through $A$ is allowed by linearity, and therefore

$$A(d_4\mathbf{e}_4) = d_4\,(A\mathbf{e}_4) = d_4 \cdot (\text{fourth column of } A),$$

using that $A\mathbf{e}_4$ is the fourth column of $A$. That column is given, so

$$
d_4 \begin{pmatrix} 1 \\[0.25em] 8 \\[0.25em] -9 \\[0.25em] 3 \\[0.25em] 1 \end{pmatrix}
= \begin{pmatrix} d_4 \\[0.25em] 8d_4 \\[0.25em] -9d_4 \\[0.25em] 3d_4 \\[0.25em] d_4 \end{pmatrix}.
$$

There is enough information here too, and notice that only the single number $d_4$ is involved.

**5.** We defined an $m \times n$ matrix $U$ to be **upper triangular** if all its entries below the diagonal vanish: $u_{ij} = 0$ when $i > j$. Examples include:

$$
\begin{pmatrix} 2 & 1 \\[0.25em] 0 & -3 \end{pmatrix},
\qquad
\begin{pmatrix} 1 & -3 & 1 \\[0.25em] 0 & 5 & 2 \\[0.25em] 0 & 0 & 7 \end{pmatrix},
\qquad
\begin{pmatrix} 2 & 4 \\[0.25em] 0 & 9 \\[0.25em] 0 & 0 \end{pmatrix},
\qquad
\begin{pmatrix} 2 & 1 & 1 & -3 \\[0.25em] 0 & -1 & 0 & 5 \\[0.25em] 0 & 0 & 6 & 3 \\[0.25em] 0 & 0 & 0 & -1 \end{pmatrix}
$$

There is no requirement on diagonal entries or on entries above the diagonal, so some of them could vanish; e.g., an $n \times n$ zero matrix is upper triangular. The only requirement is that entries below the diagonal must vanish.

**(a).** For $1 \leq i \leq n$, let $V_i = \text{span}(\mathbf{e}_1, \ldots, \mathbf{e}_i)$ be the span of the first $i$ standard basis vectors of $\mathbb{R}^n$ (equivalently: $V_i$ consists of vectors whose entries beyond the $i$th position all vanish); define $V_i = \mathbb{R}^n$ for $i > n$. Letting $\mathbf{e}'_1, \ldots, \mathbf{e}'_m$ denote the standard basis of $\mathbb{R}^m$, likewise define $V'_i = \text{span}(\mathbf{e}'_1, \ldots, \mathbf{e}'_i)$ for $1 \leq i \leq m$ (and $V'_i = \mathbb{R}^m$ for $i > m$). If $L : \mathbb{R}^n \to \mathbb{R}^m$ is a linear transformation and $A$ is the corresponding $m \times n$ matrix, use the relationship between the columns of $A$ and the vectors $L(\mathbf{e}_i)$ to explain why $A$ is upper triangular precisely when $L$ carries $V_i$ into $V'_i$ for every $i$.

The conventions for $i$ beyond $n$ or $m$ are there so that the phrase "for every $i$" makes sense when the matrix is not square.

**Solution:** Before any general argument, look at what the condition says on a concrete matrix. Take the $3 \times 3$ example from the list above,

$$
A = \begin{pmatrix} 1 & -3 & 1 \\[0.25em] 0 & 5 & 2 \\[0.25em] 0 & 0 & 7 \end{pmatrix},
$$

and read off its columns, which are the vectors $L(\mathbf{e}_1), L(\mathbf{e}_2), L(\mathbf{e}_3)$:

$$
L(\mathbf{e}_1) = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} \in V'_1,
\qquad
L(\mathbf{e}_2) = \begin{pmatrix} -3 \\ 5 \\ 0 \end{pmatrix} \in V'_2,
\qquad
L(\mathbf{e}_3) = \begin{pmatrix} 1 \\ 2 \\ 7 \end{pmatrix} \in V'_3 .
$$

Each column is allowed one more nonzero slot than the last, and the staircase of zeros below the diagonal is exactly that pattern

In general the $j$th column of $A$ is $L(\mathbf{e}_j)$. Saying that all entries below the diagonal vanish says that in each column $j$, the entries in positions $i > j$ are zero, so the only entries of $L(\mathbf{e}_j)$ that may be nonzero are those in positions $1, 2, \ldots, j$. A vector of $\mathbb{R}^m$ whose entries vanish past position $j$ is precisely a vector of $V'_j$. Hence

$$A \text{ is upper triangular} \iff L(\mathbf{e}_j) \in V'_j \text{ for every } j. \tag{$\ast$}$$

What remains is to upgrade $(\ast)$, which only talks about the special vectors $\mathbf{e}_j$, into the statement about all of $V_i$ at once.

One direction is immediate. If $L$ carries $V_i$ into $V'_i$ for every $i$, then since $\mathbf{e}_j$ is itself a vector in $V_j$, we get $L(\mathbf{e}_j) \in V'_j$, which is $(\ast)$.

For the other direction, suppose $(\ast)$ holds and take any vector in $V_i$. Let us do this with the example first, with $i = 2$. A vector of $V_2$ looks like $\mathbf{x} = (c_1, c_2, 0)$, that is $\mathbf{x} = c_1\mathbf{e}_1 + c_2\mathbf{e}_2$, so by linearity

$$L(\mathbf{x}) = c_1L(\mathbf{e}_1) + c_2L(\mathbf{e}_2) = c_1\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} + c_2\begin{pmatrix} -3 \\ 5 \\ 0 \end{pmatrix} = \begin{pmatrix} c_1 - 3c_2 \\ 5c_2 \\ 0 \end{pmatrix},$$

whose third entry is $0$ no matter what $c_1$ and $c_2$ are. So $L(\mathbf{x}) \in V'_2$. The third entry had to vanish because it was $0$ in *both* of the two columns being combined.

The general argument is the same sentence with indices. Any $\mathbf{x} \in V_i$ can be written $\mathbf{x} = c_1\mathbf{e}_1 + \cdots + c_i\mathbf{e}_i$, precisely because its entries past position $i$ vanish, so

$$L(\mathbf{x}) = c_1L(\mathbf{e}_1) + \cdots + c_iL(\mathbf{e}_i).$$

Two facts finish it. First, the subspaces are **nested**,

$$V'_1 \subseteq V'_2 \subseteq V'_3 \subseteq \cdots,$$

since allowing more nonzero slots only enlarges the collection; so each term $L(\mathbf{e}_j)$, which lies in $V'_j$ by $(\ast)$, also lies in the larger $V'_i$ whenever $j \leq i$. Second, $V'_i$ is a [linear subspace](span_subspaces_and_dimension.md#span-and-linear-subspaces), hence closed under sums and scalar multiples, so the whole combination stays inside $V'_i$. Therefore $L(\mathbf{x}) \in V'_i$, and since $\mathbf{x} \in V_i$ was arbitrary, $L$ carries $V_i$ into $V'_i$.

Together with $(\ast)$, the two directions give the claim.

**(b).** Using (a) and the relationship between matrix multiplication and composition of linear functions, show that a product $U_1U_2$ of an upper triangular $m \times n$ matrix $U_1$ and an upper triangular $n \times p$ matrix $U_2$ is an upper triangular $m \times p$ matrix.

**Solution:** Write $L_1 = T_{U_1} : \mathbb{R}^n \to \mathbb{R}^m$ and $L_2 = T_{U_2} : \mathbb{R}^p \to \mathbb{R}^n$. Since matrix multiplication was defined to express composition, $U_1U_2$ is the $m \times p$ matrix of $L_1 \circ L_2 : \mathbb{R}^p \to \mathbb{R}^m$. Three spaces are involved, so name the staircase of subspaces in each:

$$V_i \subseteq \mathbb{R}^p, \qquad V'_i \subseteq \mathbb{R}^n, \qquad V''_i \subseteq \mathbb{R}^m,$$

each the span of the first $i$ standard basis vectors of its own space (with the conventions of part (a) once $i$ runs past the dimension).

Part (a) applied to each factor says

$$L_2(V_i) \subseteq V'_i \quad \text{for every } i, \qquad L_1(V'_i) \subseteq V''_i \quad \text{for every } i .$$

Now fix an $i$ and follow a vector $\mathbf{x} \in V_i$ through the two stages.

**Stage 1:** Apply $L_2$. Because $L_2(V_i) \subseteq V'_i$, the output $L_2(\mathbf{x})$ lands somewhere in $V'_i$.

**Stage 2:** Apply $L_1$ to that output. The second fact says $L_1$ sends *every* vector of $V'_i$ into $V''_i$. Our vector $L_2(\mathbf{x})$ is one of those vectors, by Stage 1. So $L_1(L_2(\mathbf{x})) \in V''_i$.

Since $(L_1 \circ L_2)(\mathbf{x})$ *means* $L_1(L_2(\mathbf{x}))$, we have shown that $(L_1 \circ L_2)(\mathbf{x}) \in V''_i$ for every $\mathbf{x} \in V_i$, which is exactly the statement

$$(L_1 \circ L_2)(V_i) \subseteq V''_i .$$

The same reasoning written as a chain of sets, rather than by chasing one vector, is

$$(L_1 \circ L_2)(V_i) = L_1\bigl(L_2(V_i)\bigr) \subseteq L_1(V'_i) \subseteq V''_i ,$$

where the first equality is the definition of composition, the middle inclusion holds because $L_2(V_i)$ sits inside $V'_i$ and feeding a *smaller* set into $L_1$ produces a smaller output set, and the last inclusion is the second fact above.

Since $i$ was arbitrary, $L_1 \circ L_2$ carries $V_i$ into $V''_i$ for every $i$. Applying part (a) in the reverse direction to the composite, its matrix $U_1U_2$ is upper triangular. $\blacksquare$

Watch it happen with

$$
U_1 = \begin{pmatrix} 2 & 1 \\[0.25em] 0 & -3 \end{pmatrix},
\qquad
U_2 = \begin{pmatrix} 1 & 4 \\[0.25em] 0 & 5 \end{pmatrix},
\qquad
U_1U_2 = \begin{pmatrix} 2 & 13 \\[0.25em] 0 & -15 \end{pmatrix}.
$$

Track $\mathbf{e}_1$ through the two stages: $U_2$ sends it to $(1, 0)$, which lies in the first-slot-only subspace, and then $U_1$ sends $(1,0)$ to $(2, 0)$, still first-slot-only.

**6.** Let

$$
C = \begin{pmatrix}
3 & -2 & 2 \\[0.25em]
-6 & 4 & -4 \\[0.25em]
12 & -8 & 8
\end{pmatrix}.
$$

Find a $3 \times 1$ matrix $A$ and a $1 \times 3$ matrix $B$ for which $C = AB$.

**Solution:** First see what a product of this shape looks like. Write

$$
A = \begin{pmatrix} a_1 \\[0.25em] a_2 \\[0.25em] a_3 \end{pmatrix},
\qquad
B = \begin{pmatrix} b_1 & b_2 & b_3 \end{pmatrix}.
$$

Thus,

$$
AB = \begin{pmatrix}
a_1b_1 & a_1b_2 & a_1b_3 \\[0.25em]
a_2b_1 & a_2b_2 & a_2b_3 \\[0.25em]
a_3b_1 & a_3b_2 & a_3b_3
\end{pmatrix}.
$$

Look at the rows. The first row is $a_1(b_1, b_2, b_3)$, the second is $a_2(b_1, b_2, b_3)$, and the third is $a_3(b_1, b_2, b_3)$. So **every row of $AB$ is a multiple of the single row $B$**, and the multipliers are the entries of $A$.

We can spot that:

$$
A = \begin{pmatrix} 1 \\[0.25em] -2 \\[0.25em] 4 \end{pmatrix},
\qquad
B = \begin{pmatrix} 3 & -2 & 2 \end{pmatrix}.
$$