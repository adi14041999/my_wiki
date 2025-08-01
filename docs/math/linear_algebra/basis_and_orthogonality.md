# Basis and orthogonality

A basis for a nonzero linear subspace $V$ in $\mathbb{R}^n$ is a spanning set for $V$ consisting of exactly $\dim(V)$ vectors.

If $\dim(V) = 2$ then a basis for $V$ consists of any v, w for which $\text{span}(v,w) = V$.

One basis of $\mathbb{R}^3$ is given by $\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$, $\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$, $\mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$; this is called the standard basis of $\mathbb{R}^3$. But many other triples of vectors are also a basis of $\mathbb{R}^3$.

Although we have a way to figure out the dimension of the span of 2 or 3 nonzero vectors, we have to confront the reality that for the span of 4 or more nonzero vectors in $\mathbb{R}^n$ it becomes rather cumbersome to figure out the dimension via algebra alone; we need another way.

A collection of vectors v$_1$, . . . , v$_k$ in $\mathbb{R}^n$ is called orthogonal if $\mathbf{v}_i \cdot \mathbf{v}_j = 0$ whenever $i \neq j$. In words, the vectors are all perpendicular to one another.

If v$_1$, . . . , v$_k$ is an orthogonal collection of nonzero vectors in $\mathbb{R}^n$ then it is a basis for $\text{span}(v_1, \ldots, v_k)$. In particular, $\text{span}(v_1, \ldots, v_k)$ then has dimension $k$ and we call v$_1$, . . . , v$_k$ an orthogonal basis for its span (a single nonzero vector is always an orthogonal basis for its span!).

The span of a collection of $k$ vectors in $\mathbb{R}^n$ has dimension at most $k$ (e.g., three vectors in $\mathbb{R}^3$ lying in a common plane through 0 have span with dimension less than 3). Orthogonality is a useful way to guarantee that $k$ given nonzero $n$-vectors have a $k$-dimensional span.

**Example:** Consider the span $V$ of the following three vectors in $\mathbb{R}^5$:

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 3 \\ 2 \\ 1 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \\ 2 \\ 0 \\ 3 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 0 \\ 3 \\ 0 \\ 2 \\ 1 \end{bmatrix}$$

This collection of three vectors is not orthogonal, since, for example, $\mathbf{v}_1 \cdot \mathbf{v}_2 = 1 + 0 + 6 + 0 + 3 = 10$. We can show that $\dim(V) = 3$, so the triple $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$ is a basis of $V$ (if $\dim(V) = 2$, then the triple would not be a basis of V, just a regular spanning set), but not an orthogonal basis of $V$.

**Note:** There is a systematic process for finding an orthogonal basis for the span of $k$ vectors in $\mathbb{R}^n$ called the "Gram–Schmidt process".

Every nonzero linear subspace of $\mathbb{R}^n$ has an orthogonal basis.

There is an especially convenient type of orthogonal basis for a nonzero linear subspace of $\mathbb{R}^n$. A collection of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ in $\mathbb{R}^n$ is called **orthonormal** if they are orthogonal to each other and in addition they are all unit vectors; that is, $\mathbf{v}_i \cdot \mathbf{v}_i = 1$ for all $i$ (ensuring $\|\mathbf{v}_i\| = \sqrt{\mathbf{v}_i \cdot \mathbf{v}_i} = \sqrt{1} = 1$ for all $i$).

Any orthonormal collection of vectors is a basis of its span.

For any $n$ the analogous orthonormal collection of $n$ vectors $\mathbf{e}_1, \ldots, \mathbf{e}_n$ in $\mathbb{R}^n$ can be written down (i.e., $\mathbf{e}_i$ has its $i$th entry equal to $1$ and all other entries are $0$), and this spans $\mathbb{R}^n$; it is called the **standard basis** of $\mathbb{R}^n$, and shows $\dim(\mathbb{R}^n) = n$ (as we expect). In particular, (with $V = \mathbb{R}^n$), every linear subspace of $\mathbb{R}^n$ has dimension at most $n$ and the only $n$-dimensional one is $\mathbb{R}^n$ itself (as geometric intuition may suggest).

In the special case $n = 3$, the vectors $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3 \in \mathbb{R}^3$ are often respectively denoted as $\mathbf{i}, \mathbf{j}, \mathbf{k}$ in physics and engineering contexts.

**Example:** The triple

$$\begin{bmatrix} 1 \\ 2 \\ 4 \end{bmatrix}, \quad \begin{bmatrix} -6 \\ 1 \\ 1 \end{bmatrix}, \quad \begin{bmatrix} -2 \\ -25 \\ 13 \end{bmatrix}$$

is an orthogonal basis for $\mathbb{R}^3$. How can one "see" this? One can check by hand that it is an orthogonal collection of vectors, so this collection of $3$ nonzero vectors must be a basis of its span by, and hence its span has dimension $3$.

## Fourier formula

If $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are nonzero vectors in $\mathbb{R}^n$, by definition any vector $\mathbf{v} \in \text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k)$ can be written as a linear combination

$$\mathbf{v} = \sum_{i=1}^k c_i\mathbf{v}_i$$

for some scalars $c_1, \ldots, c_k$. If the collection of $\mathbf{v}_i$'s is orthogonal, we can actually solve for the $c_i$'s in terms of $\mathbf{v}$ by the following slick technique that has useful generalizations throughout mathematics (with Fourier series, special function theory, and so on).

For instance, if we form the dot product against $\mathbf{v}_1$ then we obtain

$$\mathbf{v} \cdot \mathbf{v}_1 = c_1(\mathbf{v}_1 \cdot \mathbf{v}_1) + c_2(\mathbf{v}_2 \cdot \mathbf{v}_1) + c_3(\mathbf{v}_3 \cdot \mathbf{v}_1) + \cdots = c_1(\mathbf{v}_1 \cdot \mathbf{v}_1),$$

where the tremendous cancellation at the final equality is precisely due to the orthogonality of the collection of $\mathbf{v}_i$'s. Since $\mathbf{v}_1$ is nonzero, so $\mathbf{v}_1 \cdot \mathbf{v}_1 = \|\mathbf{v}_1\|^2$ is nonzero, we can now divide by it at both ends of our string of equalities above to obtain

$$\frac{\mathbf{v} \cdot \mathbf{v}_1}{\mathbf{v}_1 \cdot \mathbf{v}_1} = c_1.$$

In this way we have solved for $c_1$!

The same procedure works likewise to solve for each $c_i$ via forming dot products against $\mathbf{v}_i$, yielding the general formula

$$c_i = \frac{\mathbf{v} \cdot \mathbf{v}_i}{\mathbf{v}_i \cdot \mathbf{v}_i}$$

for each $i$. Substituting back into the right side of the equation for $\mathbf{v}$, we obtain the following result.

**Theorem (Fourier formula).** For any orthogonal collection of nonzero vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ in $\mathbb{R}^n$ and vector $\mathbf{v}$ in their span,

$$\mathbf{v} = \sum_{i=1}^k \frac{\mathbf{v} \cdot \mathbf{v}_i}{\mathbf{v}_i \cdot \mathbf{v}_i} \mathbf{v}_i.$$

In particular, if the $\mathbf{v}_i$'s are all unit vectors (so $\mathbf{v}_i \cdot \mathbf{v}_i = 1$ for all $i$) then $\mathbf{v} = \sum_{i=1}^k (\mathbf{v} \cdot \mathbf{v}_i)\mathbf{v}_i$.

**Example:** For the orthonormal basis $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3, \mathbf{e}_4$ of $\mathbb{R}^4$ and any $\mathbf{v} = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \\ a_4 \end{bmatrix} \in \mathbb{R}^4$, the coefficients $\mathbf{v} \cdot \mathbf{e}_i$ work out as follows:

$$\mathbf{v} \cdot \mathbf{e}_1 = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \\ a_4 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} = a_1$$

and similarly $\mathbf{v} \cdot \mathbf{e}_i = a_i$ for each $i = 1, 2, 3, 4$. Thus (since $\mathbf{e}_i \cdot \mathbf{e}_i = 1$), $\mathbf{v} = \sum_{i=1}^4 (\mathbf{v} \cdot \mathbf{e}_i)\mathbf{e}_i = \sum_{i=1}^4 a_i\mathbf{e}_i$. Unpacking the summation notation, this is just asserting

$$\begin{bmatrix} a_1 \\ a_2 \\ a_3 \\ a_4 \end{bmatrix} = a_1\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + a_2\begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} + a_3\begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix} + a_4\begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix},$$

which can be directly verified by hand since the right side is exactly

$$\begin{bmatrix} a_1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ a_2 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ a_3 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 0 \\ a_4 \end{bmatrix}.$$

In other words, the Fourier formula in the special case that $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is the orthonormal basis $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ of $\mathbb{R}^n$ is precisely the familiar fact that any vector in $\mathbb{R}^n$ can be decomposed as the sum of its "components" along the various coordinate directions. This is neither surprising nor perhaps particularly interesting, so we next give a more "typical" example.

**Example:** Consider the span $V$ of the following three vectors in $\mathbb{R}^5$:

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 3 \\ 2 \\ 1 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \\ 2 \\ 0 \\ 3 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 0 \\ 3 \\ 0 \\ 2 \\ 1 \end{bmatrix}$$

Consider the following three nonzero vectors in $V$, which form an orthogonal basis for their span:

$$\mathbf{w}_1 = \mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 3 \\ 2 \\ 1 \end{bmatrix}, \quad \mathbf{w}_2 = -2\mathbf{v}_1 + 3\mathbf{v}_2 = \begin{bmatrix} 1 \\ 3 \\ 0 \\ -4 \\ 7 \end{bmatrix}, \quad \mathbf{w}_3 = -9\mathbf{v}_1 - 24\mathbf{v}_2 + 75\mathbf{v}_3 = \begin{bmatrix} -33 \\ 201 \\ -75 \\ 132 \\ -6 \end{bmatrix}$$

Consider the vector

$$\mathbf{v} = 2\mathbf{v}_1 - \mathbf{v}_2 + \mathbf{v}_3 = \begin{bmatrix} 2 \\ 0 \\ 6 \\ 4 \\ 2 \end{bmatrix} - \begin{bmatrix} 1 \\ 1 \\ 2 \\ 0 \\ 3 \end{bmatrix} + \begin{bmatrix} 0 \\ 3 \\ 0 \\ 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 4 \\ 6 \\ 0 \end{bmatrix}$$

in $V$. Since $\{\mathbf{w}_1, \mathbf{w}_2, \mathbf{w}_3\}$ is a basis of $V$, we know that there is some expression of the form

$$\mathbf{v} = c_1\mathbf{w}_1 + c_2\mathbf{w}_2 + c_3\mathbf{w}_3$$

for unknown scalars $c_1, c_2, c_3$. What are these scalars? A brute-force approach would be to write everything out as explicit vectors to obtain

$$\begin{bmatrix} 1 \\ 2 \\ 4 \\ 6 \\ 0 \end{bmatrix} = \mathbf{v} = c_1\mathbf{w}_1 + c_2\mathbf{w}_2 + c_3\mathbf{w}_3 = c_1\begin{bmatrix} 1 \\ 0 \\ 3 \\ 2 \\ 1 \end{bmatrix} + c_2\begin{bmatrix} 1 \\ 3 \\ 0 \\ -4 \\ 7 \end{bmatrix} + c_3\begin{bmatrix} -33 \\ 201 \\ -75 \\ 132 \\ -6 \end{bmatrix} = \begin{bmatrix} c_1 + c_2 - 33c_3 \\ 3c_2 + 201c_3 \\ 3c_1 - 75c_3 \\ 2c_1 - 4c_2 + 132c_3 \\ c_1 + 7c_2 - 6c_3 \end{bmatrix},$$

and then equate corresponding vector entries on the left and right sides to get a huge system of $5$ equations in $3$ unknowns. We can entirely bypass that by computing dot products for our specific $\mathbf{v}$!

To carry this out, we use the explicit descriptions of $\mathbf{v}$ and the $\mathbf{w}_i$'s to compute

$$\mathbf{v} \cdot \mathbf{w}_1 = 25, \quad \mathbf{v} \cdot \mathbf{w}_2 = -17, \quad \mathbf{v} \cdot \mathbf{w}_3 = 861,$$

so the Fourier formula says for this particular $\mathbf{v}$ that

$$\mathbf{v} = \frac{25}{15}\mathbf{w}_1 - \frac{17}{75}\mathbf{w}_2 + \frac{861}{64575}\mathbf{w}_3 = \frac{5}{3}\mathbf{w}_1 - \frac{17}{75}\mathbf{w}_2 + \frac{1}{75}\mathbf{w}_3.$$

That's it! This is the expression for $\mathbf{v}$ as a linear combination of the orthogonal basis $\{\mathbf{w}_1, \mathbf{w}_2, \mathbf{w}_3\}$ of $V$.