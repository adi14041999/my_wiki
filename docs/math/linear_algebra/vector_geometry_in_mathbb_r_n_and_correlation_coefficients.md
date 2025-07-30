# Vector geometry in $\mathbb{R}^n$ and correlation coefficients

## Angles

The angle $0° \leq \theta \leq 180°$ between nonzero 2-vectors a = $(a_1, a_2)$ and b = $(b_1, b_2)$ satisfies

$$\cos \theta = \frac{a_1b_1 + a_2b_2}{\|a\|\|b\|}$$

The angle $0° \leq \theta \leq 180°$ between two nonzero 3-vectors a = $(a_1, a_2, a_3)$ and b = $(b_1, b_2, b_3)$ satisfies

$$\cos \theta = \frac{a_1b_1 + a_2b_2 + a_3b_3}{\|a\|\|b\|}$$

The preceding in $\mathbb{R}^2$ and $\mathbb{R}^3$ motivates how to define appropriate concepts with $n$-vectors for any $n$.

Consider $n$-vectors x = $\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$ and y = $\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$.

(i) The dot product of x and y is defined to be the scalar

$$x \cdot y = x_1y_1 + x_2y_2 + \cdots + x_ny_n = \sum_{i=1}^n x_iy_i$$

The dot product is only defined if the two vectors are $n$-vectors for the same value of $n$.

(ii) The angle $\theta$ between two nonzero $n$-vectors x, y is defined by the formula

$$\cos(\theta) = \frac{x \cdot y}{\|x\|\|y\|} \tag{2.1.4}$$

with $0° \leq \theta \leq 180°$. For emphasis: x and y must be nonzero $n$-vectors for a common $n$.

(iii) When $x \cdot y = 0$ (same as $\theta = 90°$ if x, y ≠ 0), we say x and y are perpendicular; the word orthogonal is often used for this ("orthogōnios" is ancient Greek for "right-angled"), though only rarely at the U.S. Supreme Court.

Always remember that the dot product of vectors is a scalar (it is not a vector).

The notion of angle is a definition in $\mathbb{R}^n$ for general $n$: it is motivated by the case when $n = 3$, but for general $n$ there is nothing to "physically justify". The real content making this definition for general $n$ is that (as you will learn with experience) this notion of angle behaves like our visual experience in $\mathbb{R}^2$ and $\mathbb{R}^3$ and so provides useful visual guidance with $n$-vectors for any $n$.

Whenever we speak of an angle between two lines through the origin, there is always an ambiguity (when they're not perpendicular) of whether we want the acute angle between them or the (supplementary) obtuse angle between them. This corresponds to the fact that when we set it up as a vector problem, we have to choose a direction along each line (coming out of the intersection point). Depending on the choice, we will get the acute or the obtuse angle.

## Properties of dot products
