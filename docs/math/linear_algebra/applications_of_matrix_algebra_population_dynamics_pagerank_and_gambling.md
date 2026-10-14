# Applications of matrix algebra- population dynamics and gambling

## Bird migration

Imagine there are three islands inhabited by birds. An ornithologist studying the birds observes the following migration pattern:

- All the birds on island $A$ migrate to another island each year. One half migrates to island $B$, the other half migrates to island $C$.
- Similarly, all the birds on island $B$ migrate to another island each year. One half migrates to island $A$, and the other half migrates to island $C$.
- Finally, one-third of the birds on island $C$ stay in place each year, one third migrate to island $A$, and one third migrate to island $B$.

![img](birds.png)

Initially, there are $10000$ birds on each island. What happens after $20$ years (assuming no births or deaths)?

Denote the number of birds on islands $A$, $B$, $C$ after $k$ years as $A_k$, $B_k$, $C_k$ respectively (so $A_0$, $B_0$, $C_0$ are all equal to $10000$, corresponding to the initial population of $10000$ birds on each island). Translating the above text into equations, it says that the populations after $k+1$ years are given as follows:

$$
\begin{aligned}
A_{k+1} &= \tfrac{1}{2}B_k + \tfrac{1}{3}C_k, \\[0.35em]
B_{k+1} &= \tfrac{1}{2}A_k + \tfrac{1}{3}C_k, \\[0.35em]
C_{k+1} &= \tfrac{1}{2}A_k + \tfrac{1}{2}B_k + \tfrac{1}{3}C_k.
\end{aligned}
$$

We can express this system of $3$ equations in terms of a matrix-vector product:

$$
\begin{pmatrix} A_{k+1} \\[0.35em] B_{k+1} \\[0.35em] C_{k+1} \end{pmatrix}
=
\begin{pmatrix}
0 & 1/2 & 1/3 \\[0.35em]
1/2 & 0 & 1/3 \\[0.35em]
1/2 & 1/2 & 1/3
\end{pmatrix}
\begin{pmatrix} A_k \\[0.35em] B_k \\[0.35em] C_k \end{pmatrix}.
$$

Let us call the $3 \times 3$ matrix here $M$, and write $\mathbf{n}_k$ to denote $\begin{pmatrix} A_k \\ B_k \\ C_k \end{pmatrix}$ (e.g., the initial $3$-vector $\mathbf{n}_0$ has all entries equal to $10000$), so the displayed equation says

$$\mathbf{n}_{k+1} = M\mathbf{n}_k \qquad \text{for } k = 0, 1, 2, 3, \ldots$$

The entries in each column of $M$ are non-negative and sum to $1$; any square matrix with such properties is called a **Markov matrix** (see also [Markov chains](../probability/markov_chains.md)).

Now matrix multiplication enters: feeding the relation "$\mathbf{n}_{k+1} = M\mathbf{n}_k$" into itself repeatedly yields

$$
\mathbf{n}_{20} = M\mathbf{n}_{19} = M(M\mathbf{n}_{18}) = M^2\mathbf{n}_{18} = M^2(M\mathbf{n}_{17}) = M^3\mathbf{n}_{17} = \cdots = M^{20}\mathbf{n}_0 = M^{20}\begin{pmatrix} 10000 \\[0.35em] 10000 \\[0.35em] 10000 \end{pmatrix}.
$$

Fortunately, a computer can rapidly compute $M^{20}$.

Your computer says

$$
M^{20} \approx
\begin{pmatrix}
0.286 & 0.286 & 0.286 \\[0.35em]
0.286 & 0.286 & 0.286 \\[0.35em]
0.429 & 0.429 & 0.429
\end{pmatrix},
$$

and so

$$
M^{20}\begin{pmatrix} 10000 \\[0.35em] 10000 \\[0.35em] 10000 \end{pmatrix}
\approx
\begin{pmatrix} 8570 \\[0.35em] 8570 \\[0.35em] 12860 \end{pmatrix}.
$$

Hence, there are approximately $8600$ birds on each of islands $A$ and $B$ after $20$ years. Raising a matrix to a power has modeled a chain of events feeding into itself, repeating the same behavior over and over.

!!! note
    The columns of $M^{20}$ are all the same (to the accuracy shown)! More amazingly, for all Markov matrices $M$ with positive entries as well as certain Markov matrices (such as $M$ above) with some entries equal to $0$, the powers $M^r$ for all sufficiently big $r$ have all columns essentially equal to a single vector $\mathbf{v}_M$ that is independent of $r$.

    In the above bird example, for all $r \geq 20$ it turns out that $M^r \approx M^{20}$ to high accuracy. Thus, the populations on each island stabilize after not so many years have passed: individual birds keep moving around, but eventually the total population on each island remains the same in all subsequent years.

    The equality of all columns has a remarkable consequence: the proportion of the total population that winds up on each island in the final "steady state" is always the corresponding entry in the common column vector (e.g., $42.9\%$ on island $C$), so it is independent of the way the $30000$ birds were initially distributed across the three islands!

    To see this in action, suppose that instead of spreading the birds evenly we had put **all** $30000$ of them on island $A$, so that $\mathbf{n}_0 = (30000,\ 0,\ 0)$. Then

    $$
    M^{20}\begin{pmatrix} 30000 \\[0.35em] 0 \\[0.35em] 0 \end{pmatrix}
    \approx
    \begin{pmatrix}
    0.286 & 0.286 & 0.286 \\[0.35em]
    0.286 & 0.286 & 0.286 \\[0.35em]
    0.429 & 0.429 & 0.429
    \end{pmatrix}
    \begin{pmatrix} 30000 \\[0.35em] 0 \\[0.35em] 0 \end{pmatrix}
    \approx
    \begin{pmatrix} 8570 \\[0.35em] 8570 \\[0.35em] 12860 \end{pmatrix},
    $$

    the very same steady state as before. The reason is visible once we recall that a matrix-vector product is a linear combination of the columns, with the entries of the vector as coefficients:

    $$M^{20}\mathbf{n}_0 = A_0 \cdot (\text{column } 1) + B_0 \cdot (\text{column } 2) + C_0 \cdot (\text{column } 3) \approx (A_0 + B_0 + C_0)\,\mathbf{v}_M ,$$

    since all three columns are the same vector $\mathbf{v}_M$. Only the **total** $A_0 + B_0 + C_0 = 30000$ survives; how that total was split among the islands is forgotten entirely. The explanation rests on a result that underlies both Google’s PageRank algorithm and the technique called “Markov Chain Monte Carlo” that is used everywhere for efficient sampling of probabilistic scenarios.

**The lesson from this is: matrix powers encode useful (and sometimes rather surprising!) information about chains of events.**

!!! note
    Consider $M = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. In this case we have $M^2 = I_2$, so

    $$
    M^3 = MM^2 = MI_2 = M, \qquad
    M^4 = MM^3 = MM = M^2 = I_2, \qquad
    M^5 = MM^4 = MI_2 = M,
    $$

    and so on: the powers of $M$ alternate between $M$ and $I_2$, with $M^{\text{odd}} = M$ and $M^{\text{even}} = I_2$. So in this case there is no stabilizing behavior for $M^r$ as $r$ grows!

## Gambler’s ruin and genetic evolution
