# Transformations of Random Variables

Let $X$ be a continuous random variable with PDF $f_X$, and let $Y = g(X)$, where $g$ is differentiable and strictly increasing (or strictly decreasing). Then the PDF of $Y$ is given by:

$$f_Y(y) = f_X(x) \left|\frac{dx}{dy}\right|$$

where $x = g^{-1}(y)$.

**Proof.** Let $g$ be strictly increasing. The CDF of $Y$ is:

$$F_Y(y) = P(Y \leq y) = P(g(X) \leq y) = P(X \leq g^{-1}(y)) = F_X(g^{-1}(y)) = F_X(x)$$

So by the chain rule, the PDF of $Y$ is:

$$f_Y(y) = f_X(x) \frac{dx}{dy}$$

The proof for $g$ strictly decreasing is analogous. In that case the PDF ends up as $-f_X(x) \frac{dx}{dy}$, which is nonnegative since $\frac{dx}{dy} < 0$ if $g$ is strictly decreasing. Using $\left|\frac{dx}{dy}\right|$, as in the statement of the theorem, covers both cases.

**Key points:**

1. **Differentiability**: The function $g$ must be differentiable
2. **Monotonicity**: $g$ must be strictly increasing or strictly decreasing
3. **Absolute value**: The absolute value ensures the PDF is non-negative

**Note:** When applying the change of variables formula, we can choose whether to compute $\frac{dx}{dy}$, or compute $\frac{dy}{dx}$ and take the reciprocal. By the chain rule, these give the same result, so we can do whichever is easier.

**Example:** Let $X \sim N(0, 1)$, $Y = e^X$. We name the distribution of $Y$ the Log-Normal. Now we can use the change of variables formula to find the PDF of $Y$, since $g(x) = e^x$ is strictly increasing.

Let $y = e^x$, so $x = \log y$ and $\frac{dy}{dx} = e^x$. Then:

$$f_Y(y) = f_X(x) \left|\frac{dx}{dy}\right| = \phi(x) \frac{1}{e^x} = \phi(\log y) \frac{1}{y}, \quad y > 0$$

Note that after applying the change of variables formula, we write everything on the right-hand side in terms of $y$, and we specify the support of the distribution. To determine the support, we just observe that as $x$ ranges from $-\infty$ to $\infty$, $e^x$ ranges from $0$ to $\infty$.

We can get the same result by working from the definition of the CDF, translating the event $Y \leq y$ into an equivalent event involving $X$. For $y > 0$:

$$F_Y(y) = P(Y \leq y) = P(e^X \leq y) = P(X \leq \log y) = \Phi(\log y)$$

So the PDF is again:

$$f_Y(y) = \frac{d}{dy} \Phi(\log y) = \phi(\log y) \frac{1}{y}, \quad y > 0$$

## Change of Variables in multiple Dimensions

The change of variables formula generalizes to $n$ dimensions, where it tells us how to use the joint PDF of a random vector $\mathbf{X}$ to get the joint PDF of the transformed random vector $\mathbf{Y} = g(\mathbf{X})$. The formula is analogous to the one-dimensional version, but it involves a multivariate generalization of the derivative called a Jacobian matrix.

Let $\mathbf{X} = (X_1, \ldots, X_n)$ be a continuous random vector with joint PDF $f_{\mathbf{X}}$. Let $g : A_0 \to B_0$ be an invertible function, where $A_0$ and $B_0$ are open subsets of $\mathbb{R}^n$, $A_0$ contains the support of $\mathbf{X}$, and $B_0$ is the range of $g$.

**Note:** A set $C \subseteq \mathbb{R}^n$ is open if for each $\mathbf{x} \in C$, there exists $\epsilon > 0$ such that all points with distance less than $\epsilon$ from $\mathbf{x}$ are contained in $C$. Sometimes we take $A_0 = B_0 = \mathbb{R}^n$, but often we would like more flexibility for the domain and range of $g$. For example, if $n = 2$, and $X_1$ and $X_2$ have support $(0,\infty)$, we may want to work with the open set $A_0 = (0,\infty) \times (0,\infty)$ rather than all of $\mathbb{R}^2$. When we say "$A_0$ contains the support," we mean that $A_0$ must be a superset of the support of $\mathbf{X}$. In the example above, if the support is $(0,\infty) \times (0,\infty)$, then $A_0 = (0,\infty) \times (0,\infty)$ does contain the support (in fact, it equals the support). We could also take $A_0 = \mathbb{R}^2$, which would definitely contain the support, but choosing $A_0$ to be the minimal open set containing the support is often more convenient and natural.

Let $\mathbf{Y} = g(\mathbf{X})$, and mirror this by letting $\mathbf{y} = g(\mathbf{x})$. Since $g$ is invertible, we also have $\mathbf{X} = g^{-1}(\mathbf{Y})$ and $\mathbf{x} = g^{-1}(\mathbf{y})$.

Suppose that all the partial derivatives $\frac{\partial x_i}{\partial y_j}$ exist and are continuous, so we can form the Jacobian matrix:

$$\frac{\partial \mathbf{x}}{\partial \mathbf{y}} = \begin{pmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} & \cdots & \frac{\partial x_1}{\partial y_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial x_n}{\partial y_1} & \frac{\partial x_n}{\partial y_2} & \cdots & \frac{\partial x_n}{\partial y_n}
\end{pmatrix}$$

Also assume that the determinant of this Jacobian matrix is never 0. Then the joint PDF of $\mathbf{Y}$ is:

$$f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(g^{-1}(\mathbf{y})) \left|\det\left(\frac{\partial \mathbf{x}}{\partial \mathbf{y}}\right)\right|$$

for $\mathbf{y} \in B_0$, and 0 otherwise.

That is, to convert $f_{\mathbf{X}}(\mathbf{x})$ to $f_{\mathbf{Y}}(\mathbf{y})$ we express the $\mathbf{x}$ in $f_{\mathbf{X}}(\mathbf{x})$ in terms of $\mathbf{y}$ and then multiply by the absolute value of the determinant of the Jacobian $\frac{\partial \mathbf{x}}{\partial \mathbf{y}}$.

As in the 1D case, $\left|\det\left(\frac{\partial \mathbf{x}}{\partial \mathbf{y}}\right)\right| = \left|\det\left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)\right|^{-1}$, so we can compute whichever of the two Jacobians is easier, and then at the end express the joint PDF of $\mathbf{Y}$ as a function of $\mathbf{y}$.

The idea is to apply the change of variables formula from multivariable calculus and the fact that if $A$ is a region in $A_0$ and $B = \{g(\mathbf{x}) : \mathbf{x} \in A\}$ is the corresponding region in $B_0$, then $\mathbf{X} \in A$ is equivalent to $\mathbf{Y} \in B$—they are the same event. So $P(\mathbf{X} \in A) = P(\mathbf{Y} \in B)$, which shows that:

$$\int_A f_{\mathbf{X}}(\mathbf{x}) d\mathbf{x} = \int_B f_{\mathbf{Y}}(\mathbf{y}) d\mathbf{y}$$