# Covariance and Correlation

Covariance and correlation are fundamental measures that describe the relationship between two random variables. They help us understand how variables change together and the strength and direction of their linear relationship.

## Covariance

The **covariance** between two random variables $X$ and $Y$ is defined as:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$

The covariance can also be expressed as:

$$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

This form is often more convenient for calculations.

### Interpretation and Properties

- **Positive Covariance**: When $X$ tends to be above its mean, $Y$ also tends to be above its mean OR when $X$ tends to be below its mean, $Y$ also tends to be below its mean

- **Negative Covariance**: When $X$ tends to be above its mean, $Y$ tends to be below its mean OR vice versa

- **Variance as a special case**: $\text{Cov}(X, X) = \text{Var}(X)$

- **Symmetry**: $\text{Cov}(X, Y) = \text{Cov}(Y, X)$

- **Independence**: If $X$ and $Y$ are independent, this means $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$. This results in $\text{Cov}(X, Y) = 0$, as expected

- $\text{Cov}(X, c) = 0$ where $c$ is a constant

- $\text{Cov}(cX, Y) = c \cdot \text{Cov}(X, Y)$ where $c$ is a constant

- $\text{Cov}(X, Y + Z) = \text{Cov}(X, Y) + \text{Cov}(X, Z)$

- $\text{Cov}(X + Y, Z + W) = \text{Cov}(X, Z) + \text{Cov}(X, W) + \text{Cov}(Y, Z) + \text{Cov}(Y, W)$

- For constants $a_1, a_2, \ldots, a_n$ and $b_1, b_2, \ldots, b_m$:

$$\text{Cov}\left(\sum_{i=1}^n a_i X_i, \sum_{j=1}^m b_j Y_j\right) = \sum_{i=1}^n \sum_{j=1}^m a_i b_j \text{Cov}(X_i, Y_j)$$

- $\text{Var}(X_1 + X_2) = \text{Var}(X_1) + \text{Var}(X_2) + 2\text{Cov}(X_1, X_2)$:

Using the fact that $\text{Var}(X) = \text{Cov}(X, X)$:

$$\begin{align}
\text{Var}(X_1 + X_2) &= \text{Cov}(X_1 + X_2, X_1 + X_2) \\
&= \text{Cov}(X_1, X_1) + \text{Cov}(X_1, X_2) + \text{Cov}(X_2, X_1) + \text{Cov}(X_2, X_2) \\
&= \text{Var}(X_1) + \text{Cov}(X_1, X_2) + \text{Cov}(X_2, X_1) + \text{Var}(X_2) \\
&= \text{Var}(X_1) + \text{Var}(X_2) + 2\text{Cov}(X_1, X_2)
\end{align}$$

$\text{Var}(X_1 + X_2) = \text{Var}(X_1) + \text{Var}(X_2)$ is true when $\text{Cov}(X_1, X_2) = 0$ (which means when $X_1$ and $X_2$ are uncorrelated). In particular, if $X_1$ and $X_2$ are independent, then $\text{Var}(X_1 + X_2) = \text{Var}(X_1) + \text{Var}(X_2)$. Note that $X_1$ and $X_2$ can be dependent and $\text{Cov}(X_1, X_2) = 0$. But when $X_1$ and $X_2$ are independent, it is always true that $\text{Cov}(X_1, X_2) = 0$.

**General Case - Variance of sum of n Random Variables**:

For $n$ random variables $X_1, X_2, \ldots, X_n$:

$$\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{1 \leq i < j \leq n} \text{Cov}(X_i, X_j)$$

**Note:** Zero Covariance does **not** imply independence

Let $Z \sim N(0, 1)$ be a standard normal random variable, and define:

- $X = Z$

- $Y = Z^2$

**Covariance calculation**:

$$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[Z \cdot Z^2] - \mathbb{E}[Z]\mathbb{E}[Z^2]$$

Since $Z \sim N(0, 1)$:

- $\mathbb{E}[Z] = 0$

- $\mathbb{E}[Z^2] = \text{Var}(Z) + (\mathbb{E}[Z])^2 = 1 + 0 = 1$

- $\mathbb{E}[Z^3] = 0$ (odd moments of standard normal are zero)

Therefore: $\text{Cov}(X, Y) = 0 - 0 \cdot 1 = 0$

**Dependence**: $X$ and $Y$ are clearly dependent because knowing $X = Z$ completely determines $Y = Z^2$. For example, if $X = 2$, then $Y$ must be 4.

## Correlation

The **correlation coefficient** (or Pearson correlation) between two random variables $X$ and $Y$ is defined as:

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X) \text{Var}(Y)}} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

where $\sigma_X$ and $\sigma_Y$ are the standard deviations of $X$ and $Y$ respectively.

### Interpretation and Properties

- **Range**: $-1 \leq \rho_{X,Y} \leq 1$

- **Scale Invariance**: $\rho_{aX + b, cY + d} = \rho_{X, Y}$ for $a, c > 0$

- $\rho_{X,Y} = 1$ when $Y = aX + b$ with $a > 0$

- $\rho_{X,Y} = -1$ when $Y = aX + b$ with $a < 0$

- **$\rho = 1$**: Perfect positive linear relationship

- **$\rho = -1$**: Perfect negative linear relationship  

- **$\rho = 0$**: No linear relationship

- **$|\rho| > 0.7$**: Strong linear relationship

- **$0.3 < |\rho| < 0.7$**: Moderate linear relationship

- **$|\rho| < 0.3$**: Weak linear relationship

- The correlation coefficient is essentially a normalized version of the covariance:

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

### Connection to Linear Algebra

The correlation coefficient has a beautiful geometric interpretation in terms of the angle between vectors in $\mathbb{R}^n$:

**For centered data**: If we have $n$ observations $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$, and we center the data by subtracting means:

- $\mathbf{x} = (x_1 - \bar{x}, x_2 - \bar{x}, \ldots, x_n - \bar{x})$

- $\mathbf{y} = (y_1 - \bar{y}, y_2 - \bar{y}, \ldots, y_n - \bar{y})$

Then the correlation coefficient equals the cosine of the angle between these centered vectors:

$$\rho_{X,Y} = \cos \theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

**Key insights**:

- $\rho = 1$ corresponds to $\theta = 0°$ (vectors point in same direction)

- $\rho = -1$ corresponds to $\theta = 180°$ (vectors point in opposite directions)  

- $\rho = 0$ corresponds to $\theta = 90°$ (vectors are orthogonal)

- The correlation measures how "aligned" the centered data vectors are in $\mathbb{R}^n$