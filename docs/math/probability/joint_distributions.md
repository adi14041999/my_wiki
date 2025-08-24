# Joint Distributions

A **joint distribution** describes the probability distribution of two or more random variables simultaneously. It captures not only the individual behavior of each random variable but also how they relate to each other.

For two discrete random variables $X$ and $Y$, the **joint probability mass function** (joint PMF) is defined as:

$$p_{X,Y}(x,y) = P(X = x, Y = y)$$

For continuous random variables, the **joint probability density function** (joint PDF) satisfies:

$$P((X,Y) \in A) = \iint_A f_{X,Y}(x,y) \, dx \, dy$$

The **joint cumulative distribution function** (joint CDF) is defined as:

$$F_{X,Y}(x,y) = P(X \leq x, Y \leq y)$$

For discrete random variables, this becomes:

$$F_{X,Y}(x,y) = \sum_{i \leq x} \sum_{j \leq y} p_{X,Y}(i,j)$$

**Example:** Let's consider two Bernoulli random variables $X$ and $Y$ with parameters $p$ and $q$ respectively:

- $X \sim \text{Bernoulli}(p)$ where $P(X = 1) = p$ and $P(X = 0) = 1-p$

- $Y \sim \text{Bernoulli}(q)$ where $P(Y = 1) = q$ and $P(Y = 0) = 1-q$

The joint PMF $p_{X,Y}(x,y)$ gives us the probability of each possible combination:

| $X \backslash Y$ | $0$ | $1$ |
|------------------|-----|-----|
| $0$ | $p_{X,Y}(0,0)$ | $p_{X,Y}(0,1)$ |
| $1$ | $p_{X,Y}(1,0)$ | $p_{X,Y}(1,1)$ |

**If $X$ and $Y$ are independent**, then:

$$p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y)$$

This means:

- $p_{X,Y}(0,0) = (1-p)(1-q)$

- $p_{X,Y}(0,1) = (1-p)q$

- $p_{X,Y}(1,0) = p(1-q)$

- $p_{X,Y}(1,1) = pq$

**If $X$ and $Y$ are dependent**, the joint PMF cannot be factored this way, and we need additional information to specify the relationship between them.

From the joint distribution, we can recover the individual (marginal) distributions:

$$p_X(x) = \sum_y p_{X,Y}(x,y)$$

$$p_Y(y) = \sum_x p_{X,Y}(x,y)$$

For our Bernoulli example:

- $p_X(0) = p_{X,Y}(0,0) + p_{X,Y}(0,1) = 1-p$

- $p_X(1) = p_{X,Y}(1,0) + p_{X,Y}(1,1) = p$

- $p_Y(0) = p_{X,Y}(0,0) + p_{X,Y}(1,0) = 1-q$

- $p_Y(1) = p_{X,Y}(0,1) + p_{X,Y}(1,1) = q$

For continuous random variables, the marginal PDFs are obtained by integrating the joint PDF:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy$$

$$f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx$$

**Example:** Consider two continuous random variables $X$ and $Y$ with joint PDF:

$$f_{X,Y}(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} + \frac{(y-\mu_Y)^2}{\sigma_Y^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y}\right]\right)$$

To find the marginal PDF of $X$, we integrate over $y$:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy$$

After some algebra, this gives us:

$$f_X(x) = \frac{1}{\sqrt{2\pi}\sigma_X} \exp\left(-\frac{(x-\mu_X)^2}{2\sigma_X^2}\right)$$

This shows that the marginal distribution of $X$ is $N(\mu_X, \sigma_X^2)$.

**Example:** Consider a uniform distribution over the disc $x^2 + y^2 \leq c$. The joint PDF is:

$$f_{X,Y}(x,y) = \begin{cases}
\frac{1}{\pi c} & \text{if } x^2 + y^2 \leq c \\
0 & \text{otherwise}
\end{cases}$$

To find the marginal PDF of $X$, we integrate over $y$:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy$$

For a given $x$ with $|x| \leq \sqrt{c}$, the range of $y$ is from $-\sqrt{c-x^2}$ to $\sqrt{c-x^2}$:

$$f_X(x) = \int_{-\sqrt{c-x^2}}^{\sqrt{c-x^2}} \frac{1}{\pi c} \, dy = \frac{2\sqrt{c-x^2}}{\pi c}$$

For $|x| > \sqrt{c}$, $f_X(x) = 0$.

Similarly, the marginal PDF of $Y$ is:

$$f_Y(y) = \frac{2\sqrt{c-y^2}}{\pi c}$$

This shows that the marginal distributions are not uniform - they follow a semi-circular distribution, even though the joint distribution is uniform over the disc.

To show that $X$ and $Y$ are dependent, we need to verify that:

$$f_{X,Y}(x,y) \neq f_X(x) \cdot f_Y(y)$$

Let's check this for a point inside the disc, say $(x,y) = (0,0)$:

- Joint PDF: $f_{X,Y}(0,0) = \frac{1}{\pi c}$

- Marginal PDFs: $f_X(0) = \frac{2\sqrt{c}}{\pi c}$ and $f_Y(0) = \frac{2\sqrt{c}}{\pi c}$

- Product of marginals: $f_X(0) \cdot f_Y(0) = \frac{4c}{\pi^2 c^2} = \frac{4}{\pi^2 c}$

Since $\frac{1}{\pi c} \neq \frac{4}{\pi^2 c}$, we have:

$$f_{X,Y}(0,0) \neq f_X(0) \cdot f_Y(0)$$

This proves that $X$ and $Y$ are dependent random variables. The dependence arises from the geometric constraint $x^2 + y^2 \leq c$ - knowing the value of $X$ constrains the possible values of $Y$ and vice versa.