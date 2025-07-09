# Energy-Based Models

## Parameterizing Probability Distributions

Probability distributions $p(x)$ are a key building block in generative modeling. Building a neural network that ensures $p(x) \geq 0$ is not hard. However, the real challenge lies in ensuring that the distribution satisfies the normalization constraint: for discrete variables, the sum over all possible values of $x$ must equal 1, while for continuous variables, the integral over the entire domain must equal 1.

**Problem:** $g_\theta(x) \geq 0$ is easy. But $\sum_x g_\theta(x) = Z(\theta) \neq 1$ in general, so $g_\theta(x)$ is not a valid probability mass function. For continuous variables, $\int g_\theta(x) dx = Z(\theta) \neq 1$ in general, so $g_\theta(x)$ is not a valid probability density function.

**Solution:**

$$p_\theta(x) = \frac{1}{Z(\theta)} g_\theta(x) = \frac{1}{\int g_\theta(x) dx} g_\theta(x) = \frac{1}{\text{Volume}(g_\theta)} g_\theta(x)$$

Then by definition,

$$\int p_\theta(x) dx = \int \frac{1}{Z(\theta)} g_\theta(x) dx = \frac{Z(\theta)}{Z(\theta)} = 1$$

Here, $g_\theta(x)$ is the output of the neural network with parameters $\theta$ at input $x$. The **volume** of $g_\theta$, denoted as $\text{Volume}(g_\theta)$, is defined as the integral of $g_\theta(x)$ over the entire domain: $\text{Volume}(g_\theta) = \int g_\theta(x) dx = Z(\theta)$. It is a normalizing constant (w.r.t. $x$) but changes for different $\theta$. For example, we choose $g_\theta(x)$ so that we know the volume analytically as a function of $\theta$.

The **partition function** $Z(\theta)$ is the normalization constant that ensures a probability distribution integrates (or sums) to 1. It's called a "partition function" because it partitions the unnormalized function $g_\theta(x)$ into a proper probability distribution.

**Example:** $g_{(\mu, \sigma)}(x) = e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

$$\text{Volume}(g_{(\mu, \sigma)}) = \int_{-\infty}^{\infty} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx = \sqrt{2\pi\sigma^2}$$

Therefore, the normalized probability density function is:

$$p_{(\mu, \sigma)}(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

This is the standard normal (Gaussian) distribution with mean $\mu$ and variance $\sigma^2$.
Functional forms $g_\theta(x)$ need to allow analytical integration. Despite being restrictive, they are very useful as building blocks for more complex distributions.

**Note:** What we've been doing with autoregressive models, flow models, and VAEs are essentially tricks for composing simple functions that are normalized to build more complex probabilistic models that are by construction guaranteed to be normalized. These approaches avoid the intractability of computing the partition function for complex distributions by designing architectures where normalization is preserved through the composition of simple, analytically tractable components.
