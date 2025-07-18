# Score Based Generative Modeling
## Langevin Dynamics Sampling

Langevin dynamics is a powerful MCMC method that uses gradient information to efficiently sample from complex probability distributions. For score-based models, it provides a natural way to generate samples by following the learned score field.

**Mathematical Foundation**

Langevin dynamics is based on the **Langevin equation**, a stochastic differential equation that describes the motion of particles in a potential field:

$$dx_t = \nabla_x \log p(x_t) dt + \sqrt{2} dW_t$$

where:

- $x_t$ is the particle position at time $t$

- $\nabla_x \log p(x_t)$ is the score function (gradient of log probability)

- $W_t$ is a Wiener process (Brownian motion)

- The first term is the **drift term** (gradient guidance)

- The second term is the **diffusion term** (random exploration)

**Discretized Langevin Dynamics**

For practical implementation, we discretize the continuous-time equation:

$$x_{t+1} = x_t + \epsilon \cdot \nabla_x \log p(x_t) + \sqrt{2\epsilon} \cdot \eta_t$$

where:

- $\epsilon$ is the step size (time discretization)

- $\eta_t \sim \mathcal{N}(0, I)$ is Gaussian noise

- $t$ indexes the discrete time steps

**Score-Based Langevin Sampling**

For our trained score function $s_\theta(x) \approx \nabla_x \log p_{data}(x)$, the sampling algorithm becomes:

**Algorithm: Score-Based Langevin Sampling**

1. **Initialize**: $x_0 \sim \mathcal{N}(0, I)$ (random noise)

2. **Iterate**: For $t = 0, 1, 2, \ldots, T-1$:

   - Compute score: $s_t = s_\theta(x_t)$

   - Add gradient step: $x_{t+1} = x_t + \frac{\epsilon}{2} \cdot s_t + \sqrt{2\epsilon} \cdot \eta_t$

   - Where $\eta_t \sim \mathcal{N}(0, I)$

3. **Return**: $x_T$ as the generated sample

**Intuition Behind Langevin Dynamics**

**The Drift Term ($\frac{\epsilon}{2} \cdot s_\theta(x_t)$):**

- Pushes the sample toward high-probability regions

- The score function points "uphill" in the probability landscape

- Larger step sizes $\epsilon$ lead to more aggressive movement

- The factor of $\frac{1}{2}$ comes from proper discretization of the continuous Langevin equation

**The Diffusion Term ($\sqrt{2\epsilon} \cdot \eta_t$):**

- Adds random exploration to avoid getting stuck in local modes

- Balances the deterministic gradient guidance

- Ensures the chain can escape local optima and explore the full distribution

**Balance Between Drift and Diffusion:**

- **Small $\epsilon$**: More exploration, slower convergence, better mixing

- **Large $\epsilon$**: Faster convergence, but may miss modes or become unstable

- **Optimal $\epsilon$**: Depends on the data distribution and model architecture

**Convergence Guarantees**

Under mild conditions on the target distribution and score function, Langevin dynamics provides strong theoretical guarantees:

**Asymptotic Convergence:**

If $\epsilon \to 0$ and $T \to \infty$, we are guaranteed that $x_T \sim p_{data}(x)$.

**Mathematical Interpretation:**

- **$\epsilon \to 0$**: The discretization becomes arbitrarily fine, approaching the continuous Langevin equation
- **$T \to \infty$**: The Markov chain runs for an infinite number of steps, allowing it to reach the stationary distribution
- **$x_T \sim p_{data}(x)$**: The final sample is distributed according to the target data distribution

**Challenge in Low Density Regions:**

One significant limitation of Langevin dynamics is its poor performance in **low density regions** of the target distribution:

- **Weak Score Signals**: In regions where $p_{data}(x) \approx 0$, the score function $\nabla_x \log p_{data}(x)$ becomes very small or noisy
- **Mode Collapse Risk**: The algorithm may fail to explore all modes (mode is a region where the probability density is high, i.e., data points are concentrated) of a multi-modal distribution
- **Slow convergence:** Langevin Dynamics converges very slowly. Might not even converge if we have zero probability somewhere.

This challenge motivates the development of **annealed Langevin dynamics** and other advanced sampling techniques that can better handle complex, multi-modal distributions.

## Annealed Langevin Dynamics

**Mathematical Formulation**

We define a sequence of **annealed distributions** indexed by noise level $\sigma_t$:

$$p_t(x) = \int p_{data}(y) \mathcal{N}(x; y, \sigma_t^2 I) dy$$

where each $p_t(x)$ is a smoothed version of the original data distribution.

This equation is derived from the **convolution** of the data distribution with the noise distribution. Here's the step-by-step reasoning:

If $Y \sim p_{data}(y)$ and $\epsilon \sim \mathcal{N}(0, \sigma_i^2 I)$, then the noisy sample is $X = Y + \epsilon$.

The joint distribution of $(Y, X)$ is:

$$p(y, x) = p_{data}(y) \cdot \mathcal{N}(x; y, \sigma_i^2 I)$$

The joint distribution is derived using the **chain rule of probability**:

$$p(y, x) = p(y) \cdot p(x | y)$$

where:

- $p(y) = p_{data}(y)$ is the marginal distribution of the clean data

- $p(x | y)$ is the conditional distribution of the noisy sample given the clean data

Since $X = Y + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma_i^2 I)$, the conditional distribution is:

$$p(x | y) = \mathcal{N}(x; y, \sigma_i^2 I)$$

This is because adding a constant ($y$) to a Gaussian random variable shifts the mean but preserves the variance. Therefore:

$$p(y, x) = p_{data}(y) \cdot \mathcal{N}(x; y, \sigma_i^2 I)$$

To get the distribution of $X$ alone, we marginalize over $Y$:

$$p_{\sigma_i}(x) = \int p(y, x) dy = \int p_{data}(y) \mathcal{N}(x; y, \sigma_i^2 I) dy$$

We're using the **law of total probability** (also called marginalization). When we have a joint distribution $p(y, x)$, to find the marginal distribution of $x$ alone, we integrate out the other variable:

$$p_{\sigma_i}(x) = \int p(y, x) dy$$

This is because:

- The joint distribution $p(y, x)$ gives us the probability of both $y$ AND $x$ occurring together

- To find the probability of just $x$ (regardless of what $y$ is), we sum over all possible values of $y$

- In continuous probability, "summing" becomes integration

**Intuition**: We're asking "What's the probability of observing a noisy sample $x$?" The answer is the sum of probabilities over all possible clean samples $y$ that could have generated this noisy sample.

**Final Form**: The noise-perturbed distribution is:

$$p_{\sigma_i}(x) = \int p_{data}(y) \mathcal{N}(x; y, \sigma_i^2 I) dy$$

We use multiple scales of noise perturbations simultaneously. Suppose we always perturb the data with isotropic Gaussian noise, and let there be a total of $L$ increasing standard deviations $\sigma_1 < \sigma_2 < \ldots < \sigma_L$. We first perturb the data distribution $p_{data}(y)$ with each of the Gaussian noise $\mathcal{N}(0, \sigma_i^2 I)$ to obtain a noise-perturbed distribution (the final form we derived above):

$$p_{\sigma_i}(x) = \int p_{data}(y) \mathcal{N}(x; y, \sigma_i^2 I) dy$$

Note that we can easily draw samples from $p_{\sigma_i}(x)$ by sampling $y \sim p_{data}(y)$ and computing $x = y + \sigma_i \epsilon$, with $\epsilon \sim \mathcal{N}(0, I)$.

We estimate the score function of each noise-perturbed distribution, $\nabla_x \log p_{\sigma_i}(x)$, by training a Denoising Score Matching Model (when parameterized with a neural network) with score matching, such that $s_\theta(x, \sigma_i) \approx \nabla_x \log p_{\sigma_i}(x)$ for all $i$. The training objective for $s_\theta$ is a weighted sum of Fisher divergences for all noise scales. In particular, we use the objective below:

$$\mathcal{L}(\theta) = \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{p_{\sigma_i}(x)} \left[ \| s_\theta(x, \sigma_i) - \nabla_x \log p_{\sigma_i}(x) \|_2^2 \right]$$

where $\lambda(\sigma_i)$ is a positive weighting function, often chosen to be $\lambda(\sigma_i) = \sigma_i^2$. The objective $\mathcal{L}(\theta)$ can be optimized with score matching, exactly as in optimizing the naive score-based model.

**Denoising Score Matching Format:**

We can rewrite the objective in Denoising Score Matching model format:

$$\mathcal{L}(\theta) = \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{y \sim p_{data}(y), x \sim \mathcal{N}(x; y, \sigma_i^2 I)} \left[ \left\| s_\theta(x, \sigma_i) - \frac{y - x}{\sigma_i^2} \right\|_2^2 \right]$$

**Note**: The noise scales $\sigma_1, \sigma_2, \ldots, \sigma_L$ are typically chosen to be in a **geometric progression**, meaning $\sigma_{i+1} = \alpha \cdot \sigma_i$ for some constant $\alpha < 1$. This ensures that the noise levels decrease exponentially, providing a smooth annealing schedule from high noise to low noise.

Perturbing an image with multiple scales of Gaussian noise:
![Perturbing an image with multiple scales of Gaussian noise](duoduo.jpg)

After training our score-based model $s_\theta(x, \sigma_i)$, we can produce samples from it by running Langevin dynamics for $\sigma_L, \sigma_{L-1}, \ldots, \sigma_1$ in sequence. This method is called Annealed Langevin dynamics, since the noise scale decreases (anneals) gradually over time.

We can start from unstructured noise, modify images according to the scores, and generate nice samples:
![Annealed Langevin dynamics](celeba_large.gif)



