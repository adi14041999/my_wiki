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

This challenge motivates the development of **annealed Langevin dynamics** and other advanced sampling techniques that can better handle complex, multi-modal distributions.