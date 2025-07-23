# Score Based Diffusion Models

## Quick Recap: Score Based Models

From our exploration of score-based generative modeling, we learned several key concepts:

**Score Function**: The gradient of the log probability density, $\nabla_x \log p(x)$, which points "uphill" in the probability landscape toward high-density regions.

**Score Matching**: A training objective that learns the score function by minimizing the Fisher divergence between the learned and true score functions.

**Score Matching Objective**: The original score matching objective is:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{data}(x)} \left[ \frac{1}{2} \| s_\theta(x) \|_2^2 + \text{tr}(\nabla_x s_\theta(x)) \right]$$

where $\text{tr}(\nabla_x s_\theta(x))$ is the trace of the Jacobian of the score function, which is computationally expensive to evaluate.

**Denoising Score Matching (DSM)**: A practical variant that trains the score function to predict the direction from noisy to clean data, avoiding the need to compute the true score function.

**DSM Objective**: The denoising score matching objective is:

$$\mathcal{L}(\theta) = \mathbb{E}_{y \sim p_{data}(y)} \mathbb{E}_{x \sim \mathcal{N}(x; y, \sigma^2 I)} \left[ \frac{1}{2} \left\| s_\theta(x) - \frac{y - x}{\sigma^2} \right\|_2^2 \right]$$

where $s_\theta(x)$ learns to predict the score function of the noise-perturbed distribution, and $\frac{y - x}{\sigma^2}$ is the target score function that points from noisy sample $x$ toward clean data $y$.

**Langevin Dynamics**: A continuous-time stochastic process that uses the score function to guide sampling:

$$dx_t = \nabla_x \log p(x_t) dt + \sqrt{2} dW_t$$

**Discretized Form**: For practical implementation:

$$x_{t+1} = x_t + \frac{\epsilon}{2} \cdot s_\theta(x_t) + \sqrt{2\epsilon} \cdot \eta_t$$

**Mode Collapse**: Standard Langevin dynamics struggles with multi-modal distributions and low-density regions.

**Annealed Langevin Dynamics**: Addresses this by using multiple noise scales $\sigma_1 < \sigma_2 < \ldots < \sigma_L$, creating a sequence of increasingly noisy distributions that are easier to sample from.

**Stochastic Differential Equations (SDEs)**: General framework for continuous-time stochastic processes:

$$dx = f(x, t)dt + g(t)dw$$

**Reverse SDE**: Any SDE has a corresponding reverse process for sampling:

$$dx = [f(x, t) - g^2(t)\nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

**Time-Dependent Score Models**: Neural networks that learn $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ for continuous-time processes.

**Key insights:**

1. **Score functions act as denoisers**: They point from noisy to clean data
2. **Multiple noise scales help**: Annealing from high to low noise improves sampling
3. **Continuous-time generalizes discrete**: SDEs provide a unified framework
4. **Reverse processes enable generation**: The reverse SDE naturally incorporates the score function for sampling

## Diffusion Models as Score Based Models & Hierarchical VAEs

**Iterative Denoising perspective**: In annealed Langevin dynamics with multiple noise scales, the sampling process can be viewed as **iterative denoising**. Starting from high noise levels and gradually reducing noise, each step uses the score function to denoise the sample, progressively refining it from a noisy state toward the clean data distribution.

**Training perspective**: The inverse process involves **iteratively adding Gaussian noise** to clean data during training. By corrupting data with increasing levels of noise, the model learns to predict the score function at each noise level, enabling it to reverse the corruption process during sampling.

![Iterative denoising](iter_denoise.png)

**VAE Perspective**: This entire framework can be viewed as a **VAE** where:

- **Encoder process**: The forward process that converts clean data to noise through iterative corruption

- **Decoder process**: The reverse process that generates samples by iteratively denoising from noise

**Noise Perturbation process**: Each $x_t$ represents a noise-perturbed density that is obtained by adding Gaussian noise to $x_{t-1}$. This creates a Markov chain where each step adds a small amount of noise to the previous state.

We can write the forward process as a conditional distribution:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

where $\beta_t$ is the noise schedule that determines how much noise is added at each step.

The joint distribution of the entire forward process is:

$$q(x_1, x_2, \ldots, x_T | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

This factorization follows from the **chain rule of probability** and the **Markov property** of the forward process:

**Chain Rule**: For any joint distribution, we can write:

$$q(x_1, x_2, \ldots, x_T | x_0) = q(x_1 | x_0) \cdot q(x_2 | x_0, x_1) \cdot q(x_3 | x_0, x_1, x_2) \cdots q(x_T | x_0, x_1, \ldots, x_{T-1})$$

**Markov Property**: In the forward process, each $x_t$ depends only on $x_{t-1}$, not on earlier states:

$$q(x_t | x_0, x_1, \ldots, x_{t-1}) = q(x_t | x_{t-1})$$

Substituting the Markov property into the chain rule:

$$q(x_1, x_2, \ldots, x_T | x_0) = q(x_1 | x_0) \cdot q(x_2 | x_1) \cdot q(x_3 | x_2) \cdots q(x_T | x_{T-1})$$

This can be written compactly as:

$$q(x_1, x_2, \ldots, x_T | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

This represents the probability of the entire noise corruption sequence, where each step depends only on the previous step (Markov property).

**Comparison with VAEs**: In a typical VAE, you would take $x_0$ and map it via a neural network to obtain some mean and standard deviation to parameterize the distribution of the latent variable. Here, we obtain the distribution of the latent variables through the **predefined noise corruption procedure** we defined above, rather than learning it with a neural network.

**Multistep transitions**: A key advantage of this process is that we can compute transitions between any two time steps efficiently. For example, we can directly compute $q(x_t | x_0)$ without going through all intermediate steps.

Starting from $x_0$, we can write:

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$

where $\alpha_t = 1 - \beta_t$ and $\epsilon_{t-1} \sim \mathcal{N}(0, I)$.

Recursively substituting:

$$x_t = \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2}) + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$

Continuing this recursion, we get:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ and $\epsilon \sim \mathcal{N}(0, I)$.

**Result**: The multistep transition is:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

This allows us to sample $x_t$ directly from $x_0$ in a single step, making training much more efficient.

**Diffusion analogy**: We can think of this as a **diffusion process**. This is like a diffuser where given an initial state, we keep adding noise at every step. This is analogous to **heat diffusion** in a space- just as heat spreads out and becomes more uniform over time, our data distribution becomes increasingly noisy and uniform Gaussian as we add more noise at each step.

The process gradually "diffuses" the structured information in the data into random noise, creating a smooth transition from the complex data distribution to a simple Gaussian noise distribution.

![Diffusion](diff.png)

The ideal sampling process would be:

1. Sample $x_T$ from $\pi(x_T)$. Start with pure noise from the prior distribution
2. Iteratively sample from the true denoising distribution $q(x_{t-1} | x_t)$.

This would generate samples by following the exact reverse of the forward diffusion process, gradually denoising from pure noise back to clean data.

The challenge however, is that we don't know the true denoising distributions $q(x_{t-1} | x_t)$. While the forward process $q(x_t | x_{t-1})$ is predefined and tractable, the reverse process is not.

However, we can learn an approximation $p_\theta(x_{t-1} | x_t)$ which is a Gaussian distribution with learned parameters:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

where $\mu_\theta(x_t, t)$ is a neural network that learns the mean of the denoising distribution, and $\sigma_t^2 I$ is the fixed variance schedule.

This is similar to a VAE decoder:

**VAE Decoder**:

$$p_\theta(x | z) = \mathcal{N}(x; \mu_\theta(z), \sigma_\theta^2(z) I)$$

**Diffusion reverse process**:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The diffusion decoder $p_\theta(x_{t-1} | x_t)$ is trying to learn to approximate the true denoising distributions $q(x_{t-1} | x_t)$.

The joint distribution of the learned reverse process is:

$$p_\theta(x_0, x_1, \ldots, x_{T-1} | x_T) = \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

Let's derive the joint distribution of the learned reverse process step by step.

In the general case of $n$ random variables $X_1, X_2, \ldots, X_n$, the values of an arbitrary subset of variables can be known and one can ask for the joint probability of all other variables. For example, if the values of $X_{k+1}, X_{k+2}, \ldots, X_n$ are known, the probability for $X_1, X_2, \ldots, X_k$ given these known values is:

$$P(X_1, X_2, \ldots, X_k|X_{k+1}, X_{k+2}, \ldots, X_n) = \frac{P(X_1, X_2, \ldots, X_n)}{P(X_{k+1}, X_{k+2}, \ldots, X_n)}$$

This is the fundamental definition of conditional probability for multiple random variables.

For any three events $A$, $B$, and $C$, the joint conditional probability is defined as:

$$P(A, B|C) = \frac{P(A, B, C)}{P(C)}$$

We can write the joint probability $P(A, B, C)$ using the chain rule:

$$P(A, B, C) = P(A|B, C) \cdot P(B, C)$$

Substituting this into our definition:

$$P(A, B|C) = \frac{P(A|B, C) \cdot P(B, C)}{P(C)}$$

We can write $P(B, C)$ as:

$$P(B, C) = P(B|C) \cdot P(C)$$

$$P(A, B|C) = \frac{P(A|B, C) \cdot P(B|C) \cdot P(C)}{P(C)}$$

The $P(C)$ terms cancel out:

$$P(A, B|C) = P(A|B, C) \cdot P(B|C)$$

The learned reverse process consists of a sequence of conditional distributions:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

where $\mu_\theta(x_t, t)$ is a neural network that learns the mean of the denoising distribution.

For the joint distribution $p_\theta(x_0, x_1, \ldots, x_{T-1} | x_T)$, we can apply the chain rule:

$$p_\theta(x_0, x_1, \ldots, x_{T-1} | x_T) = p_\theta(x_0 | x_1, \ldots, x_T) \cdot p_\theta(x_1 | x_2, \ldots, x_T) \cdots p_\theta(x_{T-1} | x_T)$$

In the reverse process, we assume that each $x_{t-1}$ depends only on $x_t$, not on future states. This is the **reverse Markov property**:

$$p_\theta(x_{t-1} | x_t, x_{t+1}, \ldots, x_T) = p_\theta(x_{t-1} | x_t)$$

Substituting the reverse Markov property into the chain rule:

$$p_\theta(x_0, x_1, \ldots, x_{T-1} | x_T) = p_\theta(x_0 | x_1) \cdot p_\theta(x_1 | x_2) \cdots p_\theta(x_{T-1} | x_T)$$

This can be written compactly as:

$$p_\theta(x_0, x_1, \ldots, x_{T-1} | x_T) = \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

To get the complete joint distribution, we need to include the prior distribution over $x_T$:

$$p_\theta(x_0, x_1, \ldots, x_T) = p(x_T) \cdot p_\theta(x_0, x_1, \ldots, x_{T-1} | x_T)$$

$$p_\theta(x_0, x_1, \ldots, x_T) = p(x_T) \cdot \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

A crucial aspect of the diffusion process is choosing the values of $\bar{\alpha}_t$ such that after many steps, we are left with pure noise. This ensures that the forward process converges to a simple, known distribution.

Common choices for the noise schedule include:

1. **Linear Schedule**: $\beta_t = \frac{t}{T} \cdot \beta_{\text{max}}$
2. **Cosine Schedule**: $\beta_t = \cos\left(\frac{t}{T} \cdot \frac{\pi}{2}\right)$
3. **Quadratic Schedule**: $\beta_t = \left(\frac{t}{T}\right)^2 \cdot \beta_{\text{max}}$

**Example**: For a linear schedule with $\beta_{\text{max}} = 0.02$ and $T = 1000$, we get $\beta_1 = 0.00002$, $\beta_{500} = 0.01$ and $\beta_{1000} = 0.02$.

Once we have trained the diffusion model and learned the reverse process $p_\theta(x_{t-1} | x_t)$, we can generate new samples by running the reverse process. Here's how sampling works. 

Sample $x_T$ from the prior distribution $x_T \sim \mathcal{N}(x_T; 0, I)$.

For $t = T, T-1, \ldots, 1$, sample from the learned reverse process $x_{t-1} \sim p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$.

After $T$ steps, we obtain $x_0$, which is our generated sample.

This entire diffusion framework can be viewed as a **Variational Autoencoder (VAE)** with a crucial difference: **the encoder is fixed and predefined, while only the decoder is learned**.

**Standard VAE Structure**:

- **Encoder**: $q_\phi(z | x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) I)$

- **Decoder**: $p_\theta(x | z) = \mathcal{N}(x; \mu_\theta(z), \sigma_\theta^2(z) I)$

- **Prior**: $p(z) = \mathcal{N}(z; 0, I)$

**Vanilla VAE ELBO (Non-KL form)**:

$$ELBO_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]$$

**Hierarchical VAE Structure** (z₂ → z₁ → x):

- **Encoder**: $q_\phi(z_1, z_2 | x) = q_\phi(z_1 | x) \cdot q_\phi(z_2 | z_1)$

  - $q_\phi(z_1 | x) = \mathcal{N}(z_1; \mu_\phi(x), \sigma_\phi^2(x) I)$

  - $q_\phi(z_2 | z_1) = \mathcal{N}(z_2; \mu_\phi(z_1), \sigma_\phi^2(z_1) I)$

- **Decoder**: $p_\theta(x, z_1 | z_2) = p_\theta(x | z_1) \cdot p_\theta(z_1 | z_2)$

  - $p_\theta(x | z_1) = \mathcal{N}(x; \mu_\theta(z_1), \sigma_\theta^2(z_1) I)$

  - $p_\theta(z_1 | z_2) = \mathcal{N}(z_1; \mu_\theta(z_2), \sigma_\theta^2(z_2) I)$

- **Prior**: $p(z_2) = \mathcal{N}(z_2; 0, I)$

**Hierarchical VAE ELBO (Non-KL form)**:

$$ELBO_{\text{HVAE}} = \mathbb{E}_{q_\phi(z_1,z_2|x)} \left[ \log \frac{p_\theta(x, z_1, z_2)}{q_\phi(z_1, z_2|x)} \right]$$

Following the hierarchical VAE formulation, we can write the ELBO for diffusion models. In diffusion models, we have a sequence of latent variables $x_1, x_2, \ldots, x_T$ where $x_T$ is the most abstract (pure noise) and $x_0$ is the data.

**Diffusion Model Structure** (x_T → x_{T-1} → ... → x_1 → x_0):

- **Encoder**: $q(x_1, x_2, \ldots, x_T | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$ - Fixed noise corruption process

- **Decoder**: $p_\theta(x_0, x_1, \ldots, x_{T-1} | x_T) = \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$ - Learned denoising process

- **Prior**: $p(x_T) = \mathcal{N}(x_T; 0, I)$

**Diffusion Model ELBO (Non-KL form)**:

$$ELBO_{\text{Diff}} = \mathbb{E}_{q(x_1,\ldots,x_T|x_0)} \left[ \log \frac{p_\theta(x_0, x_1, \ldots, x_T)}{q(x_1, \ldots, x_T|x_0)} \right]$$

The Negative Evidence Lower BOund (NELBO) is the negative of the ELBO, which is what we actually minimize during training:

$$\mathcal{L}_{\text{Diff}} = -\mathbb{E}_{q(x_1,\ldots,x_T|x_0)} \left[ \log \frac{p_\theta(x_0, x_1, \ldots, x_T)}{q(x_1, \ldots, x_T|x_0)} \right]$$

This can be rewritten as:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{q(x_1,\ldots,x_T|x_0)} \left[ -\log \frac{p_\theta(x_0, x_1, \ldots, x_T)}{q(x_1, \ldots, x_T|x_0)} \right]$$

The decoder learns to predict the mean function $\mu_\theta(x_t, t)$ for the reverse process. Let's derive how this function is parameterized.

The true reverse process $q(x_{t-1} | x_t, x_0)$ can be derived using Bayes' theorem. For Gaussian distributions, this gives us:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \mu_t(x_t, x_0), \sigma_t^2 I)$$

where it can be shown that:

$$\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

and:

$$\sigma_t^2 = \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$$

The learned reverse process is:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

Since we want the learned process to approximate the true reverse process, we parameterize $\mu_\theta(x_t, t)$ to match the form of $\mu_t(x_t, x_0)$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

where $\epsilon_\theta(x_t, t)$ is a neural network that predicts the noise $\epsilon$ given $x_t$ and $t$.

### Rewriting the ELBO for Diffusion Models
Let's rewrite the diffusion model ELBO and transform it to resemble denoising score matching.

Starting with the diffusion model ELBO:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{q(x_1,\ldots,x_T|x_0)} \left[ -\log \frac{p_\theta(x_0, x_1, \ldots, x_T)}{q(x_1, \ldots, x_T|x_0)} \right]$$

We can expand this as:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{q(x_1,\ldots,x_T|x_0)} \left[ -\log p_\theta(x_0, x_1, \ldots, x_T) + \log q(x_1, \ldots, x_T|x_0) \right]$$

The learned joint distribution is:

$$p_\theta(x_0, x_1, \ldots, x_T) = p(x_T) \cdot \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

The true joint distribution is:

$$q(x_1, \ldots, x_T | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

Substituting these:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{q(x_1,\ldots,x_T|x_0)} \left[ -\log p(x_T) - \sum_{t=1}^T \log p_\theta(x_{t-1} | x_t) + \sum_{t=1}^T \log q(x_t | x_{t-1}) \right]$$

For Gaussian distributions, the log-likelihood is:

$$\log \mathcal{N}(x; \mu, \sigma^2 I) = -\frac{1}{2\sigma^2} \|x - \mu\|^2 + C$$

where $C$ is a constant that doesn't depend on the parameters.

For the learned reverse process:

$$\log p_\theta(x_{t-1} | x_t) = -\frac{1}{2\sigma_t^2} \|x_{t-1} - \mu_\theta(x_t, t)\|^2 + C$$

For the true forward process:

$$\log q(x_t | x_{t-1}) = -\frac{1}{2\beta_t} \|x_t - \sqrt{1 - \beta_t} x_{t-1}\|^2 + C$$

Using the definition of $\mu_\theta(x_t, t)$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

The squared error term becomes:

$$\|x_{t-1} - \mu_\theta(x_t, t)\|^2 = \left\|x_{t-1} - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)\right\|^2$$

From the forward process, we know:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

And from the multistep transition:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_{t-1}$$

Substituting these into the squared error:

$$\|x_{t-1} - \mu_\theta(x_t, t)\|^2 = \left\|\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_{t-1} - \frac{1}{\sqrt{\alpha_t}} \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)\right\|^2$$

Using the relationship $\bar{\alpha}_t = \bar{\alpha}_{t-1} \cdot \alpha_t$, we can simplify:

$$\|x_{t-1} - \mu_\theta(x_t, t)\|^2 = \left\|\frac{\beta_t}{\sqrt{\alpha_t(1 - \bar{\alpha}_t)}} (\epsilon - \epsilon_\theta(x_t, t))\right\|^2$$

This simplifies to:

$$\|x_{t-1} - \mu_\theta(x_t, t)\|^2 = \frac{\beta_t^2}{\alpha_t(1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

Substituting back into the ELBO:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{q(x_1,\ldots,x_T|x_0)} \left[ -\log p(x_T) + \sum_{t=1}^T \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2 + \sum_{t=1}^T \frac{1}{2\beta_t} \|x_t - \sqrt{1 - \beta_t} x_{t-1}\|^2 \right]$$

The key term in the ELBO is:

$$\sum_{t=1}^T \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

Let's define:

$$\lambda_t = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)}$$

From the forward process, we know:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$.

The expectation $\mathbb{E}_{q(x_1,\ldots,x_T|x_0)}$ can be rewritten as:

$$\mathbb{E}_{x_0 \sim p_{data}(x_0)} \mathbb{E}_{\epsilon_1, \ldots, \epsilon_T \sim \mathcal{N}(0, I)}$$

Since each $x_t$ is generated as:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$$

The ELBO can be simplified to:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \lambda_t \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)\|^2 \right] + \text{constant terms}$$

where:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$$

This can be written more compactly as:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \lambda_t \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)\|^2 \right] + \text{constant terms}$$

where $x_0 \sim p_{data}(x_0)$ (clean data), $\epsilon \sim \mathcal{N}(0, I)$ (noise), $t \sim \text{Uniform}(1, T)$ (timestep)

The diffusion model ELBO is equivalent to:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \lambda_t \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)\|^2 \right]$$

where $\lambda_t = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)}$ is the weighting factor for each timestep.

**Note**
In the original ELBO, we had two summation-terms:

1. $\sum_{t=1}^T \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$ (noise prediction term)

2. $\sum_{t=1}^T \frac{1}{2\beta_t} \|x_t - \sqrt{1 - \beta_t} x_{t-1}\|^2$ (forward process term)

The second summation-term $\sum_{t=1}^T \frac{1}{2\beta_t} \|x_t - \sqrt{1 - \beta_t} x_{t-1}\|^2$ represents the log-likelihood of the forward process $q(x_t | x_{t-1})$. This summation-term does **not depend on the model parameters $\theta$** because the forward process is fixed and predefined. It only depends on the noise schedule $\beta_t$ and the data. When we take the gradient with respect to $\theta$ to optimize the model, this summation-term vanishes.

### Sampling

While the ELBO loss $\mathcal{L}_{\text{Diff}}$ and the score-based objective are roughly equivalent in terms of what they learn, the **sampling procedures** differ between these two approaches.

In a **Score-Based Model (SBM)**, sampling is performed using **Langevin dynamics**. In a **Diffusion Model (VAE form)**, sampling follows the **learned reverse process**.

The connection between the two approaches comes from the relationship between the score function and the noise predictor:

**Score function**: $s_\theta(x_t, t) = \nabla_x \log p_t(x_t)$

**Noise predictor**: $\epsilon_\theta(x_t, t)$ predicts the noise added during the forward process

**Relationship**: For Gaussian noise, the score function is proportional to the negative noise.

From the forward process, we have:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$.

The distribution of $x_t$ given $x_0$ is:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

The score function is the gradient of the log probability density:

$$s(x_t, t) = \nabla_{x_t} \log q(x_t | x_0)$$

For the Gaussian distribution $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$:

$$\log q(x_t | x_0) = -\frac{1}{2(1 - \bar{\alpha}_t)} \|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2 + C$$

where $C$ is a constant that doesn't depend on $x_t$.

Taking the gradient with respect to $x_t$:

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{1}{1 - \bar{\alpha}_t} (x_t - \sqrt{\bar{\alpha}_t} x_0)$$

From the forward process, we can express $x_0$ in terms of $x_t$ and $\epsilon$:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$$

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{1}{1 - \bar{\alpha}_t} \left(x_t - \sqrt{\bar{\alpha}_t} \cdot \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}\right)$$

Simplifying the expression:

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{1}{1 - \bar{\alpha}_t} \left(x_t - (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)\right)$$

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{1}{1 - \bar{\alpha}_t} \cdot \sqrt{1 - \bar{\alpha}_t} \epsilon$$

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

Therefore, the score function is:

$$s(x_t, t) = \nabla_{x_t} \log q(x_t | x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

In practice, we learn:

- **Score function**: $s_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t | x_0)$

- **Noise predictor**: $\epsilon_\theta(x_t, t) \approx \epsilon$

Therefore, $s_\theta(x_t, t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$.

Both sampling methods work because they're learning the same underlying structure:

1. **Score-based**: Learns the gradient of the log-density at each noise level
2. **Diffusion**: Learns the noise that was added during the forward process

Since the score function and noise predictor are mathematically related, both approaches can generate high-quality samples, but they use different sampling algorithms.