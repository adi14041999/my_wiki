# Score Based Models

## Score Matching

**Energy-Based Model Probability Distribution**

In Energy-Based Models, the probability distribution is defined as:

$$p_\theta(x) = \frac{1}{Z(\theta)} e^{f_\theta(x)}$$

where:

- $f_\theta(x)$ is the energy function (neural network)

- $Z(\theta) = \int e^{f_\theta(x)} dx$ is the partition function (intractable)

Taking the logarithm of the probability distribution:

$$\log p_\theta(x) = f_\theta(x) - \log Z(\theta)$$

Notice that the partition function $Z(\theta)$ appears as a constant term that doesn't depend on $x$.

**Stein Score Function**

The **Stein score function** $s_\theta(x)$ is defined as the gradient of the log probability with respect to $x$:

$$s_\theta(x) = \nabla_x \log p_\theta(x)$$

For Energy-Based Models, the score function equals the gradient of the energy function:

$$s_\theta(x) = \nabla_x \log p_\theta(x) = \nabla_x (f_\theta(x) - \log Z(\theta)) = \nabla_x f_\theta(x)$$

The partition function term $\log Z(\theta)$ disappears because it doesn't depend on $x$.

**Score as a Vector Field**

The score function $s_\theta(x)$ is a **vector field** that assigns a vector to each point $x$ in the data space. This vector has both:

1. **Magnitude**: How quickly the log probability changes

2. **Direction**: The direction of steepest increase in log probability

**Intuition**: The score vector points "uphill" in the log probability landscape, indicating the direction where the model assigns higher probability.

**Example: Gaussian Distribution**

Consider a Gaussian distribution with mean $\mu$ and covariance $\Sigma$:

$$p(x) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)\right)$$

**Log Probability:**

$$\log p(x) = -\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu) - \frac{1}{2}\log((2\pi)^d |\Sigma|)$$

**Score Function:**

$$s(x) = \nabla_x \log p(x) = -\Sigma^{-1}(x - \mu)$$

**Interpretation:**

- The score points toward the mean $\mu$ (direction of higher probability)

- The magnitude is proportional to the distance from the mean

- For isotropic Gaussian ($\Sigma = \sigma^2 I$): $s(x) = -\frac{1}{\sigma^2}(x - \mu)$

This example shows how the score function naturally guides samples toward high-probability regions of the distribution.

### **Score Matching: Comparing Distributions via Vector Fields**

The core idea of score matching is that we want to compare two probability distributions by comparing their respective vector fields of gradients (score functions).

**The Key Insight:**

Instead of directly comparing probability densities $p_{data}(x)$ and $p_\theta(x)$ (which requires computing the intractable partition function), we compare their score functions:

- **Data Score**: $s_{data}(x) = \nabla_x \log p_{data}(x)$
- **Model Score**: $s_\theta(x) = \nabla_x \log p_\theta(x) = \nabla_x f_\theta(x)$

This measures how different the "pointing directions" are at each location $x$.

**L2 Distance Between Score Functions**

One way to compare the score functions is to calculate the average L2 distance between the score of $p_{data}$ and $p_\theta$:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \|s_\theta(x) - s_{data}(x)\|^2 \right]$$

**Note**: This loss function is also called the **Fisher divergence** between $p_{data}(x)$ and $p_\theta(x)$. The Fisher divergence measures the difference between two probability distributions by comparing their score functions (gradients of log densities) rather than the densities themselves.

**Understanding the L2 Distance:**

The L2 norm $\|s_\theta(x) - s_{data}(x)\|^2$ measures the squared Euclidean distance between two vectors:

$$\|s_\theta(x) - s_{data}(x)\|^2 = \sum_{i=1}^d (s_\theta(x)_i - s_{data}(x)_i)^2$$

where $d$ is the dimension of the data space.

**Score matching** is a method for training Energy-Based Models by minimizing the Fisher divergence between the data distribution $p_{data}(x)$ and the model distribution $p_\theta(x)$:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \|s_\theta(x) - s_{data}(x)\|^2 \right]$$

where $s_\theta(x) = \nabla_x \log p_\theta(x)$ and $s_{data}(x) = \nabla_x \log p_{data}(x)$ are the score functions of the model and data distributions respectively.

But how do we figure out $\nabla_x \log p_{data}(x)$ given only samples?

**Score Matching Reformulation (Univariate Case)**

For the univariate case where $x \in \mathbb{R}$, we can rewrite the score matching objective to avoid needing the data score. Let's expand the squared difference:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \left(\frac{d}{dx} \log p_\theta(x) - \frac{d}{dx} \log p_{data}(x)\right)^2 \right]$$

Expanding the square:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \left(\frac{d}{dx} \log p_\theta(x)\right)^2 - \frac{d}{dx} \log p_\theta(x) \cdot \frac{d}{dx} \log p_{data}(x) + \frac{1}{2} \left(\frac{d}{dx} \log p_{data}(x)\right)^2 \right]$$

The key insight is to use integration by parts on the cross term. For any function $f(x)$ and $g(x)$:

$$\int f(x) \frac{d}{dx} g(x) dx = f(x)g(x) - \int \frac{d}{dx} f(x) \cdot g(x) dx$$

Setting $f(x) = \frac{d}{dx} \log p_\theta(x)$ and $g(x) = p_{data}(x)$, we get:

$$\mathbb{E}_{x \sim p_{data}} \left[ \frac{d}{dx} \log p_\theta(x) \cdot \frac{d}{dx} \log p_{data}(x) \right] = \int \frac{d}{dx} \log p_\theta(x) \cdot \frac{d}{dx} \log p_{data}(x) \cdot p_{data}(x) dx$$

Using the chain rule: $\frac{d}{dx} \log p_{data}(x) \cdot p_{data}(x) = \frac{d}{dx} p_{data}(x)$, we get:

$$= \int \frac{d}{dx} \log p_\theta(x) \cdot \frac{d}{dx} p_{data}(x) dx$$

Using integration by parts:

$$= \left. \frac{d}{dx} \log p_\theta(x) \cdot p_{data}(x) \right|_{-\infty}^{\infty} - \int \frac{d^2}{dx^2} \log p_\theta(x) \cdot p_{data}(x) dx$$

**Why does the boundary term vanish?**

The boundary term $\left. \frac{d}{dx} \log p_\theta(x) \cdot p_{data}(x) \right|_{-\infty}^{\infty}$ vanishes under reasonable assumptions:

1. **Data distribution decay**: $p_{data}(x) \to 0$ as $|x| \to \infty$ (most real-world distributions have finite support or decay to zero)
2. **Model score boundedness**: $\frac{d}{dx} \log p_\theta(x)$ grows at most polynomially as $|x| \to \infty$
3. **Product decay**: The product $\frac{d}{dx} \log p_\theta(x) \cdot p_{data}(x) \to 0$ as $|x| \to \infty$

This is a standard assumption in score matching literature and holds for most practical distributions.

Assuming the boundary term vanishes (which is reasonable for well-behaved distributions), we get:

$$\mathbb{E}_{x \sim p_{data}} \left[ \frac{d}{dx} \log p_\theta(x) \cdot \frac{d}{dx} \log p_{data}(x) \right] = -\mathbb{E}_{x \sim p_{data}} \left[ \frac{d^2}{dx^2} \log p_\theta(x) \right]$$

Substituting back into the original objective:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \left(\frac{d}{dx} \log p_\theta(x)\right)^2 + \frac{d^2}{dx^2} \log p_\theta(x) \right] + \text{constant}$$

where the constant term $\frac{1}{2} \mathbb{E}_{x \sim p_{data}} \left[ \left(\frac{d}{dx} \log p_{data}(x)\right)^2 \right]$ doesn't depend on $\theta$ and can be ignored during optimization.

**Key Insight**: This reformulation allows us to train the model using only samples from $p_{data}(x)$ and the derivatives of our model's log probability, without needing access to the data score function.

**Score Matching Reformulation (Multivariate Case)**

For the multivariate case where $x \in \mathbb{R}^d$, we can extend the univariate derivation. The score matching objective becomes:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \|\nabla_x \log p_\theta(x) - \nabla_x \log p_{data}(x)\|^2 \right]$$

Expanding the squared norm:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \|\nabla_x \log p_\theta(x)\|^2 - \nabla_x \log p_\theta(x)^T \nabla_x \log p_{data}(x) + \frac{1}{2} \|\nabla_x \log p_{data}(x)\|^2 \right]$$

The key insight is to use integration by parts on the cross term. For the multivariate case, we need to handle each component separately. Let $s_\theta(x)_i$ and $s_{data}(x)_i$ denote the $i$-th component of the respective score functions.

For each component $i$, we have:

$$\mathbb{E}_{x \sim p_{data}} \left[ s_\theta(x)_i \cdot s_{data}(x)_i \right] = \int s_\theta(x)_i \cdot s_{data}(x)_i \cdot p_{data}(x) dx$$

Using the chain rule: $s_{data}(x)_i \cdot p_{data}(x) = \frac{\partial}{\partial x_i} p_{data}(x)$, we get:

$$= \int s_\theta(x)_i \cdot \frac{\partial}{\partial x_i} p_{data}(x) dx$$

Using integration by parts (assuming boundary terms vanish):

$$= -\int \frac{\partial}{\partial x_i} s_\theta(x)_i \cdot p_{data}(x) dx = -\mathbb{E}_{x \sim p_{data}} \left[ \frac{\partial}{\partial x_i} s_\theta(x)_i \right]$$

**Why do the boundary terms vanish in the multivariate case?**

For each component $i$, the boundary term is:

$$\left. s_\theta(x)_i \cdot p_{data}(x) \right|_{x_i = -\infty}^{x_i = \infty}$$

This vanishes under similar assumptions as the univariate case:

1. **Data distribution decay**: $p_{data}(x) \to 0$ as $\|x\| \to \infty$ in any direction
2. **Model score boundedness**: Each component $s_\theta(x)_i$ grows at most polynomially as $\|x\| \to \infty$
3. **Product decay**: The product $s_\theta(x)_i \cdot p_{data}(x) \to 0$ as $\|x\| \to \infty$ for each component

These assumptions ensure that the boundary terms vanish for all components, allowing us to apply integration by parts component-wise.

Summing over all components:

$$\sum_{i=1}^d \mathbb{E}_{x \sim p_{data}} \left[ s_\theta(x)_i \cdot s_{data}(x)_i \right] = -\sum_{i=1}^d \mathbb{E}_{x \sim p_{data}} \left[ \frac{\partial}{\partial x_i} s_\theta(x)_i \right] = -\mathbb{E}_{x \sim p_{data}} \left[ \text{tr}(\nabla_x s_\theta(x)) \right]$$

where $\text{tr}(\nabla_x s_\theta(x)) = \sum_{i=1}^d \frac{\partial}{\partial x_i} s_\theta(x)_i$ is the trace of the Jacobian matrix of the score function.

Substituting back into the original objective:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \|\nabla_x \log p_\theta(x)\|^2 + \text{tr}(\nabla_x \nabla_x \log p_\theta(x)) \right] + \text{constant}$$

where the constant term $\frac{1}{2} \mathbb{E}_{x \sim p_{data}} \left[ \|\nabla_x \log p_{data}(x)\|^2 \right]$ doesn't depend on $\theta$ and can be ignored during optimization.

**Key Insight**: The multivariate case introduces the trace of the Hessian matrix $\text{tr}(\nabla_x \nabla_x \log p_\theta(x))$.

### Score Matching Algorithm

The score matching algorithm follows these steps:

**Sample a mini-batch of datapoints**: $\{x_1, x_2, \ldots, x_n\} \sim p_{data}(x)$

**Estimate the score matching loss with the empirical mean**: 

$$\mathcal{L}_{SM}(\theta) \approx \frac{1}{n} \sum_{i=1}^n \left[ \frac{1}{2} \|\nabla_x \log p_\theta(x_i)\|^2 + \text{tr}(\nabla_x \nabla_x \log p_\theta(x_i)) \right]$$

**Stochastic gradient descent**: Update parameters using gradients of the estimated loss

**Advantages:**
* **No need to sample from EBM**: Unlike other training methods for energy-based models, score matching doesn't require generating samples from the model during training. This avoids the computational expense and potential instability of MCMC sampling.
* **Direct optimization**: The objective directly measures how well the model's score function matches the data distribution's score function.
* **Theoretically sound**: Score matching provides a consistent estimator under mild conditions.

**Disadvantages:**
* **Computing the Hessian is expensive**: The term $\text{tr}(\nabla_x \nabla_x \log p_\theta(x))$ requires computing second derivatives, which scales quadratically with the input dimension and can be computationally prohibitive for large models.
* **Memory requirements**: Storing and computing Hessians for large neural networks requires significant memory.
* **Numerical instability**: Second derivatives can be numerically unstable, especially for deep networks.

**Computational Complexity:**
For a model with $d$ input dimensions and $m$ parameters, computing the Hessian trace requires $O(d^2 \cdot m)$ operations, making it impractical for high-dimensional data like images.

## Recap: Distances for Training EBMs

When training Energy-Based Models, we need to measure how close our model distribution $p_\theta(x)$ is to the data distribution $p_{data}(x)$. Here are the main approaches:

### Contrastive Divergence

Contrastive divergence measures the difference between the data distribution and the model distribution using KL divergence:

$$\mathcal{L}_{CD}(\theta) = D_{KL}(p_{data}(x) \| p_\theta(x)) - D_{KL}(p_\theta(x) \| p_{data}(x))$$

**Key insight**: This objective encourages the model to match the data distribution while preventing mode collapse.

**Challenge**: Computing the KL divergence requires sampling from the model distribution $p_\theta(x)$, which is typically done using MCMC methods like Langevin dynamics or Hamiltonian Monte Carlo.

### Fisher Divergence (Score Matching)

Fisher divergence measures the difference between the score functions (gradients of log densities) of the two distributions:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \|\nabla_x \log p_\theta(x) - \nabla_x \log p_{data}(x)\|^2 \right]$$

**Key insight**: Instead of comparing probability densities directly, we compare their gradients, which avoids the need to compute the intractable partition function.

**Advantage**: No need to sample from the model during training, making it computationally more efficient than contrastive divergence.

**Challenge**: Requires computing second derivatives (Hessian) of the log probability, which can be expensive for high-dimensional data.

## Noise Contrastive Estimation

**Learning an EBM by contrasting it with a noise distribution.**

We have the data distribution $p_{data}(x)$. We have the noise distribution $p_n(x)$ which should be analytically tractable and easy to sample from. We train a discriminator $D_\theta(x) \in [0, 1]$ to distinguish between data samples and noise samples.

### GAN Objective

The discriminator is trained to maximize the probability of correctly classifying data and noise samples. The objective is:

$$\mathcal{L}_{NCE}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \log D_\theta(x) \right] + \mathbb{E}_{x \sim p_n} \left[ \log(1 - D_\theta(x)) \right]$$

This is the standard binary cross-entropy loss for binary classification, where $D_\theta(x)$ represents the probability that $x$ comes from the data distribution and $1 - D_\theta(x)$ represents the probability that $x$ comes from the noise distribution.

### Optimal Discriminator

The optimal discriminator $D^*(x)$ that maximizes this objective is given by:

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_n(x)}$$

**Connection to Model Distribution:**

By training the discriminator, we are implicitly learning $p_\theta(x)$ which approximates $p_{data}(x)$. Here's how this works:

* **Discriminator as Ratio Estimator**: The optimal discriminator $D^*(x)$ represents the ratio of data probability to the sum of data and noise probabilities.

* **Model Distribution Recovery**: If we parameterize our model distribution as $p_\theta(x) = \frac{e^{f_\theta(x)}}{Z(\theta)}$, then the discriminator learns to approximate:

$$D_\theta(x) = \frac{e^{f_\theta(x)}}{e^{f_\theta(x)} + p_n(x)}$$

The partition function $Z(\theta)$ disappears because we parameterize the discriminator directly in terms of the unnormalized energy function $e^{f_\theta(x)}$, not the normalized probability $p_\theta(x) = \frac{e^{f_\theta(x)}}{Z(\theta)}$.

If we had used the normalized probability, we would have:

$$D_\theta(x) = \frac{p_\theta(x)}{p_\theta(x) + p_n(x)} = \frac{\frac{e^{f_\theta(x)}}{Z(\theta)}}{\frac{e^{f_\theta(x)}}{Z(\theta)} + p_n(x)} = \frac{e^{f_\theta(x)}}{e^{f_\theta(x)} + Z(\theta) \cdot p_n(x)}$$

But by parameterizing directly with $e^{f_\theta(x)}$, we avoid the need to compute $Z(\theta)$ entirely. This is the key advantage of noise contrastive estimation - it sidesteps the intractable partition function by working with unnormalized probabilities.

$$D_\theta(x) \approx \frac{p_{data}(x)}{p_{data}(x) + p_n(x)}$$

* **Energy Function Learning**: Through the discriminator training, the function $f_\theta(x)$ learns to capture the structure of the data distribution, effectively becoming an energy function that represents $\log p_{data}(x)$ up to a constant.

**Key Insight**: The discriminator training process transforms the intractable problem of learning $p_\theta(x)$ directly into a tractable binary classification problem, where the learned discriminator implicitly encodes the energy function of the data distribution.

## Denoising & Slicing Score Matching