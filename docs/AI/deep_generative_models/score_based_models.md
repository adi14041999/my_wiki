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

$$= \int \frac{d}{dx} \log p_\theta(x) \cdot \frac{d}{dx} p_{data}(x) dx$$

Using integration by parts:

$$= \left. \frac{d}{dx} \log p_\theta(x) \cdot p_{data}(x) \right|_{-\infty}^{\infty} - \int \frac{d^2}{dx^2} \log p_\theta(x) \cdot p_{data}(x) dx$$

Assuming the boundary term vanishes (which is reasonable for well-behaved distributions), we get:

$$\mathbb{E}_{x \sim p_{data}} \left[ \frac{d}{dx} \log p_\theta(x) \cdot \frac{d}{dx} \log p_{data}(x) \right] = -\mathbb{E}_{x \sim p_{data}} \left[ \frac{d^2}{dx^2} \log p_\theta(x) \right]$$

Substituting back into the original objective:

$$\mathcal{L}_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \frac{1}{2} \left(\frac{d}{dx} \log p_\theta(x)\right)^2 + \frac{d^2}{dx^2} \log p_\theta(x) \right] + \text{constant}$$

where the constant term $\frac{1}{2} \mathbb{E}_{x \sim p_{data}} \left[ \left(\frac{d}{dx} \log p_{data}(x)\right)^2 \right]$ doesn't depend on $\theta$ and can be ignored during optimization.

**Key Insight**: This reformulation allows us to train the model using only samples from $p_{data}(x)$ and the derivatives of our model's log probability, without needing access to the data score function.

## Noise Contrastive Estimation
## Training Score Based Models
## Denoising & Slicing Score Matching