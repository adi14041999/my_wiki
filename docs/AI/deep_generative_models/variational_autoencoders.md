# Variational autoencoders
## Representation
Consider a directed, latent variable model as shown below.

![Graphical model for a directed, latent variable model.](vae_zx.png)

In the model above, $z$ and $x$ denote the latent and observed variables respectively. The joint distribution expressed by this model is given as

$$p(x,z) = p(x|z)p(z).$$

From a generative modeling perspective, this model describes a generative process for the observed data $x$ using the following procedure:

$$z \sim p(z)$$

$$x \sim p(x|z)$$

If one adopts the belief that the latent variables $z$ somehow encode semantically meaningful information about $x$, it is natural to view this generative process as first generating the "high-level" semantic information about $x$ first before fully generating $x$.

We now consider a family of distributions $\mathcal{Z}$ where $p(z) \in \mathcal{Z}$ describes a probability distribution over $z$. Next, consider a family of conditional distributions $\mathcal{X|Z}$ where $p(x|z) \in \mathcal{X|Z}$ describes a conditional probability distribution over $x$ given $z$. Then our hypothesis class of generative models is the set of all possible combinations

$$\mathcal{X,Z} = \{p(x,z) \mid p(z) \in \mathcal{Z}, p(x|z) \in \mathcal{X|Z}\}.$$

Given a dataset $\mathcal{D} = \{x^{(1)}, \ldots, x^{(n)}\}$, we are interested in the following learning and inference tasks:

1. Selecting $p \in \mathcal{X,Z}$ that "best" fits $\mathcal{D}$.
2. Given a sample $x$ and a model $p \in \mathcal{X,Z}$, what is the posterior distribution over the latent variables $z$?

The posterior distribution $p(z|x)$ represents our updated beliefs about the latent variables $z$ after observing the data $x$. In other words, it tells us what values of $z$ are most likely to have generated the observed $x$. This is particularly useful for tasks like feature extraction, where we want to understand what latent factors might have generated our observed data.

## Learning Directed Latent Variable Models

One way to measure how closely $p(x,z)$ fits the observed dataset $\mathcal{D}$ is to measure the Kullback-Leibler (KL) divergence between the data distribution (which we denote as $p_{data}(x)$) and the model's marginal distribution $p(x) = \int p(x,z)dz$. The distribution that "best" fits the data is thus obtained by minimizing the KL divergence.

$$\min_{p \in \mathcal{X,Z}} D_{KL}(p_{data}(x) \| p(x)).$$

As we have seen previously, optimizing an empirical estimate of the KL divergence is equivalent to maximizing the marginal log-likelihood $\log p(x)$ over $\mathcal{D}$:

$$\max_{p \in \mathcal{X,Z}} \sum_{x \in \mathcal{D}} \log p(x) = \max_{p \in \mathcal{X,Z}} \sum_{x \in \mathcal{D}} \log \int p(x,z)dz.$$

However, it turns out this problem is generally intractable for high-dimensional $z$ as it involves an integration (or sums in the case $z$ is discrete) over all the possible latent sources of variation $z$. This intractability arises from several challenges:

1. **Computational Complexity**: The integral $\int p(x,z)dz$ requires evaluating the joint distribution $p(x,z)$ for all possible values of $z$. In high-dimensional spaces, this becomes computationally prohibitive as the number of points to evaluate grows exponentially with the dimension of $z$.

2. **Numerical Integration**: Even if we could evaluate the integrand at all points, computing the integral numerically becomes increasingly difficult as the dimension of $z$ grows. Traditional numerical integration methods like quadrature become impractical in high dimensions.

3. **Posterior Inference**: The intractability of the marginal likelihood also makes it difficult to compute the posterior distribution $p(z|x)$, which is crucial for tasks like feature extraction and data generation.

This intractability motivates the need for approximate inference methods, such as variational inference. One option is to estimate the objective via Monte Carlo. For any given datapoint $x$, we can obtain the following estimate for its marginal log-likelihood:

$$\log p(x) \approx \log \frac{1}{k} \sum_{i=1}^k p(x|z^{(i)}), \text{ where } z^{(i)} \sim p(z)$$

This Monte Carlo estimate is derived as follows:

First, recall that the marginal likelihood $p(x)$ can be written as an expectation:

$$p(x) = \int p(x|z)p(z)dz = \mathbb{E}_{z \sim p(z)}[p(x|z)]$$

The Monte Carlo method approximates this expectation by drawing $k$ samples from $p(z)$ and computing their average:

$$\mathbb{E}_{z \sim p(z)}[p(x|z)] \approx \frac{1}{k} \sum_{i=1}^k p(x|z^{(i)}), \text{ where } z^{(i)} \sim p(z)$$

Taking the logarithm of both sides gives us our final estimate:

$$\log p(x) \approx \log \frac{1}{k} \sum_{i=1}^k p(x|z^{(i)}), \text{ where } z^{(i)} \sim p(z)$$

This approximation becomes more accurate as $k$ increases, but at the cost of more computational resources. The key insight is that we're using random sampling to approximate the intractable integral, trading exact computation for statistical estimation.

Rather than maximizing the log-likelihood directly, an alternate is to instead construct a lower bound that is more amenable to optimization. To do so, we note that evaluating the marginal likelihood $p(x)$ is at least as difficult as as evaluating the posterior $p(z|x)$ for any latent vector $z$ since by definition $p(z|x) = p(x,z)/p(x)$.

Next, we introduce a variational family $\mathcal{Q}$ of distributions that approximate the true, but intractable posterior $p(z|x)$. Further henceforth, we will assume a parameteric setting where any distribution in the model family $\mathcal{X,Z}$ is specified via a set of parameters $\theta \in \Theta$ and distributions in the variational family $\mathcal{Q}$ are specified via a set of parameters $\lambda \in \Lambda$.

Given $\mathcal{X,Z}$ and $\mathcal{Q}$, we note that the following relationships hold true for any $x$ and all variational distributions $q_\lambda(z) \in \mathcal{Q}$:

$$\log p_\theta(x) = \log \int p_\theta(x,z)dz = \log \int \frac{q_\lambda(z)}{q_\lambda(z)}p_\theta(x,z)dz \geq \mathbb{E}_{q_\lambda(z)}\left[\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right] := \text{ELBO}(x;\theta,\lambda)$$

where we have used Jensen's inequality in the final step. The key insight here is that since the logarithm function is concave, Jensen's inequality tells us that for any random variable $X$ and concave function $f$, we have $\mathbb{E}[f(X)] \leq f(\mathbb{E}[X])$. In our case:

We first multiply and divide by $q_\lambda(z)$ inside the integral to get:

$$\log \int \frac{q_\lambda(z)}{q_\lambda(z)}p_\theta(x,z)dz = \log \int q_\lambda(z)\frac{p_\theta(x,z)}{q_\lambda(z)}dz$$

The integral $\int q_\lambda(z)\frac{p_\theta(x,z)}{q_\lambda(z)}dz$ can be seen as an expectation $\mathbb{E}_{q_\lambda(z)}\left[\frac{p_\theta(x,z)}{q_\lambda(z)}\right]$

Since $\log$ is a concave function, Jensen's inequality gives us:

$$\log \mathbb{E}_{q_\lambda(z)}\left[\frac{p_\theta(x,z)}{q_\lambda(z)}\right] \geq \mathbb{E}_{q_\lambda(z)}\left[\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right]$$

This inequality is what allows us to obtain a lower bound on the log-likelihood, which we call the Evidence Lower BOund (ELBO). The ELBO admits a tractable unbiased Monte Carlo estimator

$$\frac{1}{k}\sum_{i=1}^k \log\frac{p_\theta(x,z^{(i)})}{q_\lambda(z^{(i)})}, \text{ where } z^{(i)} \sim q_\lambda(z),$$

so long as it is easy to sample from and evaluate densities for $q_\lambda(z)$.

In summary, we can learn a latent variable model by maximizing the ELBO with respect to both the model parameters $\theta$ and the variational parameters $\lambda$ for any given datapoint $x$:

$$\max_\theta \sum_{x \in \mathcal{D}} \max_\lambda \mathbb{E}_{q_\lambda(z)}\left[\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right].$$

This optimization objective can be broken down into two parts:

1. **Inner Optimization**: For each datapoint $x$, we find the best variational parameters $\lambda$ that make $q_\lambda(z)$ as close as possible to the true posterior $p(z|x)$. This is done by maximizing the ELBO with respect to $\lambda$. 

   Why do we need $q_\lambda(z)$ to approximate $p(z|x)$? Since $p(x) = p(x,z)/p(z|x)$, as $q_\lambda(z)$ tends to $p(z|x)$, the ratio $p(x,z)/q_\lambda(z)$ tends to $p(x)$. This means that by making our variational approximation closer to the true posterior, we get a better estimate of the marginal likelihood $p(x)$.

2. **Outer Optimization**: Across all datapoints in the dataset $\mathcal{D}$, we find the best model parameters $\theta$ that maximize the average ELBO. This improves the generative model's ability to explain the data.

The outer sum $\sum_{x \in \mathcal{D}}$ is necessary because we want to learn a model that works well for all datapoints in our dataset, not just a single example. This is equivalent to maximizing the average ELBO across all datapoints.

## Black-Box Variational Inference
We shall focus on first-order stochastic gradient methods for optimizing the ELBO.
This inspires Black-Box Variational Inference (BBVI), a general-purpose Expectation-Maximization-like algorithm for variational learning of latent variable models, where, for each mini-batch $\mathcal{B} = \{x^{(1)}, \ldots, x^{(m)}\}$, the following two steps are performed.

**Step 1**

We first do per-sample optimization of $q$ by iteratively applying the update

$$\lambda^{(i)} \leftarrow \lambda^{(i)} + \tilde{\nabla}_\lambda \text{ELBO}(x^{(i)}; \theta, \lambda^{(i)}),$$

where $\text{ELBO}(x; \theta, \lambda) = \mathbb{E}_{q_\lambda(z)}\left[\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right]$, and $\tilde{\nabla}_\lambda$ denotes an unbiased estimate of the ELBO gradient. This step seeks to approximate the log-likelihood $\log p_\theta(x^{(i)})$.

**Step 2**

We then perform a single update step based on the mini-batch

$$\theta \leftarrow \theta + \tilde{\nabla}_\theta \sum_i \text{ELBO}(x^{(i)}; \theta, \lambda^{(i)}),$$

which corresponds to the step that hopefully moves $p_\theta$ closer to $p_{data}$.

## Gradient Estimation

The gradients $\nabla_\lambda \text{ELBO}$ and $\nabla_\theta \text{ELBO}$ can be estimated via Monte Carlo sampling. While it is straightforward to construct an unbiased estimate of $\nabla_\theta \text{ELBO}$ by simply pushing $\nabla_\theta$ through the expectation operator, the same cannot be said for $\nabla_\lambda$. Instead, we see that

$$\nabla_\lambda \mathbb{E}_{q_\lambda(z)}\left[\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right] = \mathbb{E}_{q_\lambda(z)}\left[\left(\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right) \cdot \nabla_\lambda \log q_\lambda(z)\right].$$

This equality follows from the log-derivative trick (also commonly referred to as the REINFORCE trick). To derive this, we start with the gradient of the expectation:

$$\nabla_\lambda \mathbb{E}_{q_\lambda(z)}\left[\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right] = \nabla_\lambda \int q_\lambda(z) \log\frac{p_\theta(x,z)}{q_\lambda(z)} dz$$

Using the product rule and chain rule:

$$= \int \nabla_\lambda q_\lambda(z) \cdot \log\frac{p_\theta(x,z)}{q_\lambda(z)} + q_\lambda(z) \cdot \nabla_\lambda \log\frac{p_\theta(x,z)}{q_\lambda(z)} dz$$

The second term vanishes because:
$\nabla_\lambda \log\frac{p_\theta(x,z)}{q_\lambda(z)} = \nabla_\lambda [\log p_\theta(x,z) - \log q_\lambda(z)]$.
Since $p_\theta(x,z)$ doesn't depend on $\lambda$, $\nabla_\lambda \log p_\theta(x,z) = 0$. Therefore, $\nabla_\lambda \log\frac{p_\theta(x,z)}{q_\lambda(z)} = -\nabla_\lambda \log q_\lambda(z)$. 
When we multiply by $q_\lambda(z)$ and integrate, we get:

$$\int q_\lambda(z) \cdot (-\nabla_\lambda \log q_\lambda(z)) dz = -\int \nabla_\lambda q_\lambda(z) dz = -\nabla_\lambda \int q_\lambda(z) dz = -\nabla_\lambda 1 = 0$$

where we used the fact that $\int q_\lambda(z) dz = 1$ for any valid probability distribution.

For the first term, we use the identity $\nabla_\lambda q_\lambda(z) = q_\lambda(z) \nabla_\lambda \log q_\lambda(z)$:

$$= \int q_\lambda(z) \nabla_\lambda \log q_\lambda(z) \cdot \log\frac{p_\theta(x,z)}{q_\lambda(z)} dz$$

This can be rewritten as an expectation:

$$= \mathbb{E}_{q_\lambda(z)}\left[\left(\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right) \cdot \nabla_\lambda \log q_\lambda(z)\right]$$

The gradient estimator $\tilde{\nabla}_\lambda \text{ELBO}$ is thus

$$\frac{1}{k}\sum_{i=1}^k \left[\left(\log\frac{p_\theta(x,z^{(i)})}{q_\lambda(z^{(i)})}\right) \cdot \nabla_\lambda \log q_\lambda(z^{(i)})\right], \text{ where } z^{(i)} \sim q_\lambda(z).$$

However, it is often noted that this estimator suffers from high variance. One of the key contributions of the variational autoencoder paper is the reparameterization trick, which introduces a fixed, auxiliary distribution $p(\epsilon)$ and a differentiable function $T(\epsilon; \lambda)$ such that the procedure

$$\epsilon \sim p(\epsilon)$$

$$z \leftarrow T(\epsilon; \lambda),$$

is equivalent to sampling from $q_\lambda(z)$. This two-step procedure works as follows:

1. First, we sample $\epsilon$ from a fixed distribution $p(\epsilon)$ that doesn't depend on $\lambda$ (e.g., standard normal)
2. Then, we transform this sample using a deterministic function $T(\epsilon; \lambda)$ that depends on $\lambda$

The key insight is that if we choose $T$ appropriately, the distribution of $z = T(\epsilon; \lambda)$ will be exactly $q_\lambda(z)$. For example, if $q_\lambda(z)$ is a normal distribution with mean $\mu_\lambda$ and standard deviation $\sigma_\lambda$, we can use:

$p(\epsilon) = \mathcal{N}(0, 1)$

$T(\epsilon; \lambda) = \mu_\lambda + \sigma_\lambda \cdot \epsilon$

By the Law of the Unconscious Statistician, we can see that

$$\nabla_\lambda \mathbb{E}_{q_\lambda(z)}\left[\log\frac{p_\theta(x,z)}{q_\lambda(z)}\right] = \mathbb{E}_{p(\epsilon)}\left[\nabla_\lambda \log\frac{p_\theta(x,T(\epsilon; \lambda))}{q_\lambda(T(\epsilon; \lambda))}\right].$$

In contrast to the REINFORCE trick, the reparameterization trick is often noted empirically to have lower variance and thus results in more stable training.

## Parameterizing Distributions via Deep Neural Networks

So far, we have described $p_\theta(x,z)$ and $q_\lambda(z)$ in the abstract. To instantiate these objects, we consider choices of parametric distributions for $p_\theta(z)$, $p_\theta(x|z)$, and $q_\lambda(z)$. A popular choice for $p_\theta(z)$ is the unit Gaussian

$$p_\theta(z) = \mathcal{N}(z|0,I),$$

in which case $\theta$ is simply the empty set since the prior is a fixed distribution.

In the case where $p_\theta(x|z)$ is a Gaussian distribution, we can thus represent it as

$$p_\theta(x|z) = \mathcal{N}(x|\mu_\theta(z), \Sigma_\theta(z)),$$

where $\mu_\theta(z)$ and $\Sigma_\theta(z)$ are neural networks that specify the mean and covariance matrix for the Gaussian distribution over $x$ when conditioned on $z$.

Finally, the variational family for the proposal distribution $q_\lambda(z)$ needs to be chosen judiciously so that the reparameterization trick is possible. Many continuous distributions in the location-scale family can be reparameterized. In practice, a popular choice is again the Gaussian distribution, where

$$\begin{align*}
\lambda &= (\mu, \Sigma) \\
q_\lambda(z) &= \mathcal{N}(z|\mu, \Sigma) \\
p(\varepsilon) &= \mathcal{N}(z|0,I) \\
T(\varepsilon; \lambda) &= \mu + \Sigma^{1/2}\varepsilon,
\end{align*}$$

where $\Sigma^{1/2}$ is the Cholesky decomposition of $\Sigma$. For simplicity, practitioners often restrict $\Sigma$ to be a diagonal matrix (which restricts the distribution family to that of factorized Gaussians).

The reparameterization trick consists of four key steps:

1. **Parameter Definition**: We define the variational parameters $\lambda$ as a tuple containing the mean vector $\mu$ and covariance matrix $\Sigma$ of our Gaussian distribution. These parameters will be learned during training.

2. **Variational Distribution**: We specify that our variational distribution $q_\lambda(z)$ is a Gaussian distribution parameterized by $\mu$ and $\Sigma$. This is the distribution we ideally want to sample from.

3. **Auxiliary Distribution**: Instead of sampling directly from $q_\lambda(z)$, we introduce a fixed auxiliary distribution $p(\varepsilon)$ which is a standard normal distribution (mean 0, identity covariance). This distribution doesn't depend on our parameters $\lambda$.

4. **Transformation Function**: We define a deterministic function $T(\varepsilon; \lambda)$ that transforms samples from the auxiliary distribution into samples from our variational distribution. The transformation is given by $\mu + \Sigma^{1/2}\varepsilon$, where $\Sigma^{1/2}$ is the Cholesky decomposition of $\Sigma$.

The key insight is that instead of sampling directly from $q_\lambda(z)$, we can:
1. Sample $\varepsilon$ from the standard normal distribution $p(\varepsilon)$
2. Transform it using $T(\varepsilon; \lambda)$ to make it seem like we're getting a sample from $q_\lambda(z)$

This trick is crucial because it allows us to compute gradients with respect to $\lambda$ through the sampling process. Since the transformation $T$ is differentiable, we can backpropagate through it to update the parameters $\lambda$ during training. This is why the reparameterization trick often leads to lower variance in gradient estimates compared to the REINFORCE trick.

## Amortized Variational Inference

A noticeable limitation of black-box variational inference is that Step 1 executes an optimization subroutine that is computationally expensive. Recall that the goal of Step 1 is to find

$$\lambda^* = \arg\max_{\lambda \in \Lambda} \text{ELBO}(x; \theta, \lambda).$$

For a given choice of $\theta$, there is a well-defined mapping from $x \mapsto \lambda^*$. A key realization is that this mapping can be learned. In particular, one can train an encoding function (parameterized by $\phi$) $f_\phi: \mathcal{X} \to \Lambda$ (where $\Lambda$ is the space of $\lambda$ parameters) on the following objective

$$\max_\phi \sum_{x \in \mathcal{D}} \text{ELBO}(x; \theta, f_\phi(x)).$$

It is worth noting at this point that $f_\phi(x)$ can be interpreted as defining the conditional distribution $q_\phi(z|x)$. With a slight abuse of notation, we define

$$\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\right],$$

and rewrite the optimization problem as

$$\max_\phi \sum_{x \in \mathcal{D}} \text{ELBO}(x; \theta, \phi).$$

It is also worth noting that optimizing $\phi$ over the entire dataset as a subroutine every time we sample a new mini-batch is clearly not reasonable. However, if we believe that $f_\phi$ is capable of quickly adapting to a close-enough approximation of $\lambda^*$ given the current choice of $\theta$, then we can interleave the optimization of $\phi$ and $\theta$. This yields the following procedure, where for each mini-batch $\mathcal{B} = \{x^{(1)}, \ldots, x^{(m)}\}$, we perform the following two updates jointly:

$$\begin{align*}
\phi &\leftarrow \phi + \tilde{\nabla}_\phi \sum_{x \in \mathcal{B}} \text{ELBO}(x; \theta, \phi) \\
\theta &\leftarrow \theta + \tilde{\nabla}_\theta \sum_{x \in \mathcal{B}} \text{ELBO}(x; \theta, \phi),
\end{align*}$$

rather than running BBVI's Step 1 as a subroutine. By leveraging the learnability of $x \mapsto \lambda^*$, this optimization procedure amortizes the cost of variational inference. If one further chooses to define $f_\phi$ as a neural network, the result is the variational autoencoder.

### Steps of Amortized Variational Inference

Let's break down the amortized variational inference procedure in detail:

1. **Initial Setup**:
   - We have a dataset $\mathcal{D} = \{x^{(1)}, \ldots, x^{(n)}\}$
   - We have a generative model $p_\theta(x,z)$ with parameters $\theta$
   - We want to learn both the model parameters $\theta$ and the variational parameters $\lambda$ for each datapoint

2. **Traditional BBVI Approach**:
   - For each datapoint $x$, we would need to run an optimization to find:

$$\lambda^* = \arg\max_{\lambda \in \Lambda} \text{ELBO}(x; \theta, \lambda)$$
   
   - This is computationally expensive as it requires running an optimization subroutine for each datapoint

3. **Key Insight - Learnable Mapping**:
   - Instead of optimizing $\lambda$ separately for each $x$, we realize that there's a mapping from $x$ to $\lambda^*$
   - This mapping can be learned using a function $f_\phi: \mathcal{X} \to \Lambda$ parameterized by $\phi$
   - The function $f_\phi$ takes a datapoint $x$ and outputs the variational parameters $\lambda$

4. **Training the Encoder**:
   - We train $f_\phi$ to maximize the ELBO across all datapoints:

$$\max_\phi \sum_{x \in \mathcal{D}} \text{ELBO}(x; \theta, f_\phi(x))$$

   - This is equivalent to learning a conditional distribution $q_\phi(z|x)$

5. **Joint Optimization**:
   - Instead of running BBVI's Step 1 as a subroutine, we interleave the optimization of $\phi$ and $\theta$
   - For each mini-batch $\mathcal{B} = \{x^{(1)}, \ldots, x^{(m)}\}$, we perform two updates:

$$\begin{align*}
\phi &\leftarrow \phi + \tilde{\nabla}_\phi \sum_{x \in \mathcal{B}} \text{ELBO}(x; \theta, \phi) \\
\theta &\leftarrow \theta + \tilde{\nabla}_\theta \sum_{x \in \mathcal{B}} \text{ELBO}(x; \theta, \phi)
\end{align*}$$

6. **Practical Implementation**:
   - When $f_\phi$ is implemented as a neural network, we get a variational autoencoder
   - The encoder network $f_\phi$ maps inputs $x$ to variational parameters
   - The decoder network maps latent variables $z$ to reconstructed inputs
   - Both networks are trained end-to-end using the ELBO objective

The key advantage of this approach is that it amortizes the cost of variational inference by learning a single function $f_\phi$ that can quickly approximate the optimal variational parameters for any input $x$, rather than running a separate optimization for each datapoint.

### Decomposition of the Negative ELBO
Starting with the definition of the ELBO:

$$\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\right]$$

We can expand the joint distribution $p_\theta(x,z)$ using the chain rule of probability:

$$p_\theta(x,z) = p_\theta(x|z)p(z)$$

Substituting this into the ELBO:

$$\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}\right]$$

Using the properties of logarithms, we can split this into three terms:

$$\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi(z|x)}[\log p(z)] - \mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x)]$$

The second and third terms can be combined to form the KL divergence between $q_\phi(z|x)$ and $p(z)$:

$$\mathbb{E}_{q_\phi(z|x)}[\log p(z)] - \mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x)] = -\mathbb{E}_{q_\phi(z|x)}\left[\log\frac{q_\phi(z|x)}{p(z)}\right] = -D_{KL}(q_\phi(z|x) \| p(z))$$

Therefore, the ELBO can be written as:

$$\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

It is insightful to note that the negative ELBO can be decomposed into two terms:

$$-\text{ELBO}(x; \theta, \phi) = \underbrace{-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Divergence}}$$

This decomposition reveals two key components of the training objective:

1. **Reconstruction Loss**: $-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$
   - This term measures how well the model can reconstruct the input $x$ from its latent representation $z$
   - It encourages the encoder to produce latent codes that preserve the essential information about the input
   - In practice, this is often implemented as the mean squared error or binary cross-entropy between the input and its reconstruction

2. **KL Divergence**: $D_{KL}(q_\phi(z|x) \| p(z))$
   - This term measures how far the approximate posterior $q_\phi(z|x)$ is from the prior $p(z)$
   - It encourages the latent space to follow the prior distribution (typically a standard normal distribution)




