# Recap at this point

## The Generative Modeling Framework

We have some data from an unknown probability distribution, that we denote $p_{data}$. We have a model family $\mathcal{M}$ which is a set of probability distributions. We denote some kind of notion of similarity between $p_{data}$ and the distributions in $\mathcal{M}$. We try to find the probability distribution $p_\theta$ that is the closest to $p_{data}$ in this notion of similarity.

## Model Families We've Explored

We have seen different ways of constructing probability distributions in $\mathcal{M}$:

1. **Autoregressive Models**: $p_\theta(x) = \prod_{i=1}^N p_\theta(x_i|x_{<i})$
2. **Variational Autoencoders (VAEs)**: $p_\theta(x) = \int p_\theta(x,z)dz$
3. **Normalizing Flow Models**: $p_\theta(x) = p_z(f^{-1}_\theta(x)) \left|\det\left(\frac{\partial f^{-1}_\theta(x)}{\partial x}\right)\right|$

## The Likelihood-Based Training Paradigm

The key thing is we always try to assign some probability assigned by the model to a data point. All the above families are trained by minimizing KL divergence $D_{KL}(p_{data} || p_\theta)$, or equivalently maximizing likelihoods (or approximations). In these techniques, the machinery involves setting up models such that we can evaluate likelihoods (or approximations) pretty efficiently.

**Mathematical Foundation:**

$$\arg\min_\theta D_{KL}(p_{data} || p_\theta) = \arg\max_\theta \mathbb{E}_{x \sim p_{data}}[\log p_\theta(x)]$$

This equivalence shows that minimizing KL divergence is equivalent to maximizing the expected log-likelihood of the data.

## Alternative Similarity Measures

However, the training objective of maximizing likelihoods is not the only way to measure similarity between $p_{data}$ and $p_\theta$. There are other ways to measure the notion of similarity:

1. **Wasserstein Distance**: Measures the minimum "cost" of transporting mass from one distribution to another
2. **Maximum Mean Discrepancy (MMD)**: Compares distributions using kernel methods
3. **Adversarial Training**: Uses a discriminator to distinguish between real and generated samples
4. **Energy-Based Models**: Learn an energy function that assigns low energy to real data and high energy to fake data

## The Likelihood vs. Sample Quality Dilemma

**The Problem:**
It is possible that models with high likelihood could be bad at sample generation and vice versa. This creates a fundamental tension in generative modeling.

### Why This Disconnect Occurs

**1. Likelihood Measures Average Performance:**

$$\mathbb{E}_{x \sim p_{data}}[\log p_\theta(x)] = \int p_{data}(x) \log p_\theta(x) dx$$

This measures how well the model assigns probability to the average data point, not necessarily how well it captures the fine-grained structure needed for high-quality generation.

**2. Sample Quality Requires Fine-Grained Structure:**
High-quality generation requires the model to capture:
- Sharp boundaries between different modes
- Fine details and textures
- Proper spatial relationships
- Realistic variations within modes

**3. Different Optimization Objectives:**
- **Likelihood**: Optimizes for probability assignment over the entire distribution
- **Sample Quality**: Requires optimization of perceptual and structural properties

### Implications for Model Design

**The Training Objective Trade-off:**
Although training objective of maximizing likelihoods could be a good one, it might not be the best one if the objective is to generate the best samples.

**This suggests that it might be useful to disentangle likelihoods and sample quality.**

## Alternative Training Paradigms

**1. Adversarial Training (GANs):**
- **Objective**: Direct optimization of sample quality through adversarial training
- **Advantage**: Can generate very high-quality samples
- **Disadvantage**: No explicit likelihood computation, training instability

**2. Hybrid Approaches:**
- **VAE-GAN**: Combines likelihood-based training with adversarial training
- **Flow-GAN**: Combines normalizing flows with adversarial training
- **Objective**: Balance between likelihood and sample quality

**3. Perceptual Losses:**
- **Objective**: Use pre-trained networks to measure perceptual similarity
- **Advantage**: Better alignment with human perception
- **Example**: LPIPS (Learned Perceptual Image Patch Similarity)

**4. Multi-Objective Training:**
- **Objective**: Combine multiple loss functions
- **Example**: $\mathcal{L} = \mathcal{L}_{likelihood} + \lambda \mathcal{L}_{perceptual} + \mu \mathcal{L}_{adversarial}$

## Conclusion

The tension between likelihood and sample quality is a fundamental challenge in generative modeling. While likelihood-based training provides a principled framework, it may not always lead to the best sample quality. This motivates the exploration of alternative training paradigms and evaluation metrics that better align with the ultimate goal of generating high-quality, diverse samples.

The key insight is that **generative modeling is not just about fitting a distribution to data, but about creating models that can generate samples that are both high-quality and diverse**. This requires careful consideration of both the training objective and the evaluation metrics used to assess model performance.

