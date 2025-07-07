# Generative Adversarial Networks

## Introduction: GANs as a Paradigm Shift

GANs are unique from all the other model families that we have seen so far, such as autoregressive models, VAEs, and normalizing flow models, because we do not train them using maximum likelihood.

### The Traditional Likelihood-Based Paradigm

All the generative models we've explored so far follow a similar training paradigm:

1. **Autoregressive Models**: Maximize $\log p_\theta(x) = \sum_{i=1}^N \log p_\theta(x_i|x_{<i})$
2. **Variational Autoencoders**: Maximize the ELBO $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$
3. **Normalizing Flow Models**: Maximize $\log p_\theta(x) = \log p_z(f^{-1}_\theta(x)) + \log |\det(\frac{\partial f^{-1}_\theta(x)}{\partial x})|$

**Common Theme**: All these models are trained by maximizing some form of likelihood or likelihood approximation.

### GANs: A Different Approach

**GANs break away from this paradigm entirely.** Instead of maximizing likelihood, GANs use **adversarial training** - a fundamentally different approach to generative modeling. We'll get to what a Generator and a Discriminator are in a bit but here is a quick table showing how GAN is different.

**Key Differences:**

| Aspect | Likelihood-Based Models | GANs |
|--------|------------------------|------|
| **Training Objective** | Maximize likelihood/ELBO | Minimax game between generator and discriminator |
| **Loss Function** | $\mathcal{L} = -\log p_\theta(x)$ | $\mathcal{L}_G = -\log D(G(z))$, $\mathcal{L}_D = -\log D(x) - \log(1-D(G(z)))$ |
| **Model Evaluation** | Direct likelihood computation | No explicit likelihood computation |
| **Training Stability** | Generally stable | Can be unstable, requires careful tuning |
| **Sample Quality** | May produce blurry samples | Often produces sharp, realistic samples |

## Likelihood-Free Learning

**Why not use maximum likelihood?** In fact, it is not so clear that better likelihood numbers necessarily correspond to higher sample quality. We know that the optimal generative model will give us the best sample quality and highest test log-likelihood. However, models with high test log-likelihoods can still yield poor samples, and vice versa.

### The Likelihood vs. Sample Quality Disconnect

To see why, consider pathological cases in which our model is comprised almost entirely of noise, or our model simply memorizes the training set:

1. **Noise Model**: A model that outputs pure noise might assign some probability to real data points, leading to a non-zero (though poor) likelihood, but produces completely useless samples.

2. **Memorization Model**: A model that perfectly memorizes the training set will have very high likelihood on training data but will only reproduce exact training examples, lacking generalization and diversity.

Therefore, we turn to **likelihood-free training** with the hope that optimizing a different objective will allow us to disentangle our desiderata of obtaining high likelihoods as well as high-quality samples.

### The Two-Sample Test Framework

Recall that maximum likelihood required us to evaluate the likelihood of the data under our model $p_\theta$. A natural way to set up a likelihood-free objective is to consider the **two-sample test**, a statistical test that determines whether or not a finite set of samples from two distributions are from the same distribution using only samples from $P$ and $Q$.

**Concretely**, given $S_1 = \{x \sim P\}$ and $S_2 = \{x \sim Q\}$, we compute a test statistic $T$ according to the difference in $S_1$ and $S_2$ that, when less than a threshold $\alpha$, accepts the null hypothesis that $P = Q$.

### Application to Generative Modeling

Analogously, we have in our generative modeling setup access to our training set $S_1 = \{x \sim p_{data}\}$ and $S_2 = \{x \sim p_\theta\}$. The key idea is to train the model to minimize a two-sample test objective between $S_1$ and $S_2$.

**However**, this objective becomes extremely difficult to work with in high dimensions, so we choose to optimize a **surrogate objective** that instead maximizes some distance between $S_1$ and $S_2$.

### Why this approach makes sense

**1. Avoiding pathological cases:**
- The two-sample test framework naturally avoids the noise and memorization problems
- It forces the model to learn the true underlying distribution structure

**3. Flexibility:**
- We can choose different distance metrics or test statistics
- This allows us to focus on different aspects of sample quality

**Key Insight:** GANs implement likelihood-free learning by using a neural network (the discriminator) to learn an optimal test statistic for distinguishing between real and generated data, and then training the generator to minimize this learned distance.

## GAN Objective

We thus arrive at the generative adversarial network formulation. There are two components in a GAN: (1) a generator and (2) a discriminator. The generator $G_\theta$ is a directed latent variable model that deterministically generates samples $x$ from $z$, and the discriminator $D_\phi$ is a function whose job is to distinguish samples from the real dataset and the generator.

### Components

- **Generator $G_\theta$**: A neural network that transforms noise $z \sim p(z)$ to samples $G_\theta(z)$
- **Discriminator $D_\phi$**: A neural network that outputs a probability $D_\phi(x) \in [0,1]$ indicating whether $x$ is real or generated

![GAN graphical model](gan.png)

### The Minimax Game

The generator and discriminator both play a two player minimax game, where:
- **Generator**: Minimizes a two-sample test objective ($p_{data} = p_\theta$)
- **Discriminator**: Maximizes the objective ($p_{data} \neq p_\theta$)

Intuitively, the generator tries to fool the discriminator to the best of its ability by generating samples that look indistinguishable from $p_{data}$.

### Formal Objective

The GAN objective can be written as:

$$\min_\theta \max_\phi V(G_\theta, D_\phi) = \mathbb{E}_{x \sim p_{data}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)}[\log(1-D_\phi(G_\theta(z)))]$$

### Understanding the Objective

Let's unpack this expression:

**For the Discriminator (maximizing with respect to $\phi$):**
- Given a fixed generator $G_\theta$, the discriminator performs binary classification
- It tries to assign probability 1 to data points from the training set $x \sim p_{data}$
- It tries to assign probability 0 to generated samples $x \sim p_G$

**For the Generator (minimizing with respect to $\theta$):**
- Given a fixed discriminator $D_\phi$, the generator tries to maximize $D_\phi(G_\theta(z))$
- This is equivalent to minimizing $\log(1-D_\phi(G_\theta(z)))$

### Optimal Discriminator

In this setup, the optimal discriminator is:

$$D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}$$

**Derivation:**
The discriminator's objective is to maximize:

$$\mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{x \sim p_G}[\log(1-D(x))]$$

This is maximized when:

$$D(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}$$

On the other hand, the generator minimizes this objective for a fixed discriminator $D_\phi$.

### Connection to Jensen-Shannon Divergence

After performing some algebra, plugging in the optimal discriminator $D^*_G(\cdot)$ into the overall objective $V(G_\theta, D^*_G(x))$ gives us:

$$2D_{JSD}[p_{data}, p_G] - \log 4$$

The $D_{JSD}$ term is the **Jensen-Shannon Divergence**, which is also known as the symmetric form of the KL divergence:

$$D_{JSD}[p,q] = \frac{1}{2}\left(D_{KL}\left[p, \frac{p+q}{2}\right] + D_{KL}\left[q, \frac{p+q}{2}\right]\right)$$

### Properties of JSD

The JSD satisfies all properties of the KL divergence, and has the additional perk that $D_{JSD}[p,q] = D_{JSD}[q,p]$ (symmetry).

**Key Properties:**

1. **Non-negative**: $D_{JSD}[p,q] \geq 0$

2. **Symmetric**: $D_{JSD}[p,q] = D_{JSD}[q,p]$

3. **Zero iff equal**: $D_{JSD}[p,q] = 0$ if and only if $p = q$

4. **Bounded**: $D_{JSD}[p,q] \leq \log 2$

### Optimal Solution

With this distance metric, the optimal generator for the GAN objective becomes $p_G = p_{data}$, and the optimal objective value that we can achieve with optimal generators and discriminators $G^*(\cdot)$ and $D^*_{G^*}(x)$ is $-\log 4$.

**Why $-\log 4$?**
- When $p_G = p_{data}$, we have $D_{JSD}[p_{data}, p_G] = 0$
- Therefore, $V(G^*, D^*) = 2 \cdot 0 - \log 4 = -\log 4$

## GAN Training Algorithm

Thus, the way in which we train a GAN is as follows:

**For epochs $1, \ldots, N$ do:**

1. **Sample minibatch of size $m$ from data**: $x^{(1)}, \ldots, x^{(m)} \sim p_{data}$
2. **Sample minibatch of size $m$ of noise**: $z^{(1)}, \ldots, z^{(m)} \sim p_z$
3. **Take a gradient descent step on the generator parameters $\theta$**:

$$\nabla_\theta V(G_\theta, D_\phi) = \frac{1}{m}\nabla_\theta \sum_{i=1}^m \log(1-D_\phi(G_\theta(z^{(i)})))$$

4. **Take a gradient ascent step on the discriminator parameters $\phi$**:

$$\nabla_\phi V(G_\theta, D_\phi) = \frac{1}{m}\nabla_\phi \sum_{i=1}^m [\log D_\phi(x^{(i)}) + \log(1-D_\phi(G_\theta(z^{(i)})))]$$

**Key Points:**

1. **Alternating Updates**: We update the generator and discriminator in alternating steps
2. **Minibatch Training**: We use minibatches of both real data and noise samples
3. **Generator Update**: Minimizes the probability that the discriminator correctly identifies generated samples
4. **Discriminator Update**: Maximizes the probability of correctly classifying real vs. generated samples

**Practical Considerations:**

- **Learning Rate Balance**: The learning rates for generator and discriminator must be carefully balanced
- **Update Frequency**: Often the discriminator is updated multiple times per generator update
- **Convergence Monitoring**: Training progress is monitored through discriminator accuracy and sample quality
- **Early Stopping**: Training may be stopped when the discriminator can no longer distinguish real from fake

This formulation shows that GANs are essentially implementing an adaptive two-sample test, where the discriminator learns the optimal way to distinguish between real and generated data, and the generator learns to minimize this learned distance.

## Challenges

Although GANs have been successfully applied to several domains and tasks, working with them in practice is challenging because of their: (1) unstable optimization procedure, (2) potential for mode collapse, (3) difficulty in evaluation.

**1. Unstable Optimization Procedure**

During optimization, the generator and discriminator loss often continue to oscillate without converging to a clear stopping point. Due to the lack of a robust stopping criteria, it is difficult to know when exactly the GAN has finished training.

**Causes of Instability:**
- **Minimax Nature**: The adversarial game creates competing objectives
- **Gradient Issues**: Vanishing/exploding gradients can occur
- **Learning Rate Sensitivity**: Small changes in learning rates can cause divergence
- **Network Capacity Imbalance**: If one network becomes too powerful, training collapses

**Symptoms:**
- Oscillating loss curves
- No clear convergence pattern
- Sudden collapse of training
- Generator or discriminator loss going to zero/infinity

**2. Mode Collapse**

The generator of a GAN can often get stuck producing one of a few types of samples over and over again (mode collapse). This occurs when the generator finds a few "safe" modes that consistently fool the discriminator and stops exploring the full data distribution.

**What is Mode Collapse:**
- **Definition**: Generator only produces samples from a subset of the true distribution modes
- **Example**: In image generation, only producing images of one type (e.g., only front-facing faces)
- **Problem**: Lack of diversity in generated samples

**Causes:**
- **Discriminator Overfitting**: Discriminator becomes too good at detecting certain types of fake samples
- **Generator Optimization**: Generator finds local optima that work well against current discriminator
- **Training Imbalance**: One network becomes too powerful relative to the other

**3. Difficulty in Evaluation**

Unlike likelihood-based models, GANs don't provide explicit likelihood values, making evaluation challenging.

**Evaluation Challenges:**
- **No Likelihood**: Can't use traditional metrics like log-likelihood
- **Subjective Quality**: Sample quality is often subjective and domain-specific
- **Diversity vs. Quality Trade-off**: Hard to balance sample quality with diversity
- **Mode Coverage**: Difficult to measure if all modes of the data distribution are captured

**Addressing the Challenges:**

Most fixes to these challenges are empirically driven, and there has been a significant amount of work put into developing new architectures, regularization schemes, and noise perturbations in an attempt to circumvent these issues.

## Selected GANs
Next, we focus our attention to a few select types of GAN architectures and explore them in more detail.

### f-GAN

The f-GAN optimizes the variant of the two-sample test objective that we have discussed so far, but using a very general notion of distance: the **f-divergence**. Given two densities $p$ and $q$, the f-divergence can be written as:

$$D_f(p,q) = \sup_{T} \left(\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x_{fake} \sim q}[f^*(T(x_{fake}))]\right)$$

where $f$ is any convex, lower-semicontinuous function with $f(1) = 0$. Several of the distance "metrics" that we have seen so far fall under the class of f-divergences, such as KL and Jensen-Shannon.

#### Understanding the Requirements

**What is a convex function?**
A function $f$ is convex if for any two points $x, y$ and any $\lambda \in [0,1]$, we have:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

This means that the line segment between any two points on the function lies above or on the function itself. 

**Understanding the Line Segment Property:**

Let's break down what this means geometrically:

**Two Points**: Consider any two points $(x, f(x))$ and $(y, f(y))$ on the graph of the function $f$

**Line Segment**: The line segment connecting these points consists of all points of the form:

$$(\lambda x + (1-\lambda)y, \lambda f(x) + (1-\lambda)f(y))$$

where $\lambda \in [0,1]$

**Function Value**: At the same $x$-coordinate $\lambda x + (1-\lambda)y$, the function value is:
   
$$f(\lambda x + (1-\lambda)y)$$

**Convexity Condition**: The inequality $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ means that the function value at any point on the line segment is less than or equal to the corresponding point on the straight line connecting $(x, f(x))$ and $(y, f(y))$

**Visual Interpretation:**
If you draw a straight line between any two points on a convex function's graph. The entire function between those points must lie on or below that straight line.

Convex functions have important properties:

- **Single minimum**: If a minimum exists, it's global

- **Well-behaved gradients**: Useful for optimization

- **Jensen's inequality**: $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$ for convex $f$

**What is a lower-semicontinuous function?**
A function $f$ is lower-semicontinuous at a point $x_0$ if:

$$\liminf_{x \to x_0} f(x) \geq f(x_0)$$

**Understanding $\liminf$ and the Infimum:**

The notation $\liminf_{x \to x_0} f(x)$ involves two concepts:

1. **Infimum (inf)**: The infimum of a set is the greatest lower bound. For a set $S$, $\inf S$ is the largest number that is less than or equal to all elements in $S$.

2. **Limit Inferior**: $\liminf_{x \to x_0} f(x)$ is the infimum of all limit points of $f(x)$ as $x$ approaches $x_0$.

**How $\liminf$ works:**

Consider all sequences $\{x_n\}$ that converge to $x_0$. For each sequence, we look at the limit of $f(x_n)$ (if it exists). The $\liminf$ is the infimum of all these possible limit values.

**Mathematical Definition:**

$$\liminf_{x \to x_0} f(x) = \inf \left\{ \lim_{n \to \infty} f(x_n) : x_n \to x_0 \text{ and } \lim_{n \to \infty} f(x_n) \text{ exists} \right\}$$

**Why consider Sequences even for Continuous Functions?**

You might wonder: "If $f$ is continuous, why do we need sequences? Can't we just use the regular limit?"

**Key Insight:** Lower-semicontinuity is a **weaker condition** than continuity. A function can be lower-semicontinuous without being continuous.

**The Relationship:**

1. **Continuous functions** are always lower-semicontinuous
2. **Lower-semicontinuous functions** may have discontinuities (but only "jumps up")

**Example of Lower-Semicontinuous but NOT Continuous:**
Consider $f(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \geq 0 \end{cases}$. This function is lower-semicontinuous at $x = 0$ (no "jump down"). But it's NOT continuous at $x = 0$ (there's a "jump up")

**Example of NOT Lower-Semicontinuous:**
Consider $f(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x = 0 \\ 0 & \text{if } x > 0 \end{cases}$ at $x_0 = 0$:

$\liminf_{x \to 0^-} f(x) = 0$ (approaching from left)

$\liminf_{x \to 0^+} f(x) = 0$ (approaching from right)

$\liminf_{x \to 0} f(x) = 0$ (overall limit inferior)

Since $f(0) = 1$ and $0 \not\geq 1$, this function is NOT lower-semicontinuous at $x = 0$

**Why the Sequence definition is uiversal:**

The sequence-based definition works for ALL functions, whether they're:
- Continuous everywhere
- Lower-semicontinuous but not continuous
- Neither continuous nor lower-semicontinuous

**For Continuous Functions:**
If $f$ is continuous at $x_0$, then:

$$\liminf_{x \to x_0} f(x) = \lim_{x \to x_0} f(x) = f(x_0)$$

So the sequence definition "reduces" to the regular limit, but it's still the same mathematical concept.

**Why this matters for f-Divergences:**
The f-divergence framework needs to work with functions that might not be continuous everywhere, so we need the more general sequence-based definition.

**Why do we need $f(1) = 0$?**
This requirement ensures that the f-divergence has the correct properties for a distance measure:

**Zero when distributions are equal**: When $p = q$, we have $\frac{p(x)}{q(x)} = 1$ everywhere, so:
   
$$D_f(p,p) = \mathbb{E}_{x \sim p}[f(1)] = \mathbb{E}_{x \sim p}[0] = 0$$

**Distance-like behavior**: This property ensures that the f-divergence behaves like a proper distance measure, being zero only when the distributions are identical

**Example**: For KL divergence, $f(u) = u \log u$ satisfies $f(1) = 1 \cdot \log 1 = 0$, making it a valid choice for an f-divergence.

**Important Clarification: KL Divergence and Convexity**

You might be wondering: "But $\log u$ is concave, so how can KL divergence be an f-divergence?" This is a great observation! The key is that the $f$ function for KL divergence is $f(u) = u \log u$, not just $\log u$.

**The KL Divergence Formula:**
The KL divergence between distributions $p$ and $q$ is:

$$D_{KL}(p||q) = \mathbb{E}_{x \sim p}\left[\log\frac{p(x)}{q(x)}\right] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)}\log\frac{p(x)}{q(x)}\right]$$

Notice that the second form matches the f-divergence formula with $f(u) = u \log u$.

#### f-Divergence Examples

**Common f-divergences:**

1. **KL Divergence**: $f(u) = u \log u$
2. **Reverse KL**: $f(u) = -\log u$
3. **Jensen-Shannon**: $f(u) = u \log u - (u+1) \log \frac{u+1}{2}$
4. **Total Variation**: $f(u) = \frac{1}{2}|u-1|$
5. **Pearson χ²**: $f(u) = (u-1)^2$

#### Setting up the f-GAN Objective

To set up the f-GAN objective, we borrow two commonly used tools from convex optimization: the **Fenchel conjugate** and **duality**. Specifically, we obtain a lower bound to any f-divergence via its Fenchel conjugate:

$$D_f(p,q) \geq \sup_{T \in \mathcal{T}} \left(\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x_{fake} \sim q}[f^*(T(x_{fake}))]\right)$$

Where $f^*$ is the Fenchel conjugate of $f$:

$$f^*(t) = \sup_{u \in \text{dom}(f)} (tu - f(u))$$

**What is the Fenchel Conjugate?**

The Fenchel conjugate $f^*$ of a function $f$ is defined as:

$$f^*(t) = \sup_{u \in \text{dom}(f)} (tu - f(u))$$

**Detailed Geometric Interpretation:**

The Fenchel conjugate $f^*(t)$ has a beautiful geometric interpretation that helps us understand what it represents. For each value of $t$, we consider the family of lines $y = tu + c$ with slope $t$. We look for the line $y = tu + c$ that touches the graph of $f(u)$ from below and has the largest $y$-intercept $c$. The Fenchel conjugate $f^*(t)$ is the negative of this $y$-intercept: $f^*(t) = -c$


**Economic Intuition for the Fenchel Conjugate:**

One of the most intuitive ways to understand the convex conjugate function $f^*(t)$ is through an economic lens. Imagine $f(u)$ as the cost function representing the total expense incurred to produce a quantity $u$ of a certain product. The variable $y$ corresponds to the market price per unit of that product.

In this context, the product $xy$ represents the revenue generated by selling $x$ units at price $t$. The term $f(u)$, as mentioned, is the cost of producing those units. Therefore, the expression $tu - f(u)$ represents the profit earned by producing and selling $u$ units at price $t$.

The convex conjugate $f^*(t)$ is defined as the supremum (or maximum) of this profit over all possible production quantities $u$:

$$f^*(t) = \sup_u (tu - f(u))$$

Thus, $f^*(t)$ gives the optimal profit achievable at the market price $t$, assuming the producer chooses the best production quantity $u$ to maximize profit.

**Geometric Interpretation in Economics:**

Now, consider the graph of the cost function $f(u)$. Assume $f$ is convex, continuous, and differentiable, which is a reasonable assumption for many cost functions in economics.

The slope of the cost curve at any point $u$ is given by the derivative $f'(u)$. This derivative represents the marginal cost — the additional cost to produce one more unit at quantity $u$.

The condition for optimal production quantity $u$ at price $t$ arises from maximizing profit:

$$\max_u \{tu - f(u)\}$$

Taking the derivative with respect to $u$ and setting it to zero for an optimum:

$$t - f'(u) = 0 \implies t = f'(u)$$

This means the optimal production quantity $u$ is found where the price $t$ equals the marginal cost $f'(u)$.

Geometrically, this corresponds to finding a tangent line to the graph of $f(u)$ that has slope $t$. Using a ruler, you can "slide" the line around until it just touches the cost curve without crossing it. The point of tangency corresponds to the optimal $u$.

Importantly, the vertical intercept of this tangent line relates directly to the optimal profit. The tangent line can be expressed as:

$$\ell(u) = f(u_0) + f'(u_0)(u - u_0)$$

At $u = 0$, the intercept is:

$$\ell(0) = f(u_0) - u_0 f'(u_0)$$

Notice that:

$$-(u_0 t - f(u_0)) = f(u_0) - u_0 t$$

Since $t = f'(u_0)$, the intercept equals the negative of the optimal profit. Therefore, the intercept of the tangent line with slope $t$ gives $-f^*(t)$.

**Key Properties:**
1. **Convexity**: If $f$ is convex, then $f^*$ is also convex
2. **Duality**: $(f^*)^* = f$ (the conjugate of the conjugate is the original function)
3. **Domain**: The domain of $f^*$ depends on the behavior of $f$

**Examples of Fenchel Conjugates:**

**Example 1: KL Divergence**

For $f(u) = u \log u$:

$$f^*(t) = \sup_{u > 0} (tu - u \log u)$$

To find this, we set the derivative to zero:

$$\frac{d}{du}(tu - u \log u) = t - \log u - 1 = 0$$

$$\log u = t - 1$$

$$u = e^{t-1}$$

Substituting back:

$$f^*(t) = te^{t-1} - e^{t-1}(t-1) = e^{t-1}$$

**Example 2: Reverse KL**

For $f(u) = -\log u$:

$$f^*(t) = \sup_{u > 0} (tu + \log u)$$

Setting derivative to zero:

$$\frac{d}{du}(tu + \log u) = t + \frac{1}{u} = 0$$

$$u = -\frac{1}{t}$$

Substituting back:

$$f^*(t) = t(-\frac{1}{t}) + \log(-\frac{1}{t}) = -1 + \log(-\frac{1}{t}) = -1 - \log(-t)$$

**Example 3: Total Variation**

For $f(u) = \frac{1}{2}|u-1|$:

$$f^*(t) = \sup_{u} (tu - \frac{1}{2}|u-1|)$$

This gives:

$$f^*(t) = \begin{cases} t & \text{if } |t| \leq \frac{1}{2} \\ +\infty & \text{otherwise} \end{cases}$$

**The Duality Principle:**

The Fenchel conjugate provides a way to transform optimization problems. The key insight is that:

**Primal Problem**: $\inf_{u} f(u)$

**Dual Problem**: $\sup_{t} -f^*(t)$

**The Primal-Dual Relationship:**

The f-divergence can be expressed in both forms:

$$D_f(p,q) = \mathbb{E}_{x \sim q}\left[f\left(\frac{p(x)}{q(x)}\right)\right] = \sup_{T} \left(\mathbb{E}_{x \sim p}[T(x)]- \mathbb{E}_{x_{fake} \sim q}[f^*(T(x_{fake}))]\right)$$

Where:

- **Primal form**: $\mathbb{E}_{x \sim q}\left[f\left(\frac{p(x)}{q(x)}\right)\right]$ (direct computation)

- **Dual form**: $\sup_{T} \left(\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x_{fake} \sim q}[f^*(T(x_{fake}))]\right)$ (optimization problem)

**Derivation: From Primal to Dual Form**

Let's walk through the step-by-step derivation of how we transform the primal form into the dual form:

**Step 1: Start with the Primal Form**

The f-divergence is defined as:

$$D_f(p,q) = \mathbb{E}_{x \sim q}\left[f\left(\frac{p(x)}{q(x)}\right)\right]$$

This is the "primal form" - it directly computes the divergence by evaluating the function $f$ at the ratio $\frac{p(x)}{q(x)}$.

**Step 2: Apply the Fenchel Conjugate Identity**

The key insight comes from the Fenchel conjugate identity. For any convex function $f$ and any point $u$, we have:

$$f(u) = \sup_{t} (tu - f^*(t))$$

This is a fundamental result in convex analysis known as the **Fenchel-Moreau theorem**. It states that a convex function can be recovered from its conjugate.

**Step 3: Substitute the Identity**

We substitute this identity into our primal form:

$$D_f(p,q) = \mathbb{E}_{x \sim q}\left[\sup_{t} \left(t \cdot \frac{p(x)}{q(x)} - f^*(t)\right)\right]$$

**Step 4: Exchange Supremum and Expectation**

This is the crucial step. We can exchange the supremum and expectation under certain conditions (satisfied for convex $f$):

$$D_f(p,q) = \sup_{T} \mathbb{E}_{x \sim q}\left[T(x) \cdot \frac{p(x)}{q(x)} - f^*(T(x))\right]$$

Here, we've replaced the variable $t$ with a function $T(x)$ that can depend on $x$.

**Step 5: Simplify the Expression**

We can rewrite the expectation:

$$\mathbb{E}_{x \sim q}\left[T(x) \cdot \frac{p(x)}{q(x)} - f^*(T(x))\right] = \mathbb{E}_{x \sim q}\left[T(x) \cdot \frac{p(x)}{q(x)}\right] - \mathbb{E}_{x \sim q}[f^*(T(x))]$$

The first term can be simplified using the definition of expectation:

$$\mathbb{E}_{x \sim q}\left[T(x) \cdot \frac{p(x)}{q(x)}\right] = \int T(x) \cdot \frac{p(x)}{q(x)} \cdot q(x) dx = \int T(x) \cdot p(x) dx = \mathbb{E}_{x \sim p}[T(x)]$$

**Step 6: Arrive at the Dual Form**

Putting it all together:

$$D_f(p,q) = \sup_{T} \left(\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x \sim q}[f^*(T(x))]\right)$$

In other words (to distinguish the two different inputs to the Discriminator):

$$D_f(p,q) = \sup_{T} \left(\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x_{fake} \sim q}[f^*(T(x_{fake}))]\right)$$

This is the **dual form** of the f-divergence.

**Why This Derivation Works:**

1. **Convexity**: The convexity of $f$ ensures that the Fenchel conjugate identity holds
2. **Exchange of Supremum and Expectation**: This is valid because we're optimizing over a convex set of functions
3. **Duality Gap**: Under certain conditions, there is no duality gap, meaning the primal and dual forms give the same value

**Key Insights from the Derivation:**

1. **From Direct Computation to Optimization**: The primal form requires direct computation of $f(\frac{p(x)}{q(x)})$, while the dual form transforms this into an optimization problem over functions $T$.

2. **Role of the Fenchel Conjugate**: The conjugate $f^*$ appears naturally in the dual form.

3. **Connection to GANs**: The dual form is perfect for GANs because we can parameterize $T$ as a neural network (the discriminator). The optimization becomes a minimax game. We can use gradient-based optimization.

**Example: KL Divergence Derivation**

Let's see this in action for KL divergence where $f(u) = u \log u$:

**Primal Form:**

$$D_{KL}(p||q) = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} \log \frac{p(x)}{q(x)}\right]$$

**Step 1:** Use the Fenchel conjugate identity for $f(u) = u \log u$

We know that $f^*(t) = e^{t-1}$ (from our earlier examples)

**Step 2:** Substitute:

$$D_{KL}(p||q) = \mathbb{E}_{x \sim q}\left[\sup_{t} \left(t \cdot \frac{p(x)}{q(x)} - e^{t-1}\right)\right]$$

**Step 3:** Exchange supremum and expectation:

$$D_{KL}(p||q) = \sup_{T} \mathbb{E}_{x \sim q}\left[T(x) \cdot \frac{p(x)}{q(x)} - e^{T(x)-1}\right]$$

**Step 4:** Simplify:

$$D_{KL}(p||q) = \sup_{T} \left(\mathbb{E}_{x \sim p}[T(x)] - \mathbb{E}_{x \sim q}[e^{T(x)-1}]\right)$$

This gives us the dual form for KL divergence, which can be used in f-GAN training.

#### Steps in f-GAN

**Step 1: Choose an f-divergence**
Select a convex function $f$ with $f(1) = 0$ (e.g., KL divergence, Jensen-Shannon, etc.)

**Step 2: Compute the Fenchel conjugate**
Find $f^*$ analytically or numerically

**Step 3: Parameterize the dual variable**
Replace $T$ with a neural network $T_\phi$ parameterized by $\phi$

**Step 4: Set up the minimax game**

$$\min_\theta \max_\phi F(\theta,\phi) = \mathbb{E}_{x \sim p_{data}}[T_\phi(x)] - \mathbb{E}_{x_{fake} \sim p_{G_\theta}}[f^*(T_\phi(x_{fake}))]$$

**Understanding the Roles:**

- **Generator ($G_\theta$)**: Tries to minimize the divergence estimate
- **Discriminator ($T_\phi$)**: Tries to tighten the lower bound by maximizing the dual objective

**Key Insight:**
The discriminator $T_\phi$ is not a binary classifier like in standard GANs, but rather a function.

**Important Distinction: Vanilla GAN vs f-GAN Generator**

There is a fundamental difference between how generators work in vanilla GANs versus f-GANs:

**Vanilla GAN Generator:**

- **Explicit generator network**: $G_\theta(z)$ where $z \sim p(z)$ (noise)

- **Direct transformation**: Noise $z$ → Generated sample $G_\theta(z)$

- **Objective**: $\min_\theta \max_\phi V(G_\theta, D_\phi) = \mathbb{E}_{x \sim p_{data}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)}[\log(1-D_\phi(G_\theta(z)))]$

**f-GAN Generator:**

- **No explicit generator network**: We work with $p_{G_\theta}$ as a distribution directly

- **Implicit generator**: The "generator" is whatever mechanism produces samples from $p_{G_\theta}$

- **Objective**: $\min_\theta \max_\phi F(\theta,\phi) = \mathbb{E}_{x \sim p_{data}}[T_\phi(x)] - \mathbb{E}_{x_{fake} \sim p_{G_\theta}}[f^*(T_\phi(x_{fake}))]$

**The Key Insight:**

In f-GAN, the "generator" is **implicit** - it's whatever mechanism produces samples from the distribution $p_{G_\theta}$. This could be:

1. **A neural network** $G_\theta$ that transforms noise (like in vanilla GANs)
2. **A flow-based model** that transforms a base distribution
3. **A VAE decoder** that generates from a latent space
4. **Any other generative model** that produces samples from $p_{G_\theta}$

**Why This Matters:**

The f-GAN framework is **more general** than vanilla GANs because:

- **Vanilla GANs**: Require a specific generator architecture $G_\theta(z)$
- **f-GANs**: Can work with any generative model that produces samples from $p_{G_\theta}$

**Practical Implementation:**

In practice, when implementing f-GAN, you would typically:

1. **Choose a generative model** (e.g., a neural network $G_\theta$)
2. **Use it to generate samples** from $p_{G_\theta}$
3. **Apply the f-GAN objective** to train both the generator and discriminator

So in the f-GAN formulation, there's no explicit $G_\theta$ network like in vanilla GANs. The "generator" is the abstract distribution $p_{G_\theta}$, and the actual implementation depends on what generative model you choose to use.

#### Advantages of f-GAN

1. **Unified Framework**: One formulation covers many different GAN variants
2. **Theoretical Rigor**: Based on well-established convex optimization theory
3. **Flexibility**: Can adapt the divergence measure to the specific problem
4. **Stability**: Some f-divergences may lead to more stable training

#### Practical Considerations

- **Choice of f**: Different f-divergences have different properties
- **Fenchel Conjugate**: Must be computable for the chosen f-divergence
- **Training**: Similar alternating optimization as standard GANs
- **Evaluation**: Same challenges as other GAN variants