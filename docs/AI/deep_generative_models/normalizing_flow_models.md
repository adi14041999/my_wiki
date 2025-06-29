# Normalizing Flow Models

So far we have learned two types of likelihood based generative models:

**Autoregressive Models**: $p_\theta(x) = \prod_{i=1}^N p_\theta(x_i|x_{<i})$

**Variational autoencoders**: $p_\theta(x) = \int p_\theta(x,z)dz$

The two methods have relative strengths and weaknesses. Autoregressive models provide tractable likelihoods but no direct mechanism for learning features, whereas variational autoencoders can learn feature representations but have intractable marginal likelihoods.

## Change of Variables Formula

In normalizing flows, we wish to map simple distributions (easy to sample and evaluate densities) to complex ones (learned via data). The change of variables formula describes how to evaluate densities of a random variable that is a deterministic transformation from another variable.

Let's start with the univariate case and then generalize to multivariate random variables.

### Univariate Case

Consider two random variables $Z$ and $X$ related by a strictly monotonic function $f: \mathbb{R} \rightarrow \mathbb{R}$ such that $X = f(Z)$. We want to find the probability density function of $X$ in terms of the density of $Z$.

The key insight comes from the fact that probabilities must be preserved under the transformation. For any interval $[a, b]$ in the $X$ space:

$$P(a \leq X \leq b) = P(f^{-1}(a) \leq Z \leq f^{-1}(b))$$

This can be written as:

$$\int_a^b p_X(x) dx = \int_{f^{-1}(a)}^{f^{-1}(b)} p_Z(z) dz$$

To perform the substitution $z = f^{-1}(x)$, we need to express $dz$ in terms of $dx$. Since $z = f^{-1}(x)$, we can use the chain rule to find:

$$\frac{dz}{dx} = \frac{d}{dx}f^{-1}(x) = \frac{1}{f'(f^{-1}(x))}$$

This follows from the inverse function theorem: if $y = f(x)$, then $\frac{dx}{dy} = \frac{1}{f'(x)}$.

Therefore, $dz = \frac{1}{f'(f^{-1}(x))} dx$. However, we need to take the absolute value because probability densities must be non-negative. If $f'(f^{-1}(x)) < 0$ (meaning $f$ is decreasing), then $\frac{1}{f'(f^{-1}(x))} < 0$, which would make the density negative. Therefore, we use:

$$dz = \frac{1}{|f'(f^{-1}(x))|} dx$$

Substituting this into our integral:

$$\int_a^b p_X(x) dx = \int_{f^{-1}(a)}^{f^{-1}(b)} p_Z(z) dz = \int_a^b p_Z(f^{-1}(x)) \cdot \frac{1}{|f'(f^{-1}(x))|} dx$$

Since this equality must hold for all intervals $[a, b]$, the integrands must be equal:

$$p_X(x) = p_Z(f^{-1}(x)) \cdot \frac{1}{|f'(f^{-1}(x))|}$$

This is the univariate change of variables formula. The factor $\frac{1}{|f'(f^{-1}(x))|}$ accounts for how the transformation stretches or compresses the probability mass.

**Why should $f$ be monotonic?** The monotonicity requirement ensures that $f$ is invertible (one-to-one), which is crucial for the change of variables formula to work correctly. If $f$ were not monotonic, there could be multiple values of $z$ that map to the same value of $x$, making the inverse function $f^{-1}$ ill-defined. This would violate the fundamental assumption that we can uniquely determine the original variable $z$ from the transformed variable $x$.

For example, if $f(z) = z^2$ (which is not monotonic on $\mathbb{R}$), then both $z = 2$ and $z = -2$ map to $x = 4$. This creates ambiguity in the inverse mapping and would require special handling to account for multiple pre-images.

### Multivariate Case

For the multivariate case, we have random variables $\mathbf{Z}$ and $\mathbf{X}$ related by a bijective function $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$ such that $\mathbf{X} = f(\mathbf{Z})$.

The key insight is that the probability mass in any region must be preserved under the transformation. For any region $A$ in the $\mathbf{X}$ space:

$$P(\mathbf{X} \in A) = P(\mathbf{Z} \in f^{-1}(A))$$

This can be written as:

$$\int_A p_X(\mathbf{x}) d\mathbf{x} = \int_{f^{-1}(A)} p_Z(\mathbf{z}) d\mathbf{z}$$

To perform the multivariate substitution $\mathbf{z} = f^{-1}(\mathbf{x})$, we need to understand how the volume element $d\mathbf{z}$ transforms. The Jacobian matrix $\frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}}$ is an $n \times n$ matrix where:

$$\left[\frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}}\right]_{ij} = \frac{\partial f^{-1}_i(\mathbf{x})}{\partial x_j}$$

This matrix describes how small changes in $\mathbf{x}$ correspond to changes in $\mathbf{z}$. In multivariate calculus, when we perform a change of variables $\mathbf{z} = f^{-1}(\mathbf{x})$, the volume element transforms as:

$$d\mathbf{z} = \left|\det\left(\frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}}\right)\right| d\mathbf{x}$$

This is the multivariate generalization of the univariate substitution $dz = \frac{1}{|f'(f^{-1}(x))|} dx$. The determinant of the Jacobian matrix measures how the transformation affects the volume of a small region:
- If $|\det(J)| > 1$, the transformation expands volume
- If $|\det(J)| < 1$, the transformation contracts volume  
- If $|\det(J)| = 1$, the transformation preserves volume

Substituting this into our integral:

$$\int_A p_X(\mathbf{x}) d\mathbf{x} = \int_{f^{-1}(A)} p_Z(\mathbf{z}) d\mathbf{z} = \int_A p_Z(f^{-1}(\mathbf{x})) \left|\det\left(\frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}}\right)\right| d\mathbf{x}$$

Since this equality must hold for all regions $A$, the integrands must be equal:

$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \left|\det\left(\frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}}\right)\right|$$

This is the multivariate change of variables formula. The determinant of the Jacobian matrix accounts for how the transformation affects the volume of probability mass.

**Alternative Form Using Forward Mapping:**
Using the property that $\det(A^{-1}) = \det(A)^{-1}$ for any invertible matrix $A$, we can rewrite this as:

$$p_X(\mathbf{x}) = p_Z(\mathbf{z}) \left|\det\left(\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\right)\right|^{-1}$$

This form is often more convenient in practice because it uses the forward mapping $f$ rather than the inverse mapping $f^{-1}$.

**Final result**: Let $Z$ and $X$ be random variables which are related by a mapping $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$ such that $X = f(Z)$ and $Z = f^{-1}(X)$. Then

$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \left|\det\left(\frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}}\right)\right|$$

There are several things to note here:

- $\mathbf{x}$ and $\mathbf{z}$ need to be continuous and have the same dimension.
- $\frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}}$ is a matrix of dimension $n \times n$, where each entry at location $(i,j)$ is defined as $\frac{\partial f^{-1}(\mathbf{x})_i}{\partial x_j}$. This matrix is also known as the Jacobian matrix.
- $\det(A)$ denotes the determinant of a square matrix $A$.

For any invertible matrix $A$, $\det(A^{-1}) = \det(A)^{-1}$, so for $\mathbf{z} = f^{-1}(\mathbf{x})$ we have

$$p_X(\mathbf{x}) = p_Z(\mathbf{z}) \left|\det\left(\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\right)\right|^{-1}$$

If $\left|\det\left(\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\right)\right| = 1$, then the mapping is volume preserving, which means that the transformed distribution $p_X$ will have the same "volume" compared to the original one $p_Z$.

## Normalizing Flow Models deep dive

Let us consider a directed, latent-variable model over observed variables $X$ and latent variables $Z$. In a normalizing flow model, the mapping between $Z$ and $X$, given by $f_\theta: \mathbb{R}^n \rightarrow \mathbb{R}^n$, is deterministic and invertible such that $X = f_\theta(Z)$ and $Z = f^{-1}_\theta(X)$.

Using change of variables, the marginal likelihood $p(x)$ is given by

$$p_X(\mathbf{x}; \theta) = p_Z(f^{-1}_\theta(\mathbf{x})) \left|\det\left(\frac{\partial f^{-1}_\theta(\mathbf{x})}{\partial \mathbf{x}}\right)\right|$$

The name "normalizing flow" can be interpreted as the following:

- **"Normalizing"** means that the change of variables gives a normalized density after applying an invertible transformation. When we transform a random variable through an invertible function, the resulting density automatically integrates to 1 (is normalized) because the change of variables formula preserves the total probability mass. This is different from other methods where we might need to explicitly normalize or approximate the density.

- **"Flow"** means that the invertible transformations can be composed with each other to create more complex invertible transformations. If we have two invertible functions $f_1$ and $f_2$, then their composition $f_2 \circ f_1$ is also invertible. This allows us to build complex transformations by chaining simpler ones, creating a "flow" of transformations.

Different from autoregressive models and variational autoencoders, deep normalizing flow models require specific architectural structures:

1. **The input and output dimensions must be the same** - This is necessary for the transformation to be invertible. If the dimensions don't match, we can't uniquely map back and forth between the spaces.

2. **The transformation must be invertible** - This is fundamental to the change of variables formula and allows us to compute both the forward transformation (for sampling) and the inverse transformation (for density evaluation).

3. **Computing the determinant of the Jacobian needs to be efficient (and differentiable)** - The change of variables formula requires computing the determinant of the Jacobian matrix. For high-dimensional spaces, this can be computationally expensive, so we need architectures that make this computation tractable.

Next, we introduce several popular forms of flow models that satisfy these properties.

### Planar Flow

The Planar Flow introduces the following invertible transformation:

$$\mathbf{x} = f_\theta(\mathbf{z}) = \mathbf{z} + \mathbf{u}h(\mathbf{w}^\top\mathbf{z} + b)$$

where $\mathbf{u}, \mathbf{w}, b$ are parameters.

The absolute value of the determinant of the Jacobian is given by:

$$\left|\det\left(\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\right)\right| = |1 + h'(\mathbf{w}^\top\mathbf{z} + b)\mathbf{u}^\top\mathbf{w}|$$

However, $\mathbf{u}, \mathbf{w}, b, h(\cdot)$ need to be restricted in order to be invertible. For example, $h = \tanh$ and $h'(\mathbf{w}^\top\mathbf{z} + b)\mathbf{u}^\top\mathbf{w} \geq -1$. Note that while $f_\theta(\mathbf{z})$ is invertible, computing $f^{-1}_\theta(\mathbf{z})$ could be difficult analytically. The following models address this problem, where both $f_\theta$ and $f^{-1}_\theta$ have simple analytical forms.

### NICE and RealNVP

The Nonlinear Independent Components Estimation (NICE) model and Real Non-Volume Preserving (RealNVP) model compose two kinds of invertible transformations: additive coupling layers and rescaling layers. The coupling layer in NICE partitions a variable $\mathbf{z}$ into two disjoint subsets, say $\mathbf{z}_1$ and $\mathbf{z}_2$. Then it applies the following transformation:

**Forward mapping $\mathbf{z} \rightarrow \mathbf{x}$:**

- $\mathbf{x}_1 = \mathbf{z}_1$, which is an identity mapping.
- $\mathbf{x}_2 = \mathbf{z}_2 + m_\theta(\mathbf{z}_1)$, where $m_\theta$ is a neural network.

**Inverse mapping $\mathbf{x} \rightarrow \mathbf{z}$:**

- $\mathbf{z}_1 = \mathbf{x}_1$, which is an identity mapping.
- $\mathbf{z}_2 = \mathbf{x}_2 - m_\theta(\mathbf{x}_1)$, which is the inverse of the forward transformation.

Therefore, the Jacobian of the forward mapping is lower triangular, whose determinant is simply the product of the elements on the diagonal, which is 1. Therefore, this defines a volume preserving transformation. RealNVP adds scaling factors to the transformation:

$$\mathbf{x}_2 = \exp(s_\theta(\mathbf{z}_1)) \odot \mathbf{z}_2 + m_\theta(\mathbf{z}_1)$$

where $\odot$ denotes elementwise product. This results in a non-volume preserving transformation.

### Autoregressive Flow Models

Some autoregressive models can also be interpreted as flow models. For a Gaussian autoregressive model, one receives some Gaussian noise for each dimension of $\mathbf{x}$, which can be treated as the latent variables $\mathbf{z}$. Such transformations are also invertible, meaning that given $\mathbf{x}$ and the model parameters, we can obtain $\mathbf{z}$ exactly.

**Masked Autoregressive Flow (MAF)** uses this interpretation, where the forward mapping is an autoregressive model. However, sampling is sequential and slow, in $O(n)$ time where $n$ is the dimension of the samples.

**MAF Architecture and Mathematical Formulation:**

The MAF is comprised of **Masked Autoencoder for Distribution Estimation (MADE)** blocks, which has a special masking scheme at each layer such that the autoregressive property is preserved. In particular, we consider a Gaussian autoregressive model:

$$p(\mathbf{x}) = \prod_{i=1}^n p(x_i | \mathbf{x}_{<i})$$

such that the conditional Gaussians $p(x_i | \mathbf{x}_{<i}) = \mathcal{N}(x_i | \mu_i, (\exp(\alpha_i))^2)$ are parameterized by neural networks $\mu_i = f_{\mu_i}(\mathbf{x}_{<i})$ and $\alpha_i = f_{\alpha_i}(\mathbf{x}_{<i})$. Note that $\alpha_i$ denotes the log standard deviation of the Gaussian $p(x_i | \mathbf{x}_{<i})$.

As seen in the change of variables formula, a normalizing flow uses a series of deterministic and invertible mappings $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$ such that $\mathbf{x} = f(\mathbf{z})$ and $\mathbf{z} = f^{-1}(\mathbf{x})$ to transform a simple prior distribution $p_z$ (e.g. isotropic Gaussian) into a more expressive one. In particular, a normalizing flow which composes $k$ invertible transformations $\{f_j\}_{j=1}^k$ such that $\mathbf{x} = f_k \circ f_{k-1} \circ \cdots \circ f_1(\mathbf{z}_0)$ takes advantage of the change-of-variables property:

$$\log p(\mathbf{x}) = \log p_z(f^{-1}(\mathbf{x})) + \sum_{j=1}^k \log \left|\det\left(\frac{\partial f_j^{-1}(\mathbf{x}_j)}{\partial \mathbf{x}_j}\right)\right|$$

In MAF, the forward mapping is: $x_i = \mu_i + z_i \cdot \exp(\alpha_i)$, and the inverse mapping is: $z_i = (x_i - \mu_i)/\exp(\alpha_i)$. The log of the absolute value of the determinant of the Jacobian is:

$$\log \left|\det\left(\frac{\partial f^{-1}}{\partial \mathbf{x}}\right)\right| = -\sum_{i=1}^n \alpha_i$$

where $\mu_i$ and $\alpha_i$ are as defined above.

**Connection between $p(\mathbf{x})$ and $\log p(\mathbf{x})$ formulations:**

The two formulations are equivalent but serve different purposes:

1. **$p(\mathbf{x})$ formulation (autoregressive view)**:

$$p(\mathbf{x}) = \prod_{i=1}^n p(x_i | \mathbf{x}_{<i}) = \prod_{i=1}^n \mathcal{N}(x_i | \mu_i, (\exp(\alpha_i))^2)$$


2. **$\log p(\mathbf{x})$ formulation (flow view)**:

$$\log p(\mathbf{x}) = \log p_z(f^{-1}(\mathbf{x})) + \sum_{j=1}^k \log \left|\det\left(\frac{\partial f_j^{-1}(\mathbf{x}_j)}{\partial \mathbf{x}_j}\right)\right|$$

**How they relate:**

Taking the logarithm of the autoregressive formulation:

$$\log p(\mathbf{x}) = \sum_{i=1}^n \log p(x_i | \mathbf{x}_{<i}) = \sum_{i=1}^n \log \mathcal{N}(x_i | \mu_i, (\exp(\alpha_i))^2)$$

For a Gaussian distribution $\mathcal{N}(x | \mu, \sigma^2)$, we have:

$$\log \mathcal{N}(x | \mu, \sigma^2) = -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(x-\mu)^2}{2\sigma^2}$$

Substituting $\sigma = \exp(\alpha_i)$ and using the inverse mapping $z_i = (x_i - \mu_i)/\exp(\alpha_i)$:

$$\log p(\mathbf{x}) = \sum_{i=1}^n \left[-\frac{1}{2}\log(2\pi) - \alpha_i - \frac{z_i^2}{2}\right] = \sum_{i=1}^n \log \mathcal{N}(z_i | 0, 1) - \sum_{i=1}^n \alpha_i$$

This shows that the autoregressive formulation (using conditional Gaussians) is equivalent to the flow formulation (using change of variables with a standard normal prior and the Jacobian determinant term $-\sum_{i=1}^n \alpha_i$).

**Key insight:** The $\alpha_i$ terms serve dual purposes - they parameterize the conditional standard deviations in the autoregressive view, and they contribute to the Jacobian determinant in the flow view.

To address the sampling problem, the **Inverse Autoregressive Flow (IAF)** simply inverts the generating process. In this case, the sampling (generation), is still parallelized. However, computing the likelihood of new data points is slow.

**Forward mapping from $\mathbf{z} \rightarrow \mathbf{x}$ (parallel):**

1. Sample $z_i \sim \mathcal{N}(0,1)$ for $i = 1, \ldots, n$

2. Compute all $\mu_i, \alpha_i$ (can be done in parallel)

3. Let $x_1 = \exp(\alpha_1)z_1 + \mu_1$

4. Let $x_2 = \exp(\alpha_2)z_2 + \mu_2$

5. $\ldots$

**Inverse mapping from $\mathbf{x} \rightarrow \mathbf{z}$ (sequential):**

1. Let $z_1 = (x_1 - \mu_1)/\exp(\alpha_1)$

2. Compute $\mu_2(z_1), \alpha_2(z_1)$

3. Let $z_2 = (x_2 - \mu_2)/\exp(\alpha_2)$

4. Compute $\mu_3(z_1,z_2), \alpha_3(z_1,z_2)$

5. $\ldots$

**Key insight:** Fast to sample from, slow to evaluate likelihoods of data points (train).

**Efficient Likelihood for Generated Points:**
However, for generated points the likelihood can be computed efficiently (since the noise are already obtained). When we generate samples using IAF, we start with known noise values $\mathbf{z}$ and transform them to get $\mathbf{x}$. Since we already have the noise values, we don't need to perform the expensive sequential inverse mapping to recover them. We can directly compute the likelihood using the change of variables formula:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}) - \sum_{i=1}^n \alpha_i$$

where we already know all the $\alpha_i$ values from the forward pass. This is much faster than the $O(n)$ sequential computation required for arbitrary data points.

**Derivation of the Change of Variables Formula for IAF:**

Let's derive how we get this formula. Starting with the general change of variables formula:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}) + \log \left|\det\left(\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right)\right|$$

For IAF, the forward transformation is:

$$x_i = \exp(\alpha_i)z_i + \mu_i$$

The inverse transformation is:

$$z_i = \frac{x_i - \mu_i}{\exp(\alpha_i)}$$

The Jacobian matrix $\frac{\partial \mathbf{z}}{\partial \mathbf{x}}$ is diagonal because each $z_i$ only depends on $x_i$:

$$\frac{\partial z_i}{\partial x_j} = \begin{cases} 
\frac{1}{\exp(\alpha_i)} & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}$$

Therefore, the determinant is the product of the diagonal elements:

$$\det\left(\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right) = \prod_{i=1}^n \frac{1}{\exp(\alpha_i)} = \exp\left(-\sum_{i=1}^n \alpha_i\right)$$

Taking the absolute value and logarithm:

$$\log \left|\det\left(\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right)\right| = \log \exp\left(-\sum_{i=1}^n \alpha_i\right) = -\sum_{i=1}^n \alpha_i$$

Substituting back into the change of variables formula:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}) - \sum_{i=1}^n \alpha_i$$

This derivation shows why the likelihood computation is efficient for generated samples - we already have all the $\alpha_i$ values from the forward pass, so we just need to sum them up.

### RNADE vs. Autoregressive Flow Models

**RNADE (Real-valued Neural Autoregressive Density Estimator)** and **Autoregressive Flow models** are both autoregressive approaches but with key differences:

#### **RNADE:**
- **Direct density modeling**: RNADE directly models the conditional densities $p(x_i | x_{<i})$ using neural networks
- **Sequential sampling**: Generation requires sampling each dimension sequentially: $x_1 \sim p(x_1)$, then $x_2 \sim p(x_2|x_1)$, etc.
- **No invertible transformation**: RNADE doesn't use the change of variables formula - it directly parameterizes the conditional distributions
- **Tractable likelihood**: Can compute exact likelihoods efficiently since it directly models the density

#### **Autoregressive Flow Models (MAF/IAF):**
- **Flow interpretation**: Treats the autoregressive process as an invertible transformation from noise $\mathbf{z}$ to data $\mathbf{x}$
- **Change of variables**: Uses the change of variables formula to compute likelihoods
- **Invertible transformation**: Can be viewed as $f: \mathbf{z} \rightarrow \mathbf{x}$ where $\mathbf{z}$ is independent noise
- **Two directions**: 
  - **MAF**: Forward mapping is autoregressive (slow sampling, fast likelihood)
  - **IAF**: Inverse mapping is autoregressive (fast sampling, slow likelihood)

#### **Key Differences:**

1. **Mathematical Framework**:
   - **RNADE**: Direct density estimation $p(\mathbf{x}) = \prod_i p(x_i | x_{<i})$
   - **Autoregressive Flow**: Invertible transformation with change of variables

2. **Computational Properties**:
   - **RNADE**: Sequential in both sampling and likelihood computation
   - **Autoregressive Flow**: Can optimize for either fast sampling (IAF) or fast likelihood (MAF)

3. **Interpretation**:
   - **RNADE**: Pure autoregressive model
   - **Autoregressive Flow**: Can be viewed as both autoregressive model and flow model

4. **Flexibility**:
   - **RNADE**: Limited to sequential computation
   - **Autoregressive Flow**: Can design for specific computational needs (sampling vs. likelihood)

The key insight is that Autoregressive Flow models provide a flow-based interpretation of autoregressive models, allowing for more flexible computational trade-offs while maintaining the expressive power of autoregressive modeling.

#### **Data Types:**

**RNADE**: 
- **Designed for continuous data**: RNADE was specifically developed for real-valued data
- **Gaussian conditionals**: Typically uses Gaussian distributions for the conditional densities $p(x_i | x_{<i})$
- **Not suitable for discrete data**: The Gaussian assumption doesn't work well for categorical or discrete variables

**Autoregressive Flow Models**:
- **Primarily continuous data**: Most implementations (MAF, IAF) are designed for continuous data
- **Can be extended**: Some variants can handle discrete data through different conditional distributions
- **Flow requirement**: The change of variables formula requires continuous, differentiable transformations

**Key Limitation**: Both RNADE and standard Autoregressive Flow models are primarily designed for **continuous data**.