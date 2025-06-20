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