# Normal Distribution

The **Normal distribution** (also called the **Gaussian distribution**) is one of the most important continuous distributions in probability and statistics. It appears naturally in many contexts due to the Central Limit Theorem and provides a foundation for many statistical methods.

## Standard Normal Distribution

A continuous random variable $Z$ is said to have the **standard Normal distribution** if its PDF $f_Z$ is given by:

$$f_Z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}, \quad -\infty < z < \infty$$

We write this as $Z \sim N(0, 1)$ since, as we will show, $Z$ has mean 0 and variance 1.

The constant $\frac{1}{\sqrt{2\pi}}$ in front of the PDF may look surprising (why is something with $\pi$ needed in front of something with $e$, when there are no circles in sight?), but it's exactly what is needed to make the PDF integrate to 1. Such constants are called **normalizing constants** because they normalize the total area under the PDF to 1.

By symmetry, the mean of the standard normal distribution is 0. Here's why:

The standard normal PDF is symmetric about 0:

$$f_Z(z) = f_Z(-z) \quad \text{for all } z$$

This means the distribution looks the same on both sides of 0. The mean is defined as:

$$E[Z] = \int_{-\infty}^{\infty} z \cdot f_Z(z) \, dz$$

Let's split this integral into two parts:

$$E[Z] = \int_{-\infty}^0 z \cdot f_Z(z) \, dz + \int_0^{\infty} z \cdot f_Z(z) \, dz$$

**First integral** ($-\infty$ to 0): Let $u = -z$, so $z = -u$ and $dz = -du$

$$\int_{-\infty}^0 z \cdot f_Z(z) \, dz = \int_{\infty}^0 (-u) \cdot f_Z(-u) \cdot (-du) = \int_0^{\infty} u \cdot f_Z(u) \, du$$

**Second integral** (0 to $\infty$): This is already in the right form

$$\int_0^{\infty} z \cdot f_Z(z) \, dz$$

$$E[Z] = \int_0^{\infty} u \cdot f_Z(u) \, du + \int_0^{\infty} z \cdot f_Z(z) \, dz$$

Since $u$ and $z$ are just dummy variables, we can write this as:

$$E[Z] = \int_0^{\infty} z \cdot f_Z(z) \, dz + \int_0^{\infty} z \cdot f_Z(z) \, dz = 2 \int_0^{\infty} z \cdot f_Z(z) \, dz$$

The integrand $z \cdot f_Z(z)$ is **odd** because:

- $f_Z(z) = f_Z(-z)$ (even function)

- $z$ is odd

- Product of even and odd functions is odd

For odd functions, the integral from $-\infty$ to $\infty$ equals 0.

Therefore, $E[Z] = 0$.

Now let's show that the variance of the standard normal distribution is 1. The variance is defined as:

$$\text{Var}(Z) = E[(Z - E[Z])^2] = E[Z^2]$$

Since we already showed that $E[Z] = 0$, we have $\text{Var}(Z) = E[Z^2]$.

Calculating $E[Z^2]$:

$$E[Z^2] = \int_{-\infty}^{\infty} z^2 \cdot f_Z(z) \, dz = \int_{-\infty}^{\infty} z^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz$$

Let's use integration by parts with:

- $u = z$ and $dv = z \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz$

- $du = dz$ and $v = -\frac{1}{\sqrt{2\pi}} e^{-z^2/2}$

**Integration by parts formula**: $\int u \, dv = uv - \int v \, du$

$$E[Z^2] = \left[z \cdot \left(-\frac{1}{\sqrt{2\pi}} e^{-z^2/2}\right)\right]_{-\infty}^{\infty} - \int_{-\infty}^{\infty} \left(-\frac{1}{\sqrt{2\pi}} e^{-z^2/2}\right) \, dz$$

The boundary term evaluates to 0:

- **At $z = \infty$**: $z \cdot e^{-z^2/2} \to 0$ (exponential decay dominates)

- **At $z = -\infty$**: $z \cdot e^{-z^2/2} \to 0$ (exponential decay dominates)

Therefore:

$$E[Z^2] = 0 - \int_{-\infty}^{\infty} \left(-\frac{1}{\sqrt{2\pi}} e^{-z^2/2}\right) \, dz = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz$$

The remaining integral is exactly the integral of the PDF from $-\infty$ to $\infty$, which equals 1:

$$E[Z^2] = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz = 1$$

Therefore, $\text{Var}(Z) = E[Z^2] = 1$.

The standard Normal CDF $\Phi$ is the accumulated area under the PDF:

$$\Phi(z) = \int_{-\infty}^z f_Z(t) \, dt = \int_{-\infty}^z \frac{1}{\sqrt{2\pi}} e^{-t^2/2} \, dt$$

Some people, upon seeing the function $\Phi$ for the first time, express dismay that it is left in terms of an integral. Unfortunately, we have little choice in the matter: it turns out to be mathematically impossible to find a closed-form expression for the antiderivative of $f_Z$, meaning that we cannot express $\Phi$ as a finite sum of more familiar functions like polynomials or exponentials. But closed-form or no, it's still a well-defined function: if we give $\Phi$ an input $z$, it returns the accumulated area under the PDF from $-\infty$ up to $z$.

## General Normal Distribution

The **general Normal distribution** with mean $\mu$ and standard deviation $\sigma$ (or variance $\sigma^2$) is obtained by applying a linear transformation to the standard normal distribution. If $Z \sim N(0, 1)$ is a standard normal random variable, then:

$$X = \mu + \sigma Z$$

has a normal distribution with mean $\mu$ and standard deviation $\sigma$. We write this as $X \sim N(\mu, \sigma^2)$.

**Mean and Variance**: 

- $E[X] = E[\mu + \sigma Z] = \mu + \sigma E[Z] = \mu + \sigma \cdot 0 = \mu$

- $\text{Var}(X) = \text{Var}(\mu + \sigma Z) = \sigma^2 \text{Var}(Z) = \sigma^2 \cdot 1 = \sigma^2$

**CDF**: The CDF of $X \sim N(\mu, \sigma^2)$ is:

$$F_X(x) = \Phi\left(\frac{x - \mu}{\sigma}\right)$$

where $\Phi$ is the standard normal CDF.

**Derivation of the CDF**: Since $X = \mu + \sigma Z$, we can find the CDF by:

$$F_X(x) = P(X \leq x) = P(\mu + \sigma Z \leq x) = P\left(Z \leq \frac{x - \mu}{\sigma}\right) = \Phi\left(\frac{x - \mu}{\sigma}\right)$$

**PDF**: The PDF of $X \sim N(\mu, \sigma^2)$ is:

$$f_X(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \quad -\infty < x < \infty$$

**Derivation of the PDF from the CDF**: We can derive the PDF by taking the derivative of the CDF with respect to $x$:

$$f_X(x) = \frac{d}{dx} F_X(x) = \frac{d}{dx} \Phi\left(\frac{x - \mu}{\sigma}\right)$$

Using the chain rule and the fact that $\frac{d}{dz} \Phi(z) = \phi(z)$ (where $\phi(z)$ is the standard normal PDF):

$$f_X(x) = \phi\left(\frac{x - \mu}{\sigma}\right) \cdot \frac{d}{dx}\left(\frac{x - \mu}{\sigma}\right)$$

Since $\frac{d}{dx}\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{\sigma}$, we have:

$$f_X(x) = \phi\left(\frac{x - \mu}{\sigma}\right) \cdot \frac{1}{\sigma}$$

Substituting the standard normal PDF $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$:

$$f_X(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \cdot \frac{1}{\sigma} = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Standardization**: Any normal random variable $X \sim N(\mu, \sigma^2)$ can be standardized to a standard normal by:

$$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$

This is the inverse of the transformation $X = \mu + \sigma Z$.