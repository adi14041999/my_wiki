# Proving the Central Limit Theorem

The core of a [normal (Gaussian) distribution](normal_distribution.md) is the function $e^{-x^2}$. Among all symmetric, smooth densities concentrated near the center, probability theory singles out this shape. Why? And we have not yet explained why the [Central Limit Theorem](central_limit_theorem.md) (CLT) holds.

To start, we visualize the **convolution of two Gaussians**.

## Joint density on the plane

Let $X$ and $Y$ be independent with PDFs $f$ and $g$. Each outcome $(x, y)$ is a point in the plane. 

By independence, the joint density is

$$p(x, y) = f(x)\, g(y).$$

![img](jointdens.png)

Plotting $p(x, y)$ as a surface shows the distribution of all pairs when we sample from both variables.

![img](jointdens1.png)

## Diagonal slices and convolution

The convolution $(f * g)(s)$ asks how likely a pair is to **sum** to $s$. In the [diagonal-slices picture](convolution_continuous.md), we take the slice of $p(x, y)$ over the line

$$x + y = s$$

and consider the **area** under that slice (the cross-section of the surface above the line).

![img](slice0.png)

That area is almost $(f * g)(s)$; for a technical reason we divide by $\sqrt{2}$:

$$(f * g)(s) = \frac{1}{\sqrt{2}} \times (\text{area under the slice}).$$

![img](slice1.png)
![img](slice2.png)
![img](slice3.png)
![img](slice4.png)

Apart from the factor $\frac{1}{\sqrt{2}}$, the same picture remains: $(f * g)(s)$ aggregates the joint density $f(x)\,g(y)$ over all pairs with $x + y = s$.

## Rotational symmetry for Gaussians

Take $f(x) = e^{-x^2}$ and $g(y) = e^{-y^2}$. Then

$$p(x, y) = e^{-x^2} e^{-y^2} = e^{-(x^2 + y^2)},$$

which depends only on $r^2 = x^2 + y^2$, the squared distance from the origin. 

![img](circ0.png)
![img](circ1.png)
![img](circ2.png)

The surface is **rotationally symmetric**. 

![img](sim1.png)
![img](sim2.png)

That property characterizes Gaussians among the distributions we might convolve. The [Herschel–Maxwell derivation](normal_distribution.md#herschel-maxwell-derivation-of-the-normal-distribution) makes this precise: if coordinate errors are **independent** with the same marginal $f$, so $p(x, y) = f(x)\, f(y)$, and the joint density is **isotropic** (depends only on $r = \sqrt{x^2 + y^2}$), then $f$ must be Gaussian-shaped, $f(x) \propto e^{-b^2 x^2}$. No other marginal satisfies both postulates. 

![img](hmd.png)

For any non-Gaussian $f$, the product $f(x)\, f(y)$ is not rotationally symmetric in the plane. Diagonal slices are awkward shapes, and their areas are no easier to compute than the convolution integral itself.

![img](awk.png)

For bell curves, symmetry is the lever.

## Proving Gaussian + Gaussian = Gaussian

Fix $s$ and consider the slice over $x + y = s$. The convolution $(f * g)(s)$ is a function of $s$; we want the slice area in terms of $s$.

![img](obliq.png)

The line $x + y = s$ meets the axes at $(s, 0)$ and $(0, s)$. Its perpendicular distance from the origin is

$$d = \frac{|s|}{\sqrt{2}}.$$

![img](obliq1.png)

By rotational symmetry, that oblique slice matches a slice **parallel to the $y$-axis** at the same distance $d$ from the origin (a $45^\circ$ rotation).

![img](para0.png)
![img](para1.png)

Integrating along a vertical slice is simpler: $x$ is constant on that line. On the vertical slice at distance $d$, we have $x = s/\sqrt{2}$ (for the standard orientation), so every factor involving $x$ is constant and factors out of the integral over $y$.

All dependence on $s$ is then explicit in the prefactor; the remaining integral

$$\int_{-\infty}^{\infty} e^{-y^2}\, dy = \sqrt{\pi}$$

is a fixed constant with **no** $s$.

![img](f0.png)
![img](f1.png)

The convolution value $(f * g)(s)$ differs from that area only by the factor $1/\sqrt{2}$, which is absorbed into the overall normalization. What matters for the CLT story is the structural fact:

**The convolution of two Gaussians is again a Gaussian.**

![img](gg0.png)
![img](gg1.png)
![img](gg2.png)
![img](gg3.png)

If we restore the full constants for a mean-zero normal with arbitrary standard deviation $\sigma$,

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-x^2/(2\sigma^2)},$$

the same reasoning goes through unchanged. The factors of $\sqrt{2}$ still appear in the exponent and in front when we compare slice areas to $(f * g)(s)$.

The conclusion is that the convolution of two independent $\mathcal{N}(0, \sigma^2)$ variables is again normal, with standard deviation $\sqrt{2}\,\sigma$:

$$X \sim \mathcal{N}(0, \sigma^2),\quad Y \sim \mathcal{N}(0, \sigma^2)\ \text{independent} \quad \Longrightarrow \quad X + Y \sim \mathcal{N}(0, 2\sigma^2).$$

Equivalently, $\operatorname{SD}(X + Y) = \sqrt{2}\,\sigma$.

It would be circular to cite the [Central Limit Theorem](central_limit_theorem.md) as the justification for the fact that the convolution of two independent $\mathcal{N}(0, \sigma^2)$ variables is $\mathcal{N}(0, 2\sigma^2)$ (equivalently, that $\operatorname{SD}(X + Y) = \sqrt{2}\,\sigma$). The computation above is rather why a **Gaussian** sits at the center of the CLT in the first place and not some other limiting shape. Sums of independent normals stay normal. That stability under convolution is what makes the bell curve the shape that persists each time we add another independent summand. Convolve again and we still obtain a Gaussian.

## A high-level proof of the CLT

A standard proof of the [Central Limit Theorem](central_limit_theorem.md) proceeds in two steps.

**Step 1 (universality):** Let $X_1, X_2, \ldots$ be i.i.d. with finite variance, and write $S_n = X_1 + \cdots + X_n$. The law of $S_n$ is the $n$-fold convolution of the common marginal PDF (or PMF). After centering and scaling,

$$\frac{S_n - n\mu}{\sigma\sqrt{n}} \qquad (\mu = \mathbb{E}[X_1],\ \sigma^2 = \operatorname{Var}(X_1)),$$

these distributions converge to a single **universal** limit. This step is technical; we do not prove it here. **Moment-generating functions** are a standard tool: they show that a limit exists in the space of distributions, without yet identifying its shape.

**Step 2 (the limit is Gaussian):** Above we showed that convolving two Gaussians yields another Gaussian (equivalently, $X + Y \sim \mathcal{N}(0, 2\sigma^2)$ when $X, Y \sim \mathcal{N}(0, \sigma^2)$ are independent). The normal law is therefore a **fixed point** of convolution: further convolutions do not leave the Gaussian family. Step 1 says every finite-variance law tends to the same limit. **Gaussians lie in that class and are fixed, so the universal limit must be Gaussian.**

![img](cltproof.png)

!!! note "Slice geometry and Herschel–Maxwell"
    The [Herschel–Maxwell derivation](normal_distribution.md#herschel-maxwell-derivation-of-the-normal-distribution) shows that independence $p(x,y) = f(x)f(y)$ plus rotational symmetry force $f(x) \propto e^{-b^2 x^2}$—that is, **why** the limiting shape must be Gaussian. The derivation also says why $\pi$ is in the Gaussian formula.

    The **slice-integral** visualization proof of Gaussian + Gaussian = Gaussian is the second step of the proof of the CLT. Moreover, like the Herschel–Maxwell derivation, it leverages rotational symmetry.