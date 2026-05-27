# Notes from AI podcasts on YouTube

## Mindscape 336 | Anil Ananthaswamy on the Mathematics of Neural Nets and aI

### Perceptron Convergence Proof

### The legendary weekend: how Ted Hoff and Bernie Widrow built the first hardware neuron

In the late 1950s, Bernie Widrow was an assistant professor at Stanford. A graduate student named Ted Hoff knocked on his door looking for a PhD project. Over a single afternoon at the blackboard, the two of them sketched out what is now known as the **Least Mean Squares (LMS)** algorithm, an algebraic update rule for training a single linear neuron.

For a linear unit with weights $\mathbf{w}$, input $\mathbf{x}$, output $y = \mathbf{w}^\top \mathbf{x}$, target $d$, and error $e = d - y$, the LMS rule is

$$\mathbf{w} \leftarrow \mathbf{w} + \eta\, e\, \mathbf{x}.$$

In modern terms, this is **Stochastic Gradient Descent** on the per-sample squared-error loss

$$L(\mathbf{w}) = \tfrac{1}{2}(d - y)^2 = \tfrac{1}{2}\bigl(d - \mathbf{w}^\top \mathbf{x}\bigr)^2.$$

To see why, compute the gradient of $L$ with respect to $\mathbf{w}$. Treating $L$ as a composition $L(y(\mathbf{w}))$ with $y = \mathbf{w}^\top \mathbf{x}$, the chain rule gives

$$\nabla_{\mathbf{w}}\, L = \frac{\partial L}{\partial y}\, \nabla_{\mathbf{w}}\, y.$$

The two factors are

$$\frac{\partial L}{\partial y} = -(d - y) = -e, \qquad \nabla_{\mathbf{w}}\, y = \nabla_{\mathbf{w}}\, (\mathbf{w}^\top \mathbf{x}) = \mathbf{x}.$$

So

$$\nabla_{\mathbf{w}}\, L = -e\, \mathbf{x}.$$

A gradient-descent step with learning rate $\eta > 0$ is $\mathbf{w} \leftarrow \mathbf{w} - \eta\, \nabla_{\mathbf{w}}\, L$, which substitutes back to

$$\mathbf{w} \leftarrow \mathbf{w} - \eta\, (-e\, \mathbf{x}) = \mathbf{w} + \eta\, e\, \mathbf{x},$$

recovering the LMS rule above. Applying this update one sample at a time, rather than over a full-batch gradient, is exactly **stochastic** gradient descent. Widrow and Hoff arrived at the same rule via algebra rather than calculus, but the update is identical to SGD on the squared error.

The same evening, Hoff programmed the rule on an analog computer that Lockheed had donated to Stanford, and it worked. With the supply rooms closed for the weekend, the two walked over to a local electronics store, bought components, and built the first hardware artificial neuron in Hoff's apartment. By Monday morning they had a working device, an early ancestor of every artificial neuron in use today.

Hoff later left academia for a startup he was unsure about; Widrow encouraged him to take the offer. The startup turned out to be **Intel**, where Hoff went on to architect the [Intel 4004](https://en.wikipedia.org/wiki/Intel_4004), the first commercial microprocessor.


### Why sigmoids mattered

Early artificial neurons used a hard **threshold** activation,

$$\sigma_{\text{step}}(z) = \begin{cases} 1, & z \geq 0, \\ 0, & z < 0, \end{cases}$$

so a unit fired only once its weighted input crossed a threshold. This is non-differentiable: the slope is $0$ almost everywhere and undefined at the jump. As long as a network only had one trainable layer, that did not matter (the perceptron rule did not need a usable derivative). But once we try to stack layers and train them jointly, every node in the computational graph must be differentiable so that the **chain rule** can propagate error from the output back to every weight.

In 1986, Rumelhart, Hinton, and Williams published a three-and-a-half-page paper in *Nature* showing exactly how to do this: the **backpropagation** algorithm. A key enabling change was replacing the step function with a smooth surrogate— the **sigmoid**

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr).$$

The sigmoid has the same qualitative shape as the step (saturates near $0$ for large negative $z$ and near $1$ for large positive $z$) but is smooth everywhere, so gradients flow through it. Once every layer is differentiable, the chain rule can compute $\partial L / \partial w$ for every weight $w$ in the network, and gradient descent can train arbitrarily deep architectures.

!!! note "Sigmoids today"
    Modern networks rarely use sigmoids in hidden layers because we get vanishing gradients in deep stacks. Activations like **ReLU** ($\max(0, z)$) and its variants have largely replaced them in hidden layers, while sigmoid and softmax remain standard for output layers that produce probabilities. The conceptual shift, though, was sigmoid's. Differentiability everywhere is what made deep learning trainable.

### The curse of dimensionality

The **curse of dimensionality** predates neural networks. The cleanest illustration is the **$k$-nearest neighbors ($k$-NN)** classifier.

Suppose we represent each $10 \times 10$ pixel image as a point $\mathbf{x} \in \mathbb{R}^{100}$ ([Spatializing Data](spatializing_data.md)). Cats cluster in one region, dogs in another. Given a new image $\mathbf{x}^*$, $k$-NN classifies it by majority vote among the $k$ closest training points under some distance; usually Euclidean,

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}.$$

Choosing $k > 1$ smooths over noisy or mislabeled points (a single neighbor causes overfitting). The whole approach assumes that **similar points are closer together** than dissimilar ones.

In high dimensions, that assumption breaks down. For i.i.d. data in $\mathbb{R}^n$ with most reasonable distributions, the ratio of the maximum to the minimum pairwise distance concentrates around $1$ as $n \to \infty$:

$$\frac{d_{\max}(\mathbf{x}^*) - d_{\min}(\mathbf{x}^*)}{d_{\min}(\mathbf{x}^*)} \xrightarrow{n \to \infty} 0.$$

In other words, every point becomes roughly equidistant from every other point, so "nearest" carries almost no information. Distance-based methods like $k$-NN, naïve clustering, and similar stop working.

This matters because **dimensionality = number of features**. Each pixel of the image was a feature. As we add features we accumulate signal, but we also walk further into the curse, and naïve [vector similarity search](vector_similarity_search.md) degrades.

### Principal Component Analysis

One way to fight the curse is to **reduce dimensions** while keeping the structure that matters for the task. **Principal Component Analysis (PCA)** projects high-dimensional data onto a lower-dimensional subspace chosen to capture as much **variance** as possible.

PCA helps when the data has **low intrinsic dimensionality**: most of its variation lies in a few directions of the ambient $\mathbb{R}^n$, so a small $k$ retains nearly all the variance. We can then train a classifier on the $k$-dim representation, sidestepping the curse. PCA does **not** help when the data spreads roughly equally in all $n$ directions.

### Kernel methods

Some classification problems are not linearly separable in the input space. A standard cartoon example:

- Red dots cluster near the origin in $\mathbb{R}^2$.
- Green dots form an annular ring around them.

No straight line in the plane separates the two. But lift the data into $\mathbb{R}^3$ by appending a third coordinate $z = x^2 + y^2$:

$$\phi : (x, y) \mapsto (x,\, y,\, x^2 + y^2).$$

Now the green ring sits at large $z$ and the red core at small $z$, and a horizontal **hyperplane** $z = c$ separates them. Projecting the hyperplane back to the plane recovers a **nonlinear** decision boundary— a circle!

This generalizes. If the original data lives in $\mathbb{R}^n$ and we lift it via some feature map $\phi : \mathbb{R}^n \to \mathbb{R}^N$ with $N \gg n$, a linear classifier in $\mathbb{R}^N$ becomes a nonlinear classifier in $\mathbb{R}^n$. The catch is computational: linear classifiers (especially SVMs) depend on data through dot products

$$\langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle.$$

Computing $\phi(\mathbf{x})$ explicitly when $N$ is a million is expensive; when $N$ is *infinite*, it is impossible.

The **kernel trick** bypasses $\phi$ entirely. We choose a function $K : \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ such that

$$K(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle$$

for some implicit feature map $\phi$. Whenever the algorithm needs an inner product in the high-dimensional space, we evaluate $K(\mathbf{x}, \mathbf{y})$ on the original low-dimensional inputs $\mathbf{x}, \mathbf{y}$. No transformation to $\phi$ or high-dimensional dot product required!
