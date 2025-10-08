# Derivatives

## Scalar case

You are probably familiar with the concept of a derivative in the scalar case.

Given a function $f: \mathbb{R} \to \mathbb{R}$, the derivative of $f$ at a point $x \in \mathbb{R}$ is defined as:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Derivatives are a way to measure change. In the scalar case, the derivative of the function $f$ at the point $x$ tells us how much the function $f$ changes as the input $x$ changes by a small amount $\varepsilon$:

$$f(x + \varepsilon) \approx f(x) + \varepsilon f'(x)$$

For ease of notation we will commonly assign a name to the output of $f$, say $y = f(x)$, and write $\frac{\partial y}{\partial x}$ for the derivative of $y$ with respect to $x$. This notation emphasizes that $\frac{\partial y}{\partial x}$ is the rate of change between the variables $x$ and $y$; concretely if $x$ were to change by $\varepsilon$ then $y$ will change by approximately $\varepsilon \frac{\partial y}{\partial x}$.

We can write this relationship as

$$x \to x + \Delta x \implies y \to y + \frac{\partial y}{\partial x} \Delta x$$

You should read this as saying "changing $x$ to $x + \Delta x$ implies that $y$ will change to approximately $y + \Delta x \frac{\partial y}{\partial x}$". This notation is nonstandard, but it emphasizes the relationship between changes in $x$ and changes in $y$.

## Chain rule

The chain rule tells us how to compute the derivative of the composition of functions. In the scalar case suppose that $f, g: \mathbb{R} \to \mathbb{R}$ and $y = f(x)$, $z = g(y)$; then we can also write $z = (g \circ f)(x)$, or draw the following computational graph:

$$x \xrightarrow{f} y \xrightarrow{g} z$$

The (scalar) chain rule tells us that

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x}$$

This equation makes intuitive sense. The derivatives $\frac{\partial z}{\partial y}$ and $\frac{\partial y}{\partial x}$ give:

$$x \to x + \Delta x \implies y \to y + \frac{\partial y}{\partial x} \Delta x$$

$$y \to y + \Delta y \implies z \to z + \frac{\partial z}{\partial y} \Delta y$$

Combining these two rules lets us compute the effect of $x$ on $z$: if $x$ changes by $\Delta x$ then $y$ will change by $\frac{\partial y}{\partial x}\Delta x$, so we have $\Delta y = \frac{\partial y}{\partial x}\Delta x$. If $y$ changes by $\Delta y$ then $z$ will change by $\frac{\partial z}{\partial y}\Delta y = \frac{\partial z}{\partial y}\frac{\partial y}{\partial x}\Delta x$ which is exactly what the chain rule tells us.

## Gradient: Vector in, scalar out

This same intuition carries over into the vector case. Now suppose that $f: \mathbb{R}^N \to \mathbb{R}$ takes a vector as input and produces a scalar. The derivative of $f$ at the point $\mathbf{x} \in \mathbb{R}^N$ is now called the gradient, and it is defined as:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \lim_{\mathbf{h} \to \mathbf{0}} \frac{f(\mathbf{x} + \mathbf{h}) - f(\mathbf{x})}{\|\mathbf{h}\|}$$

Now the gradient $\nabla_{\mathbf{x}} f(\mathbf{x}) \in \mathbb{R}^N$ is a vector, with the same intuition as the scalar case. If we set $y = f(\mathbf{x})$ then we have the relationship

$$\mathbf{x} \to \mathbf{x} + \Delta \mathbf{x} \implies y \to y + \frac{\partial y}{\partial \mathbf{x}} \cdot \Delta \mathbf{x}$$

The formula changes a bit from the scalar case to account for the fact that $\mathbf{x}$, $\Delta \mathbf{x}$, and $\frac{\partial y}{\partial \mathbf{x}}$ are now vectors in $\mathbb{R}^N$ while $y$ is a scalar. In particular when multiplying $\frac{\partial y}{\partial \mathbf{x}}$ by $\Delta \mathbf{x}$ we use the dot product, which combines two vectors to give a scalar.

One nice outcome of this formula is that it gives meaning to the individual elements of the gradient $\frac{\partial y}{\partial \mathbf{x}}$. Suppose that $\Delta \mathbf{x}$ is the $i$th basis vector $\mathbf{e}_i$, so that the $i$th coordinate of $\Delta \mathbf{x}$ is 1 and all other coordinates of $\Delta \mathbf{x}$ are 0. 

Then the dot product $\frac{\partial y}{\partial \mathbf{x}} \cdot \Delta \mathbf{x}$ here is simply the $i$th coordinate of $\frac{\partial y}{\partial \mathbf{x}}$. Thus the $i$th coordinate of $\frac{\partial y }{\partial \mathbf{x}}$ times $\varepsilon$ tells us the amount by which $y$ will change if we move $\mathbf{x}$ by a small amount $\varepsilon$ along the $i$th coordinate axis (meaning, if we change only the $i$th component of $\mathbf{x}$ by $\varepsilon$, where $\varepsilon$ is the $i$th component of $\Delta \mathbf{x}$).

This means that we can also view the gradient $\frac{\partial y}{\partial \mathbf{x}}$ as a vector of partial derivatives:

$$\frac{\partial y}{\partial \mathbf{x}} = \left(\frac{\partial y}{\partial x_1}, \frac{\partial y}{\partial x_2}, \ldots, \frac{\partial y}{\partial x_N}\right)$$

where $x_i$ is the $i$th coordinate of the vector $\mathbf{x}$, which is a scalar, so each partial derivative $\frac{\partial y}{\partial x_i}$ is also a scalar.

## Jacobian: Vector in, Vector out

Now suppose that $f: \mathbb{R}^N \to \mathbb{R}^M$ takes a vector as input and produces a vector as output. Then the derivative of $f$ at a point $\mathbf{x}$, also called the Jacobian, is the $M \times N$ matrix of partial derivatives. If we again set $\mathbf{y} = f(\mathbf{x})$ then we can write:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_N} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_M}{\partial x_1} & \cdots & \frac{\partial y_M}{\partial x_N}
\end{pmatrix}$$

The Jacobian tells us the relationship between each element of $\mathbf{x}$ and each element of $\mathbf{y}$: the $(i, j)$-th element of $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is equal to $\frac{\partial y_i}{\partial x_j}$, so it tells us the amount by which $y_i$ will change if $x_j$ is changed by a small amount.

Just as in the previous cases, the Jacobian tells us the relationship between changes in the input and changes in the output:

$$\mathbf{x} \to \mathbf{x} + \Delta \mathbf{x} \implies \mathbf{y} \to \mathbf{y} + \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \Delta \mathbf{x}$$

Here $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is a $M \times N$ matrix and $\Delta \mathbf{x}$ is an $N$-dimensional vector, so the product $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \Delta \mathbf{x}$ is a matrix-vector multiplication resulting in an $M$-dimensional vector.

It's worth noting that each row of the Jacobian matrix $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is actually a gradient! Specifically, the $i$th row of $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is the gradient of the scalar function $y_i$ with respect to $\mathbf{x}$:

$$\text{Row } i \text{ of } \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \nabla_{\mathbf{x}} y_i = \left(\frac{\partial y_i}{\partial x_1}, \frac{\partial y_i}{\partial x_2}, \ldots, \frac{\partial y_i}{\partial x_N}\right)$$

This makes sense because $y_i$ is a scalar function of the vector $\mathbf{x}$, so its gradient is an $N$-dimensional vector. When we stack all $M$ of these gradient vectors as rows, we get the $M \times N$ Jacobian matrix.

This insight also explains why the matrix-vector multiplication $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \Delta \mathbf{x}$ works the way it does. Since each row of the Jacobian is a gradient, the multiplication is equivalent to computing the dot product of each gradient with $\Delta \mathbf{x}$:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \Delta \mathbf{x} = \begin{pmatrix}
\nabla_{\mathbf{x}} y_1 \cdot \Delta \mathbf{x} \\
\nabla_{\mathbf{x}} y_2 \cdot \Delta \mathbf{x} \\
\vdots \\
\nabla_{\mathbf{x}} y_M \cdot \Delta \mathbf{x}
\end{pmatrix}$$

In other words, the Jacobian-vector product is just a stack of gradient-vector products! Each component of the result tells us how much the corresponding output component $y_i$ changes when we move $\mathbf{x}$ by $\Delta \mathbf{x}$.

For example, if $M = 2$ and $N = 3$, we might have:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{pmatrix}, \quad \Delta \mathbf{x} = \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix}$$

Then:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \Delta \mathbf{x} = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{pmatrix} \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix} = \begin{pmatrix} 2(0.1) + 1(0.2) + 0(0.3) \\ 0(0.1) + 3(0.2) + 1(0.3) \end{pmatrix} = \begin{pmatrix} 0.4 \\ 0.9 \end{pmatrix}$$

The chain rule can be extended to the vector case using Jacobian matrices. Suppose that $f: \mathbb{R}^N \to \mathbb{R}^M$ and $g: \mathbb{R}^M \to \mathbb{R}^K$. Let $\mathbf{x} \in \mathbb{R}^N$, $\mathbf{y} \in \mathbb{R}^M$, and $\mathbf{z} \in \mathbb{R}^K$ with $\mathbf{y} = f(\mathbf{x})$ and $\mathbf{z} = g(\mathbf{y})$, so we have the same computational graph as the scalar case:

$$\mathbf{x} \xrightarrow{f} \mathbf{y} \xrightarrow{g} \mathbf{z}$$

The chain rule also has the same form as the scalar case:

$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

However now each of these terms is a matrix: $\frac{\partial \mathbf{z}}{\partial \mathbf{y}}$ is a $K \times M$ matrix, $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is a $M \times N$ matrix, and $\frac{\partial \mathbf{z}}{\partial \mathbf{x}}$ is a $K \times N$ matrix; the multiplication of $\frac{\partial \mathbf{z}}{\partial \mathbf{y}}$ and $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is matrix multiplication.

## Generalized Jacobian: Tensor in, Tensor out