# Tips for better problem solving

## TL;DR 

Some useful strategies to help you solve math problems better.

1. Use the defining features of the setup
2. Give things (meaningful) names
3. Leverage symmetry
4. Try describing one object in two different ways
5. Draw a picture (numbers, plots)
6. Ask or solve a simpler version of the problem
7. Read a lot, and think about math problems a lot
8. Always gut-check your answer!
9. Learn at least a little bit of programming

## Proving $\cos^2(\theta) = \frac{1}{2}[1 + \cos(2\theta)]$ using the Inscribed Angle Theorem

We'll prove the identity $\cos^2(\theta) = \frac{1}{2}[1 + \cos(2\theta)]$ using a geometric approach with the inscribed angle theorem.

Consider a circle with unit diameter. The circle has a center at $O$. Let there be a diameter $QR$ of length $1$. Let $P$ be a point on the circle not coinciding with $Q$ or $R$. Let $PM$ be a line perpendicular to the diameter $QR$. Thus, $M$ lies on the diameter. Below is a figure showing the circle.

![img](iat.png)

Let $\alpha = \angle PQR$ (the inscribed angle) and $\beta = \angle POR$ (the central angle). By the inscribed angle theorem, $\beta = 2\alpha$.

Since $QR$ is a diameter and $P$ lies on the semicircle, $\angle QPR = 90°$. This is a consequence of **Thales' theorem** (or the inscribed angle theorem): any angle inscribed in a semicircle (i.e., subtended by a diameter) is a right angle.

Since $QR$ is of length $1$ (unit diameter), and triangle $QPR$ is right-angled at $P$, we can use trigonometry. In triangle $QPR$:

- The hypotenuse is $QR = 1$

- The angle at $Q$ is $\alpha$

- Therefore, $PQ = \cos(\alpha)$

Now, we need to find the length of $QM$ in two different ways.

**First way:**

Consider triangle $PMQ$, which is right-angled at $M$ (since $PM$ is perpendicular to $QR$). In this triangle:

- The angle at $Q$ is $\alpha$

- The hypotenuse is $PQ$

- The adjacent side to angle $\alpha$ is $QM$

Therefore, $\cos(\alpha) = \frac{QM}{PQ}$.

But we already know that $PQ = \cos(\alpha)$ Substituting:

$$\cos(\alpha) = \frac{QM}{\cos(\alpha)}$$

Solving for $QM$:

$$QM = \cos(\alpha) \cdot \cos(\alpha) = \cos^2(\alpha)$$

Thus, the first way of finding $QM$ gives us $QM = \cos^2(\alpha)$.

**Second way:**

Since $M$ lies on the diameter $QR$ between $Q$ and $R$, we can express $QM$ as:

$$QM = QO + OM$$

where $QO$ is the distance from $Q$ to the center $O$, and $OM$ is the distance from the center $O$ to point $M$.

By the inscribed angle theorem, $\beta = 2\alpha$, where $\beta = \angle POR$ is the central angle and $\alpha = \angle PQR$ is the inscribed angle.

Now consider triangle $POM$, which is right-angled at $M$ (since $PM$ is perpendicular to $QR$). In this triangle:

- The angle at $O$ is $\beta = 2\alpha$

- The hypotenuse is $OP$

- The adjacent side to angle $\beta$ is $OM$

Therefore, $\cos(\beta) = \cos(2\alpha) = \frac{OM}{OP}$.

But $OP = \frac{1}{2}$ (since it is the radius, and the diameter $QR = 1$).

Substituting:

$$\cos(2\alpha) = \frac{OM}{1/2}$$

Solving for $OM$:

$$OM = \frac{\cos(2\alpha)}{2}$$

Now, to find $QO$: Since $Q$ is one endpoint of the diameter $QR$ and $O$ is the center of the circle, $QO$ is the radius. Since the diameter $QR = 1$, the radius is $\frac{1}{2}$.

Therefore:

$$QO = \frac{1}{2}$$

Now we can compute $QM$ using the second way:

$$QM = QO + OM = \frac{1}{2} + \frac{\cos(2\alpha)}{2} = \frac{1 + \cos(2\alpha)}{2}$$

Thus, the second way of finding $QM$ gives us $QM = \frac{1 + \cos(2\alpha)}{2}$.

**Completing the proof:**

Since we found $QM$ in two different ways, we can equate them:

$$\cos^2(\alpha) = \frac{1 + \cos(2\alpha)}{2}$$

This completes the proof!

### How we used some of the tips

**Use the defining features of the setup**: We leveraged key properties.

- $QR$ is a diameter (length 1)
- The circle has unit diameter, so radius is $\frac{1}{2}$
- $PM$ is perpendicular to $QR$
- $P$ lies on a semicircle, so $\angle QPR = 90°$ (Thales' theorem)
- The inscribed angle theorem: $\beta = 2\alpha$

**Give things (meaningful) names**: We used descriptive notation.

- $\alpha$ for the inscribed angle $\angle PQR$
- $\beta$ for the central angle $\angle POR$
- Points $P$, $Q$, $R$, $O$, $M$ with clear geometric meanings
- These names made it easy to refer to specific angles and lengths

**Try describing one object in two different ways**: This was the key strategy!

- We found $QM$ using triangle $PMQ$. $QM = \cos^2(\alpha)$
- We found $QM$ using the diameter $QM = QO + OM = \frac{1 + \cos(2\alpha)}{2}$
- Equating these two expressions gave us the desired identity
