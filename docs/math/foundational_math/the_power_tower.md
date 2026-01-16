# The Power Tower

## Introduction

A **power tower** (also called a **tetration**) is an expression of the form:

$$a^{a^{a^{\cdot^{\cdot^{\cdot}}}}}$$

where we have repeated exponentiation. Power towers are written from top to bottom, meaning we evaluate from the top down.

## Notation

For a finite power tower of height $n$, we write:

$${}^{n}a = a^{a^{a^{\cdot^{\cdot^{\cdot^{a}}}}}} \quad \text{($n$ copies of $a$)}$$

For example:
- ${}^{1}a = a$
- ${}^{2}a = a^a$
- ${}^{3}a = a^{a^a}$
- ${}^{4}a = a^{a^{a^a}}$

## Examples

Let's compute some simple power towers:

**For $a = 2$:**
- ${}^{1}2 = 2 = 2$
- ${}^{2}2 = 2^2 = 4$
- ${}^{3}2 = 2^{2^2} = 2^4 = 16$
- ${}^{4}2 = 2^{2^{2^2}} = 2^{16} = 65,536$

**For $a = \sqrt{2}$:**
- ${}^{1}\sqrt{2} = \sqrt{2} \approx 1.414$
- ${}^{2}\sqrt{2} = (\sqrt{2})^{\sqrt{2}} \approx 1.633$
- ${}^{3}\sqrt{2} = (\sqrt{2})^{(\sqrt{2})^{\sqrt{2}}} \approx 1.761$

Notice that for $\sqrt{2}$, the values seem to be converging!

## The Infinite Power Tower

For an infinite power tower, we consider the limit:

$$y = a^{a^{a^{\cdot^{\cdot^{\cdot}}}}} = \lim_{n \to \infty} {}^{n}a$$

If this limit exists, we say the infinite power tower **converges**.

## Convergence Condition

The infinite power tower $a^{a^{a^{\cdot^{\cdot^{\cdot}}}}}$ converges if and only if:

$$e^{-e} \leq a \leq e^{1/e}$$

where $e \approx 2.718$ is Euler's number.

This means:
- For $a$ in the range $[e^{-e}, e^{1/e}] \approx [0.066, 1.445]$, the infinite power tower converges
- For $a > e^{1/e}$ or $a < e^{-e}$, the infinite power tower diverges

## Finding the Value

If the infinite power tower converges to $y$, then:

$$y = a^{a^{a^{\cdot^{\cdot^{\cdot}}}}} = a^y$$

Taking the natural logarithm of both sides:

$$\ln(y) = \ln(a^y) = y \ln(a)$$

Therefore:

$$y \ln(a) = \ln(y)$$

Or equivalently:

$$a = y^{1/y}$$

This gives us a way to find the value of the infinite power tower for a given $a$.

## Special Cases

**When $a = e^{1/e}$:**

The infinite power tower converges to $e$:

$$e^{1/e^{e^{1/e^{\cdot^{\cdot^{\cdot}}}}} = e$$

**When $a = \sqrt{2}$:**

The infinite power tower converges to $2$:

$$\sqrt{2}^{\sqrt{2}^{\sqrt{2}^{\cdot^{\cdot^{\cdot}}}}} = 2$$

We can verify this: if $y = 2$, then $a = y^{1/y} = 2^{1/2} = \sqrt{2}$ ✓

**When $a = 1$:**

The infinite power tower is simply $1$:

$$1^{1^{1^{\cdot^{\cdot^{\cdot}}}}} = 1$$