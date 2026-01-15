# Logarithms

## Introduction

Have you ever noticed a beautiful pattern when working with powers of 10?

- $10^1 = 10$ (one zero)

- $10^2 = 100$ (two zeros)

- $10^3 = 1,000$ (three zeros)

- $10^4 = 10,000$ (four zeros)

- $10^5 = 100,000$ (five zeros)

The exponent tells us exactly how many zeros appear in the result.

This is where logarithms come in. The **logarithm base 10** (written as $\log_{10}$ or simply $\log$) of a number answers the question: "What power do I need to raise 10 to in order to get this number?"

So when we say $\log_{10}(1000) = 3$, we're saying: "10 raised to the power of 3 equals 1000."

But here's the delightful connection: **The log base 10 of $10^n$ equals $n$, which is exactly the number of zeros in $10^n$.**

For example:

- $\log_{10}(10) = 1$ → 10 has 1 zero

- $\log_{10}(100) = 2$ → 100 has 2 zeros  

- $\log_{10}(1,000) = 3$ → 1,000 has 3 zeros

- $\log_{10}(10,000) = 4$ → 10,000 has 4 zeros

This elegant relationship makes logarithms incredibly useful for understanding the scale of numbers, especially when dealing with very large or very small values.

## Definition

The logarithm is the inverse operation of exponentiation. For any positive real base $b$ (where $b \neq 1$) and any positive real number $x$, the logarithm base $b$ of $x$ is defined as:

$$\log_b(x) = y \quad \text{if and only if} \quad b^y = x$$

where $y$ is a real number.

In words: $\log_b(x)$ answers the question "To what power must I raise $b$ to get $x$?"

**Examples**

- $\log_{10}(100) = 2$ because $10^2 = 100$
- $\log_2(8) = 3$ because $2^3 = 8$
- $\ln(e^5) = 5$ because $e^5 = e^5$
- $\log_2(16) = 4$ because $2^4 = 16$
- $\ln(e) = 1$ because $e^1 = e$
- $\log_{10}(0.01) = -2$ because $10^{-2} = 0.01$

## Product Rule

The product rule for logarithms states that the logarithm of a product equals the sum of the logarithms:

$$\log_b(xy) = \log_b(x) + \log_b(y)$$

Let's derive this rule using the definition of logarithms and the properties of exponents.

Let $m = \log_b(x)$ and $n = \log_b(y)$.

By the definition of logarithms:

- $b^m = x$

- $b^n = y$

Multiply these two equations:

$$b^m \cdot b^n = x \cdot y$$

Apply the exponent rule for multiplication (add the exponents):

$$b^{m+n} = xy$$

By the definition of logarithms, if $b^{m+n} = xy$, then:

$$\log_b(xy) = m + n$$

Substitute back $m = \log_b(x)$ and $n = \log_b(y)$:

$$\log_b(xy) = \log_b(x) + \log_b(y)$$

**Example:** Using base 10

- $\log_{10}(100 \cdot 1000) = \log_{10}(100) + \log_{10}(1000) = 2 + 3 = 5$
- We can verify: $\log_{10}(100,000) = 5$ because $10^5 = 100,000$ ✓

**Example:** Using base 2

- $\log_2(4 \cdot 8) = \log_2(4) + \log_2(8) = 2 + 3 = 5$
- We can verify: $\log_2(32) = 5$ because $2^5 = 32$ ✓

**Example:** Using natural logarithm

- $\ln(e^2 \cdot e^3) = \ln(e^2) + \ln(e^3) = 2 + 3 = 5$
- We can verify: $\ln(e^5) = 5$ because $e^5 = e^5$ ✓

**Example:** With three factors

- $\log_{10}(10 \cdot 100 \cdot 1000) = \log_{10}(10) + \log_{10}(100) + \log_{10}(1000) = 1 + 2 + 3 = 6$
- We can verify: $\log_{10}(1,000,000) = 6$ because $10^6 = 1,000,000$ ✓

**Example:** With fractions

- $\log_2(8 \cdot \frac{1}{4}) = \log_2(8) + \log_2\left(\frac{1}{4}\right) = 3 + (-2) = 1$
- We can verify: $\log_2(2) = 1$ because $2^1 = 2$ ✓

## Power Rule

The power rule for logarithms allows you to move an exponent from inside the logarithm to the front as a multiplier:

$$\log_b(x^n) = n \cdot \log_b(x)$$

Let's derive this rule using the definition of logarithms and the properties of exponents.

Let $m = \log_b(x)$.

By the definition of logarithms:

$$b^m = x$$

Raise both sides to the power $n$:

$$(b^m)^n = x^n$$

Apply the exponent rule for powers (multiply the exponents):

$$b^{mn} = x^n$$

By the definition of logarithms, if $b^{mn} = x^n$, then:

$$\log_b(x^n) = mn$$

Substitute back $m = \log_b(x)$:

$$\log_b(x^n) = n \cdot \log_b(x)$$

**Example:** Using base 10

- $\log_{10}(100^3) = 3 \cdot \log_{10}(100) = 3 \cdot 2 = 6$
- We can verify: $\log_{10}(1,000,000) = 6$ because $10^6 = 1,000,000$ ✓

**Example:** Using base 2

- $\log_2(8^2) = 2 \cdot \log_2(8) = 2 \cdot 3 = 6$
- We can verify: $\log_2(64) = 6$ because $2^6 = 64$ ✓

## Change of Base Formula

Using the power rule, we can derive an important relationship between logarithms of different bases. This relationship shows that:

$$\frac{1}{\log_b(a)} = \log_a(b)$$

Let's derive this using the power rule.

Let $m = \log_b(a)$.

By the definition of logarithms:

$$b^m = a$$

Now, let's find $\log_a(b)$. We want to express $b$ in terms of base $a$.

From $b^m = a$, we can write:

$$b = a^{1/m}$$

By the definition of logarithms, if $b = a^{1/m}$, then:

$$\log_a(b) = \frac{1}{m}$$

Substitute back $m = \log_b(a)$:

$$\log_a(b) = \frac{1}{\log_b(a)}$$

Taking the reciprocal of both sides:

$$\frac{1}{\log_b(a)} = \log_a(b)$$

This shows that the reciprocal of a logarithm equals the logarithm with the base and argument swapped.

**Example:** Converting between bases

- $\frac{1}{\log_2(8)} = \log_8(2)$
- We can verify: $\log_2(8) = 3$, so $\frac{1}{3} = \log_8(2)$
- Check: $8^{1/3} = 2$ ✓

**Example:** Using base 10 and base 2

- $\frac{1}{\log_{10}(100)} = \log_{100}(10)$
- We can verify: $\log_{10}(100) = 2$, so $\frac{1}{2} = \log_{100}(10)$
- Check: $100^{1/2} = 10$ ✓

Using the relationship we just established, we can prove the general change of base formula:

$$\log_b(x) = \frac{\log_a(x)}{\log_a(b)}$$

Let's prove this by changing base from $a$ to $b$.

Let $y = \log_a(x)$.

By the definition of logarithms:

$$a^y = x$$

Now, let's express this in terms of base $b$. We know from the previous result that:

$$\log_a(b) = \frac{1}{\log_b(a)}$$

But more directly, let's take the logarithm base $b$ of both sides of $a^y = x$:

$$\log_b(a^y) = \log_b(x)$$

Apply the power rule to the left side:

$$y \cdot \log_b(a) = \log_b(x)$$

Substitute back $y = \log_a(x)$:

$$\log_b(a) \cdot \log_a(x) = \log_b(x)$$

Now, divide both sides by $\log_b(a)$:

$$\log_a(x) = \frac{\log_b(x)}{\log_b(a)}$$

Or equivalently, using the reciprocal relationship:

$$\log_b(x) = \frac{\log_a(x)}{\log_a(b)}$$

This shows that to convert a logarithm from base $a$ to base $b$, we divide the logarithm of the argument by the logarithm of the original base, both in the new base.

**Example:** Converting to base 8

- $\log_2(64) = \frac{\log_8(64)}{\log_8(2)} = \frac{2}{\log_8(2)}$
- We can verify: $\log_2(64) = 6$ because $2^6 = 64$ ✓
- Also: $\log_8(2) = \frac{1}{3}$, so $\frac{2}{1/3} = 6$ ✓

**Example:** Evaluate the sum

$$\frac{1}{\log_2(100!)} + \frac{1}{\log_3(100!)} + \frac{1}{\log_4(100!)} + \cdots + \frac{1}{\log_{100}(100!)}$$

**Solution:**

Using the reciprocal relationship $\frac{1}{\log_b(a)} = \log_a(b)$, we can rewrite each term:

$$\frac{1}{\log_2(100!)} = \log_{100!}(2)$$

$$\frac{1}{\log_3(100!)} = \log_{100!}(3)$$

$$\frac{1}{\log_4(100!)} = \log_{100!}(4)$$

$$\vdots$$

$$\frac{1}{\log_{100}(100!)} = \log_{100!}(100)$$

So the sum becomes:

$$\log_{100!}(2) + \log_{100!}(3) + \log_{100!}(4) + \cdots + \log_{100!}(100)$$

Using the product rule for logarithms:

$$\log_{100!}(2) + \log_{100!}(3) + \cdots + \log_{100!}(100) = \log_{100!}(2 \cdot 3 \cdot 4 \cdots 100)$$

Since $2 \cdot 3 \cdot 4 \cdots 100 = 100!$, we have:

$$\log_{100!}(100!) = 1$$

Therefore, the sum equals **1**.

## What is special about the natural log?

The natural logarithm $\ln(x)$ has many special properties, but one of the most beautiful is its connection to infinite series. Let's explore a remarkable result.

Consider the infinite series:

$$1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \frac{1}{5} - \frac{1}{6} + \cdots$$

This is called the **alternating harmonic series**. Remarkably, this series converges to $\ln(2)$:

$$1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \frac{1}{5} - \frac{1}{6} + \cdots = \ln(2)$$

Now consider the **harmonic series** (without alternating signs):

$$1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \frac{1}{5} + \cdots + \frac{1}{N}$$

Unlike the alternating harmonic series, this series diverges (grows without bound) as $N$ increases. 

Precisely, as $N$ approaches infinity, the difference between the harmonic sum and $\ln(N)$ approaches a constant called **Euler's constant** (denoted $\gamma$):

$$\lim_{N \to \infty} \left(1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{N} - \ln(N)\right) = \gamma \approx 0.57721\ldots$$

So for large $N$:

$$1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{N} \approx \ln(N) + \gamma$$

**Example:** Expressed as a power of 10, what is the smallest $N$ such that the harmonic series is larger than 1,000,000?

**Solution:**

We want to find the smallest $N$ such that:

$$1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{N} > 1,000,000$$

Using the approximation for large $N$:

$$\ln(N) + \gamma \approx 1,000,000$$

Since $\gamma \approx 0.577$ is much smaller than 1,000,000, we can approximate:

$$\ln(N) \approx 1,000,000$$

To express $N$ as a power of 10, we use the change of base formula. Recall that $\ln(N) = \frac{\log_{10}(N)}{\log_{10}(e)}$, or equivalently:

$$\ln(N) = \log_{10}(N) \cdot \ln(10)$$

So:

$$\log_{10}(N) \cdot \ln(10) \approx 1,000,000$$

Since $\ln(10) \approx 2.302585$, we have:

$$\log_{10}(N) \approx \frac{1,000,000}{2.302585} \approx 434,294.48$$

Therefore:

$$N \approx 10^{434,294.48} \approx 10^{434,294}$$

So the smallest $N$ (expressed as a power of 10) such that the harmonic series exceeds 1,000,000 is approximately **$10^{434,294}$**.

This demonstrates the extremely slow growth of the harmonic series—it takes an astronomically large number of terms to reach even 1,000,000!