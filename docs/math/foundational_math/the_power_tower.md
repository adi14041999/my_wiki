# The Power Tower

## Introduction

A **power tower** (also called a **tetration**) is an expression of the form:

$$a^{a^{a^{\cdot^{\cdot^{\cdot}}}}}$$

where we have repeated exponentiation. Power towers are written from top to bottom, meaning we evaluate from the top down. Let's see what happens if we evaluate from bottom to top.

Consider a power tower of height 3: $a^{a^a}$

**Evaluating from top to bottom (correct way):**

$$a^{a^a} = a^{(a^a)}$$

We first compute $a^a$, then raise $a$ to that power.

**Evaluating from bottom to top (incorrect for power towers):**

If we evaluated from bottom to top, we would compute:

$$(a^a)^a = a^{a \cdot a} = a^{a^2}$$

This is just **regular exponentiation**— we're multiplying the exponents, which gives us $a$ raised to the power $a^n$.

## Notation

For a finite power tower of height $n$, we write:

$${}^{n}a = a^{a^{a^{\cdot^{\cdot^{\cdot^{a}}}}}} \quad \text{($n$ copies of $a$)}$$

A power tower can be defined recursively.

**Base case:**

$${}^{1}a = a$$

**Recursive case:**

$${}^{n}a = a^{({}^{n-1}a)} \quad \text{for } n > 1$$

In words, a power tower of height $n$ is $a$ raised to the power of a power tower of height $n-1$.

Let's compute some simple power towers.

**For $a = 2$:**

- ${}^{1}2 = 2 = 2$

- ${}^{2}2 = 2^2 = 4$

- ${}^{3}2 = 2^{2^2} = 2^4 = 16$

- ${}^{4}2 = 2^{2^{2^2}} = 2^{16} = 65,536$

**For $a = \sqrt{2}$:**

- ${}^{1}\sqrt{2} = \sqrt{2} \approx 1.414$

- ${}^{2}\sqrt{2} = (\sqrt{2})^{\sqrt{2}} \approx 1.633$

- ${}^{3}\sqrt{2} = (\sqrt{2})^{(\sqrt{2})^{\sqrt{2}}} \approx 1.761$