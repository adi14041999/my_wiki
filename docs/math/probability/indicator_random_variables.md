# Indicator Random Variables

**Indicator random variables** are one of the most powerful and elegant tools in probability theory. They provide a bridge between probability and expectation, making complex problems surprisingly simple.

An **indicator random variable** $I_A$ for an event $A$ is defined as:

$$I_A = \begin{cases} 
1 & \text{if event } A \text{ occurs} \\
0 & \text{if event } A \text{ does not occur}
\end{cases}$$

**Key properties**:

- **Binary values**: Only takes values 0 or 1

- **Event representation**: Directly represents whether an event occurs

- **Probability connection**: $E[I_A] = P(A)$

**Bridge Between Probability and Expectation**

The fundamental connection: **$E[I_A] = P(A)$**

**Proof**:

$$E[I_A] = 1 \cdot P(I_A = 1) + 0 \cdot P(I_A = 0) = 1 \cdot P(A) + 0 \cdot P(A^c) = P(A)$$

This simple result allows us to convert probability problems into expectation problems, which are often easier to solve.

**Linearity of Expectation**

Since indicators only take values 0 and 1, they work beautifully with linearity of expectation:

$$E[I_{A_1} + I_{A_2} + \cdots + I_{A_n}] = E[I_{A_1}] + E[I_{A_2}] + \cdots + E[I_{A_n}] = P(A_1) + P(A_2) + \cdots + P(A_n)$$

**Example: Birthday Problem**

**Problem**: In a group of $n$ people, what's the expected number of people with a birthday on January 1st?

**Solution using indicators**:

- Let $I_i$ be the indicator that person $i$ has a birthday on January 1st

- $E[I_i] = P(\text{person } i \text{ born on Jan 1}) = \frac{1}{365}$

- Total expected: $E[\sum_{i=1}^n I_i] = \sum_{i=1}^n E[I_i] = n \cdot \frac{1}{365}$

**Result**: We expect $\frac{n}{365}$ people to have a birthday on January 1st.

**Without using indicators**, we can solve this directly using the definition of expectation:

Let $X$ be the number of people with a birthday on January 1st.

$X$ follows a **binomial distribution**: $X \sim \text{Binomial}(n, \frac{1}{365})$

For $X \sim \text{Binomial}(n, p)$, we know $E[X] = np$.

Therefore:

$$E[X] = n \cdot \frac{1}{365} = \frac{n}{365}$$

**Result**: Same answer, different method! The indicator method breaks down the problem into individual components, while the direct method recognizes the overall distribution. Both are valid approaches that lead to the same mathematical result.

**Example**: A permutation of numbers 1 to $n$ has a **local maximum** at the $j$-th position if the number at the $j$-th position is bigger than both its neighbors. For the first and last positions, a local maximum exists if that number is bigger than its only neighbor. Given that all $n!$ permutations are equally likely, calculate the expected number of local maxima.

**Solution using Indicator Random Variables**

Let $I_j$ be the indicator that position $j$ has a local maximum.

**Case 1: Interior positions ($2 \leq j \leq n-1$)**

For position $j$ to be a local maximum:

- The number at position $j$ must be larger than the number at position $j-1$

- The number at position $j$ must be larger than the number at position $j+1$

**Probability calculation**:

- We need to choose 3 distinct numbers from $\{1, 2, \ldots, n\}$

- The middle number must be the largest of the three

- $P(I_j = 1) = \frac{1}{3}$ (by symmetry, any of the three numbers is equally likely to be largest)

**Case 2: Boundary positions ($j = 1$ or $j = n$)**

For position 1 to be a local maximum:

- The number at position 1 must be larger than the number at position 2

**Probability calculation**:

- We need to choose 2 distinct numbers from $\{1, 2, \ldots, n\}$

- The first number must be larger than the second

- $P(I_1 = 1) = \frac{1}{2}$ (by symmetry, either number is equally likely to be larger)

Similarly, $P(I_n = 1) = \frac{1}{2}$.

Using the bridge between Probability and Expectation for IRV:

- **Interior positions**: $E[I_j] = \frac{1}{3}$ for $j = 2, 3, \ldots, n-1$

- **Boundary positions**: $E[I_1] = E[I_n] = \frac{1}{2}$

**Use linearity of expectation**

$$E[\text{Total local maxima}] = E\left[\sum_{j=1}^n I_j\right] = \sum_{j=1}^n E[I_j]$$

$$= E[I_1] + \sum_{j=2}^{n-1} E[I_j] + E[I_n]$$

$$= \frac{1}{2} + (n-2) \cdot \frac{1}{3} + \frac{1}{2}$$

$$= 1 + (n-2) \cdot \frac{1}{3} = 1 + \frac{n-2}{3} = \frac{3 + n - 2}{3} = \frac{n+1}{3}$$

The expected number of local maxima in a random permutation of $\{1, 2, \ldots, n\}$ is:

$$E[\text{Local maxima}] = \frac{n+1}{3}$$

This problem demonstrates the power of indicator random variables in complex combinatorial problems where direct counting would be extremely difficult.