# Conditional Expectation

## The Two Envelope Paradox

The two envelope paradox is a famous probability puzzle that challenges our intuition about conditional expectation and reveals subtle issues in reasoning about random variables.

You are given two envelopes. One contains twice as much money as the other. You pick one envelope at random, open it, and see that it contains $X. You are then given the choice to either:

1. Keep the envelope you have (containing $X)
2. Switch to the other envelope

**The question**: Should you switch?

At first glance, it seems like switching should be beneficial.

The other envelope contains either $X/2 or $2X, each with probability 1/2. The expected value of switching is:

$$E[\text{other envelope}] = \frac{1}{2} \cdot \frac{X}{2} + \frac{1}{2} \cdot 2X = \frac{X}{4} + X = \frac{5X}{4}$$

Since $\frac{5X}{4} > X$, you should always switch!

**The paradox**: This reasoning suggests you should always switch, regardless of which envelope you initially chose. But this is clearly wrong- if you always switch, you're back to a 50-50 choice!

The error lies in assuming that the other envelope contains $X/2$ or $2X$ with equal probability 1/2.

Let's be more careful about the random variables involved.

Let $A$ be the amount in the first envelope, and $2A$ be the amount in the second envelope. The envelopes are chosen randomly, so:

- $P(\text{you pick first}) = \frac{1}{2}$

- $P(\text{you pick second}) = \frac{1}{2}$

**Case 1**: You picked the first envelope (contains $A$)

- The other envelope contains $2A$

- If you switch, you gain $A$

**Case 2**: You picked the second envelope (contains $2A$)

- The other envelope contains $A$

- If you switch, you lose $A$

**Expected gain from switching**:

$$E[\text{gain from switching}] = \frac{1}{2} \cdot A + \frac{1}{2} \cdot (-A) = 0$$

**Conclusion**: There is no expected gain from switching. The correct strategy is to be indifferent between keeping and switching.

The original reasoning made a subtle error by treating $X$ as if it were independent of which envelope you chose. But $X$ is not independent- it depends on whether you picked the first or second envelope.

## Definition of Conditional Expectation

Conditional expectation is the expected value of a random variable given that we know the value of another random variable. It's a fundamental concept in probability theory that allows us to make predictions based on partial information.

### Discrete Case

For discrete random variables $X$ and $Y$, the conditional expectation of $X$ given $Y = y$ is:

$$E[X | Y = y] = \sum_x x \cdot P(X = x | Y = y) = \sum_x x \cdot \frac{P(X = x, Y = y)}{P(Y = y)}$$

**Key properties:**

- $E[X | Y = y]$ is a function of $y$

- It represents the average value of $X$ when we know $Y = y$

- If $X$ and $Y$ are independent, then $E[X | Y = y] = E[X]$ for all $y$

### Continuous Case

For continuous random variables $X$ and $Y$ with joint density $f_{X,Y}(x,y)$ and marginal density $f_Y(y)$, the conditional expectation of $X$ given $Y = y$ is:

$$E[X | Y = y] = \int_{-\infty}^{\infty} x \cdot f_{X|Y}(x|y) \, dx = \int_{-\infty}^{\infty} x \cdot \frac{f_{X,Y}(x,y)}{f_Y(y)} \, dx$$

where $f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$ is the conditional density of $X$ given $Y = y$.

## The Random Variable $E[X | Y]$ and its intuitive meaning

Conditional expectation is itself a random variable! We write $E[X | Y]$ to denote the random variable that takes the value $E[X | Y = y]$ when $Y = y$.

Conditional expectation answers the question: "What is the average value of $X$ when we know that $Y$ takes a specific value?"

**Example**: If $X$ is your test score and $Y$ is the number of hours you studied:

- $E[X | Y = 0]$ = average test score for students who didn't study

- $E[X | Y = 10]$ = average test score for students who studied 10 hours

- $E[X | Y]$ = a function that tells you the expected test score for any amount of study time