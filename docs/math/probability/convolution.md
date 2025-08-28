# Convolution

A convolution is a sum of independent random variables. We often add independent random variables because the sum is a useful summary of an experiment (in $n$ Bernoulli trials, we may only care about the total number of successes), and because sums lead to averages, which are also useful (in $n$ Bernoulli trials, the proportion of successes).

The main task is to determine the distribution of $T = X + Y$, where $X$ and $Y$ are independent random variables whose distributions are known.

The distribution of $T$ is found using a convolution sum or integral. As we'll see, a convolution sum is nothing more than the law of total probability, conditioning on the value of either $X$ or $Y$; a convolution integral is analogous.

Let $X$ and $Y$ be independent random variables and $T = X + Y$ be their sum.

**Discrete Case:** If $X$ and $Y$ are discrete, then the PMF of $T$ is:

$$P(T = t) = \sum_x P(Y = t - x)P(X = x) = \sum_y P(X = t - y)P(Y = y)$$

**Continuous Case:** If $X$ and $Y$ are continuous, then the PDF of $T$ is:

$$f_T(t) = \int_{-\infty}^{\infty} f_Y(t - x)f_X(x)dx = \int_{-\infty}^{\infty} f_X(t - y)f_Y(y)dy$$

**Proof:** For the discrete case, we use the Law of Total Probability (LOTP), conditioning on $X$:

$$P(T = t) = \sum_x P(X + Y = t|X = x)P(X = x) = \sum_x P(Y = t - x|X = x)P(X = x) = \sum_x P(Y = t - x)P(X = x)$$

The last equality follows from the independence of $X$ and $Y$. Conditioning on $Y$ instead, we obtain the second formula for the PMF of $T$.