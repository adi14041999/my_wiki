# Multinomial Distribution

The multinomial distribution is a generalization of the binomial distribution to multiple categories or outcomes. It describes the probability distribution of counts across several mutually exclusive and exhaustive categories.

A random vector $\mathbf{X} = (X_1, X_2, \ldots, X_k)$ follows a **multinomial distribution** with parameters $n$ and $\mathbf{p} = (p_1, p_2, \ldots, p_k)$, denoted as $\mathbf{X} \sim \text{Multinomial}(n, \mathbf{p})$, if its joint probability mass function is:

$$P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}$$

where:

- $n$ is the total number of trials

- $x_i$ is the number of outcomes in category $i$ (with $x_i \geq 0$)

- $p_i$ is the probability of category $i$ (with $0 \leq p_i \leq 1$)

- The constraint $\sum_{i=1}^k x_i = n$ must hold

- The constraint $\sum_{i=1}^k p_i = 1$ must hold

The multinomial distribution models:

- **$n$ independent trials** where each trial results in exactly one of $k$ possible outcomes

- **Fixed probabilities** $p_1, p_2, \ldots, p_k$ for each outcome

- **Counts** $X_1, X_2, \ldots, X_k$ representing how many times each outcome occurred

When $k = 2$, the multinomial distribution reduces to the binomial distribution:

- $X_1 \sim \text{Binomial}(n, p_1)$

- $X_2 = n - X_1$ (since $X_1 + X_2 = n$)

- $p_2 = 1 - p_1$

## Marginal Distribution

Let's take a look at the marginal distribution of $X_i$, which is the $i$th component of $\mathbf{X}$. Were we to blindly apply the definition, we would have to sum the joint PMF over all components of $\mathbf{X}$ other than $X_i$. The prospect of $k-1$ summations is an unpleasant one, to say the least.

Fortunately, we can avoid tedious calculations if we instead use the story of the Multinomial distribution: $X_i$ is the number of objects in category $i$, where each of the $n$ objects independently belongs to category $i$ with probability $p_i$. Define success as landing in category $i$. Then we just have $n$ independent Bernoulli trials, so the marginal distribution of $X_i$ is $\text{Bin}(n, p_i)$.

**The marginals of a Multinomial are Binomial.** Specifically, if $\mathbf{X} \sim \text{Mult}_k(n, \mathbf{p})$, then $X_i \sim \text{Bin}(n, p_i)$.

## Merging Categories

More generally, whenever we merge multiple categories together in a Multinomial random vector, we get another Multinomial random vector. For example, suppose we randomly sample $n$ people in a country with 5 political parties (if the sampling is done without replacement, the $n$ trials are not independent, but independence is a good approximation as long as the population is large relative to the sample). Let $\mathbf{X} = (X_1, \ldots, X_5) \sim \text{Mult}_5(n, (p_1, \ldots, p_5))$ represent the political party affiliations of the sample, i.e., $X_j$ is the number of people in the sample who support party $j$.

Suppose that parties 1 and 2 are the dominant parties, while parties 3 through 5 are minor third parties. If we decide that instead of keeping track of all 5 parties, we only want to count the number of people in party 1, party 2, or "other", then we can define a new random vector that lumps all the third parties into one category:

$$\mathbf{Y} = (X_1, X_2, X_3 + X_4 + X_5)$$

By the story of the Multinomial,

$$\mathbf{Y} \sim \text{Mult}_3(n, (p_1, p_2, p_3 + p_4 + p_5))$$

Of course, this idea applies to merging categories in any Multinomial, not just in the context of political parties.

**General Rule:** If $\mathbf{X} \sim \text{Mult}_k(n, \mathbf{p})$, then for any distinct $i$ and $j$, $X_i + X_j \sim \text{Bin}(n, p_i + p_j)$. The random vector of counts obtained from merging categories $i$ and $j$ is still Multinomial. For example, merging categories 1 and 2 gives:

$$(X_1 + X_2, X_3, \ldots, X_k) \sim \text{Mult}_{k-1}(n, (p_1 + p_2, p_3, \ldots, p_k))$$

## Conditional Distribution

Suppose we get to observe $X_1$, the number of objects in category 1, and we wish to update our distribution for the other categories $(X_2, \ldots, X_k)$. One way to do this is with the definition of conditional PMF:

$$P(X_2 = n_2, \ldots, X_k = n_k|X_1 = n_1) = \frac{P(X_1 = n_1, X_2 = n_2, \ldots, X_k = n_k)}{P(X_1 = n_1)}$$

The numerator is the joint PMF of the Multinomial, and the denominator is the marginal PMF of $X_1$, both of which we have already derived. However, we prefer to use the Multinomial story to deduce the conditional distribution of $(X_2, \ldots, X_k)$ without algebra.

Given that there are $n_1$ objects in category 1, the remaining $n - n_1$ objects fall into categories 2 through $k$, independently of one another. By Bayes' rule (**Bayes' rule**: $P(\text{in category } j|\text{not in category } 1) = \frac{P(\text{in category } j \text{ AND not in category } 1)}{P(\text{not in category } 1)}$), the conditional probability of falling into category $j$ is:

$$P(\text{in category } j|\text{not in category } 1) = \frac{P(\text{in category } j)}{P(\text{not in category } 1)} = \frac{p_j}{p_2 + \cdots + p_k}$$

for $j = 2, \ldots, k$. This makes intuitive sense: the updated probabilities are proportional to the original probabilities $(p_2, \ldots, p_k)$, but these must be renormalized to yield a valid probability vector.

Putting it all together, we have the following result:

**If $\mathbf{X} \sim \text{Mult}_k(n, \mathbf{p})$, then**

$$(X_2, \ldots, X_k)|X_1 = n_1 \sim \text{Mult}_{k-1}(n - n_1, (p'_2, \ldots, p'_k))$$

**where $p'_j = \frac{p_j}{p_2 + \cdots + p_k}$.**

Finally, we know that components within a Multinomial random vector are dependent since they are constrained by $X_1 + \cdots + X_k = n$.