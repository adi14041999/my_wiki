# Introduction to Machine Learning

Machine Learning (ML) is a field of study in Artificial Intelligence concerned with the development and study of systems comprising of **statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit programming language instructions**. These systems are generally called as Machine Learning Models, or ML models.

In ML, we model the relationship between independent variables (also called **features**) and a dependent variable we want to predict. Machine Learning consists of a lot of different statistical algorithms and methods. Deciding which method (example, k-means versus hierarchical clustering) to use often means just trying it and seeing how well it performs. By performance, we mean **accuracy in predictions**.

When a Machine Learning method fits the training data really well but makes poor predictions, we say that it is **overfit** to the training data.

## Cross Validation

When building a Machine Learning model, we often split the available data into three parts:

- The training set is used to fit the model.
- The validation set is used to tune hyperparameters and make design choices.
- The test set is held back until the end and used only once to estimate how well the final chosen model performs on truly unseen data.

### Stopping Training during Cross Validation

For models that train iteratively, such as neural networks or gradient boosting models, we also need a rule for deciding when training should stop. This rule can be a fixed number of epochs, a convergence condition, or early stopping.

When using cross validation, the stopping rule should be handled inside each fold. First, the overall training data is split into outer cross validation folds. For each fold, one part becomes the fold's training data and another part becomes the fold's validation data. The fold's validation data should be used only to evaluate that fold's trained model.

If we want to use early stopping, we should create a smaller inner validation split from the fold's training data. The model trains on the remaining inner training data, and training stops when performance on the inner validation split stops improving. After training stops, the resulting model is evaluated on the fold's validation data.

The separate test set remains untouched until the very end. It should not be used to choose hyperparameters, compare models, stop training, or make any other modeling decision.

### Data Leakage

Data leakage happens when information from outside the current training data is accidentally used to build a model.

For example, suppose we are trying to predict whether a loan applicant will default on a loan. If the training data includes a feature like "number of missed payments in the next six months", the model may appear very accurate. But those facts are only known after the loan has already been issued. At the time we need to make the prediction, that information would not exist yet.

Data leakage can also happen during cross validation if preprocessing is done before the data is split into folds. For example, if we scale numeric features, fill in missing values, or select features using the full dataset before cross validation, information from the validation folds can influence the training folds.

!!! note "Fit preprocessing on the training set only"
    Compute scaling statistics (**mean**, **standard deviation**, min/max, etc.) on the **training set**, then apply those same parameters to validation and test data. If you **scale the full dataset before splitting**, the statistics leak information from the **test set** into training (indirect access to held-out examples). Reported metrics become **overly optimistic**. The correct pipeline is: **split first**, fit the scaler on **train**, transform **train / val / test** with that fitted scaler.

    **Why not scale each split with its own statistics?** The model learns weights in the coordinate system defined by **training** preprocessing. At inference (and on val/test), new points must enter the network in that **same** coordinate system (subtract the **training** mean, divide by the **training** std). Recomputing mean/std on val or test would (a) **peek** at held-out distributions (leakage on test), and (b) put inputs on a **different scale** than the weights expect, so validation metrics would not reflect real deployment behavior. The fixed scaler is part of the trained pipeline, like saved model weights.

## Fundamental concepts in Statistics

Models approximate reality to let us make predictions. In ML, we build models by using algorithms on Training Data. Statistics can be used to determine if a model is useful or believable. Statistics provides us with a set of tools to quantify the variation that we find in everything and, for the purposes of ML, helps us quantify how confident we should be in those predictions.

### Histograms

Histograms are one of the most basic, but surprisingly useful, statistical tools that we can use to gain insights into data. Instead of stacking measurements that are exactly the same, we divide the range of values into bins and stack the measurements that fall in the same bin. 

If we want to estimate the probability that the next measurement will be in a particular bin, we count the number of measurements in that bin, and divide by the total number of measurements. However, the confidence we have in this estimate depends on the number of measurements. Generally speaking, the more measurements you have, the more confidence you can have in the estimate. However, sometimes getting more measurements can be expensive, or take a lot of time, or both. The good news is that we can solve this
problem with a **Probability Distribution**.

Instead of collecting a ton of data to make a histogram (and worrying about blank bins) calculating probabilities, we can let mathematical equations do all of the hard work for us.

### The Binomial Distribution

For a deeper technical treatment, see:

- [Random Variables – Binomial Distribution](../../math/probability/random_variables.md#binomial-distribution) — covers the PMF, parameters, and addition property
- [Expectation – Binomial Random Variable](../../math/probability/expectation.md#expectation-of-a-binomial-random-variable) — derives $E[X] = np$ using linearity of expectation

First, let’s imagine we’re walking down the street and we ask the first three people we meet if they prefer pumpkin pie or
blueberry pie. Based on our extensive experience judging pie contests, we know that $70%$ of people prefer pumpkin pie, while $30%$ prefer blueberry pie. 

So now let’s calculate the probability of observing that the first two people prefer pumpkin pie and the third person prefers blueberry. The probability that the first person will prefer pumpkin pie is $0.7$. The probability that the first two people will prefer pumpkin pie is $0.49$. and the probability that the first two people will prefer pumpkin pie and the third person prefers blueberry is $0.147$. 

It could have just as easily been the case that the first person said they prefer blueberry and the last two said they prefer pumpkin. Likewise, if only the second person said they prefer blueberry, we would multiply the numbers together in a different order and still get $0.147$.

That means that the probability of observing that two out of three people prefer pumpkin pie is the sum of the three possible arrangements of people’s pie preferences. However, things quickly get tedious when we start asking more people which pie they prefer. For example, if we wanted to calculate the probability of observing that two out of four people prefer pumpkin pie, we have to calculate and sum the individual probabilities from six different arrangements.

So, instead of drawing out different arrangements of pie slices, we can use the equation for the Binomial Distribution to calculate the probabilities directly. We'll use **the Binomial Distribution to calculate the probabilities in any situation that has binary outcomes**, like wins and losses, yeses and noes, or successes and failures.

The probability of observing exactly $k$ successes in $n$ independent trials is:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Where:

- $n$ = total number of trials
- $k$ = number of successes we want
- $p$ = probability of success on a single trial
- $(1-p)$ = probability of failure on a single trial
- $\binom{n}{k} = \dfrac{n!}{k!(n-k)!}$ = the binomial coefficient, counting the number of ways to arrange $k$ successes among $n$ trials

Notice the formula is a product of three factors: $\underbrace{\binom{n}{k}}_{\text{# of ways}} \times \underbrace{p^k}_{\text{"yes" factor}} \times \underbrace{(1-p)^{n-k}}_{\text{"no" factor}}$. The "yes" factor gives the probability of getting $k$ successes, the "no" factor gives the probability of the remaining $n-k$ failures, and the binomial coefficient counts how many distinct orderings of those outcomes exist.

**Example:** Three out of five people prefer pumpkin pie.

With $n = 5$, $k = 3$, and $p = 0.7$:

$$P(X = 3) = \binom{5}{3} (0.7)^3 (0.3)^2$$

$$= \frac{5!}{3! \cdot 2!} \times 0.343 \times 0.09$$

$$= 10 \times 0.03087$$

$$= 0.3087$$

There is a **30.87% chance** that exactly 3 out of 5 people prefer pumpkin pie.

### The Poisson Distribution

For a deeper technical treatment, see:

- [Poisson Distribution](../../math/probability/poisson_distribution.md) — covers the PMF, parameter $\lambda$, and key properties

If you can read, on average, $10$ pages of a book in an hour, then you can use the Poisson Distribution to calculate the probability that in the next hour, you’ll read exactly $8$ pages.

The probability of observing exactly $k$ events in a fixed interval, given an average rate of $\lambda$ events per interval, is:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Where:

- $k$ = the number of events we want the probability for
- $\lambda$ = the average rate of events per interval (the mean)
- $e \approx 2.71828$ = Euler’s number
- $k!$ = $k$ factorial, the number of ways to order $k$ events

**Example:** Probability of reading exactly 8 pages in the next hour.

With $\lambda = 10$ (average pages per hour) and $k = 8$:

$$P(X = 8) = \frac{10^8 \cdot e^{-10}}{8!}$$

$$= \frac{100{,}000{,}000 \times 0.0000454}{40{,}320}$$

$$= \frac{4{,}540}{40{,}320}$$

$$\approx 0.1126$$

There is an **11.26% chance** of reading exactly 8 pages in the next hour.

### The Normal (Gaussian) Distribution

For a deeper technical treatment, see:

- [Normal Distribution](../../math/probability/normal_distribution.md) — covers the PDF, parameters $\mu$ and $\sigma^2$, and key properties
- [Multivariate Normal Distribution](../../math/probability/multivariate_normal_distribution.md) — generalizes to multiple dimensions with mean vectors and covariance matrices
- [Central Limit Theorem](../../math/probability/central_limit_theorem.md) — explains why the normal distribution appears so often: the sum of $n$ i.i.d. (independent and identically distributed) random variables converge to it as $n \to \infty$

!!! note "A note on i.i.d."
    A set of random variables is i.i.d. (independent and identically distributed) if two conditions hold. First, each variable is **independent**— knowing the value of one tells you nothing about the others. Second, each variable is **identically distributed**— they all follow the same probability distribution with the same parameters. In machine learning, we almost always assume our training examples are i.i.d.— each data point is an independent draw from the same underlying distribution.

The Normal distribution (also called the Gaussian distribution) is the most important distribution in statistics and machine learning. It describes a symmetric, bell-shaped curve centered at a mean $\mu$, with spread controlled by the standard deviation $\sigma$.

A random variable $X$ follows a Normal distribution, written $X \sim \mathcal{N}(\mu, \sigma^2)$, with probability density function:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

Where:

- $\mu$ = the mean, controlling where the curve is centered
- $\sigma^2$ = the variance, controlling how wide or narrow the curve is ($\sigma$ is the standard deviation)
- $e \approx 2.71828$ = Euler's number
- The factor $\frac{1}{\sigma\sqrt{2\pi}}$ is a normalizing constant that ensures the total area under the curve equals 1

The special case $\mu = 0$, $\sigma^2 = 1$ is called the **standard normal distribution**, written $Z \sim \mathcal{N}(0, 1)$.

### SSR, MSE and ${R^2}$

We need to quantify the quality of a model's predictions. One way quantify the quality of a model's predictions is to calculate the
**Sum of the Squared Residuals**.

The Sum of Squared Residuals (SSR) is the sum of all squared differences between the observed and predicted values.

$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2,$$

were $y_i$ is the observed value, $\hat{y}_i$ is the predicted value, and $n$ is the number of data points.

Sum of the Squared Residuals (SSR), although great, is not super easy to interpret because it depends, in part, on how much data you have. One way to compare the two models that may be fit to different-sized datasets is to calculate the **Mean Squared Error (MSE)**, which is simply the average of the SSR.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

So, unlike the SSR, which increases when we add more data to the model, the MSE can increase or decrease depending on the average residual, which gives us a better sense of how the model is performing overall.

Unfortunately, MSEs are still difficult to interpret on their own because **the maximum values depend on the scale of the data**. For example, residuals (the differences $y_i - \hat{y}_i$) of $1, -3, 2$ mm give an $\text{MSE} = 4.7$; but express the same data in meters ($0.001, -0.003, 0.002$) and the $\text{MSE}$ drops to $4.7 \times 10^{-6}$. Same data, wildly different numbers. Both $\text{SSR}$ and $\text{MSE}$ can be used to calculate $R^2$, which is independent of both dataset size and scale.

**$R^2$** measures how much better the fitted model is at predicting $y$ compared to simply predicting the mean $\bar{y}$ for every data point. It is defined as:

$$R^2 = \frac{\text{SSR}(\text{mean}) - \text{SSR}(\text{fit})}{\text{SSR}(\text{mean})}$$

Where:

- $\text{SSR}(\text{mean}) = \sum_{i=1}^{n}(y_i - \bar{y})^2$ is the sum of squared residuals around the mean line
- $\text{SSR}(\text{fit}) = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ is the sum of squared residuals around the fitted model
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ is the mean of the observed values

The numerator is the improvement in fit over the mean baseline. Dividing by $\text{SSR}(\text{mean})$ normalises it to a $[0, 1]$ scale: $R^2 = 1$ means a perfect fit, $R^2 = 0$ means the model is no better than the mean.

**Example**: Suppose we have three observations $y = \{2, 4, 6\}$ with $\bar{y} = 4$, and our model predicts $\hat{y} = \{2.5, 4, 5.5\}$.

$$\text{SSR}(\text{mean}) = (2-4)^2 + (4-4)^2 + (6-4)^2 = 4 + 0 + 4 = 8$$

$$\text{SSR}(\text{fit}) = (2-2.5)^2 + (4-4)^2 + (6-5.5)^2 = 0.25 + 0 + 0.25 = 0.5$$

$$R^2 = \frac{8 - 0.5}{8} = \frac{7.5}{8} = 0.9375$$

**When $\text{SSR}(\text{mean}) = \text{SSR}(\text{fit})$**: the numerator is $0$, so $R^2 = 0$. The fitted model does no better than just predicting $\bar{y}$ for every point— it has learned nothing useful from the data.

**When $\text{SSR}(\text{fit}) = 0$**: every predicted value equals the observed value ($\hat{y}_i = y_i$ for all $i$), so $R^2 = \frac{\text{SSR}(\text{mean})}{\text{SSR}(\text{mean})} = 1$. The model fits the data perfectly.

**Any two data points always give $R^2 = 1$**: two points uniquely determine a line, so the fitted line passes exactly through both points, giving $\text{SSR}(\text{fit}) = 0$ and therefore $R^2 = 1$. For example, take $(x_1, y_1) = (1, 2)$ and $(x_2, y_2) = (3, 6)$:

$$\bar{y} = \frac{2 + 6}{2} = 4$$

$$\text{SSR}(\text{mean}) = (2 - 4)^2 + (6 - 4)^2 = 4 + 4 = 8$$

The fitted line through both points predicts $\hat{y}_1 = 2$ and $\hat{y}_2 = 6$ exactly, so:

$$\text{SSR}(\text{fit}) = (2 - 2)^2 + (6 - 6)^2 = 0$$

$$R^2 = \frac{8 - 0}{8} = 1$$

This is why $R^2$ alone is not a reliable measure of model quality— a model can achieve $R^2 = 1$ just by having as many parameters as data points, without learning anything meaningful. And because a small amount of random data can have a high $R^2$, any time we see a trend in a small dataset it is difficult to have confidence that a high $R^2$ value is not due to random chance. In contrast, when we see a trend in a large amount of data we can, intuitively, have more confidence that a large $R^2$ is not due to random chance because the data are not scattered around randomly like we might expect.

!!! note
    $R^2$ can equivalently be calculated using the Mean Squared Residuals instead of the SSR:

    $$R^2 = \frac{\text{MSR}(\text{mean}) - \text{MSR}(\text{fit})}{\text{MSR}(\text{mean})}$$

    This works because both the numerator and denominator of the SSR formula are divided by the same $n$, which cancels out and leaves the value of $R^2$ unchanged.

## Gradient Descent

For a deeper technical treatment, see:

- [Gradients, Local Approximations, and Gradient Descent](../../math/multivariate_calculus/gradients_local_approximations_and_gradient_descent.md) — covers the gradient, linear approximation, and gradient descent from a mathematical perspective including the proof that the gradient is normal to contours
- [Optimization](../../ai/deep_learning_for_computer_vision/optimization.md) — covers gradient descent in the context of neural networks, including numerical vs. analytic gradient computation, mini-batch gradient descent, and practical strategies
- [Backpropagation](../../ai/deep_learning_for_computer_vision/backpropagation.md) — covers how gradients are computed efficiently through a network using the chain rule, including staged computation and vectorized operations
- [Backpropagation for a Linear Layer](../../ai/deep_learning_for_computer_vision/backpropagation_for_a_linear_layer.md) — derives the backward pass and gradient computation for a linear layer explicitly using the chain rule

A major part of ML is optimizing a model’s fit to the data. Sometimes this can be done with an analytical solution, but it’s not always possible.

Gradient Descent is an iterative algorithm that incrementally steps toward an optimal solution and is used in a very wide variety of situations.

Gradient Descent starts with an initial guess of the model parameters. GD then iteratively minimizes the Loss (or Cost) Function by taking steps toward the optimal model parameters, one at a time. It does so until it finds an optimal solution or reaches a maximum number of steps.

## Likelihood vs. Probability

Before introducing Logistic Regression, it helps to understand the distinction between **probability** and **likelihood**— two concepts that are related but mean different things.

**Probability** asks: given fixed parameters, what is the chance of observing this data? For example, if we know a coin has a 70% chance of heads ($p = 0.7$), we can ask: what is the probability of seeing 8 heads in 10 flips?

**Likelihood** flips this around: given fixed observed data, how plausible are different parameter values? We observe 8 heads in 10 flips and ask: which value of $p$ makes this data most probable? The likelihood $L(p)$ treats the data as fixed and $p$ as the variable.

The key distinction: probability sums (or integrates) to 1 over all possible outcomes; likelihood does **not** sum to 1 over all parameter values— it is not a probability distribution over parameters. For a deeper treatment of this, see [Continuous Distributions – Is $L(h)$ a probability distribution?](../../math/probability/continuous_distributions.md#is-lh-a-probability-distribution) and [Bayes' Rule – Likelihood](../../math/probability/bayes_rule.md).

**Maximum Likelihood Estimation (MLE)** is the principle of choosing the parameters that maximise the likelihood of the observed data— i.e., find the $\theta$ that makes the data we saw as probable as possible.

## Logistic Regression

Logistic Regression is a fundamental supervised machine learning algorithm used for classification. Instead of predicting continuous numbers (like linear regression), it calculates the probability that a given data point belongs to a specific category.

Rather than minimizing SSR like linear regression, logistic regression uses **Maximum Likelihood Estimation**— we choose the model parameters $\theta$ that maximize the likelihood of the observed labels given the data. In practice, it is more convenient to maximize the **log-likelihood** (since the log turns a product of probabilities into a sum, which is easier to optimize):

$$\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^{n} \log P(y_i \mid x_i, \theta)$$

Where:

- $\hat{\theta}$ = the estimated parameters we are solving for (e.g. the weights of the model)
- $\arg\max_{\theta}$ = "find the value of $\theta$ that maximizes the expression" — not the maximum value itself, but the $\theta$ that achieves it
- $\sum_{i=1}^{n}$ = sum over all $n$ training examples
- $\log P(y_i \mid x_i, \theta)$ = the log-probability that the model assigns to the true label $y_i$ of example $i$, given its features $x_i$ and the current parameters $\theta$

Intuitively, for each training example we ask: how confident is the model in the correct answer? We want that confidence to be as high as possible across all examples. Summing the log-probabilities (rather than multiplying the raw probabilities) avoids numerical underflow when dealing with many small numbers, and turns the product of independent probabilities into a more tractable sum.

**How does the model assign probabilities?** Logistic regression first computes a raw score (called the **logit**) as a linear combination of the input features:

$$z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_m x_m = \theta^T x$$

This score $z$ can be any real number. To convert it into a probability between 0 and 1, it is passed through the **sigmoid function**:

$$P(y = 1 \mid x, \theta) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid squashes any value of $z$ into the range $(0, 1)$: large positive $z$ gives a probability close to 1, large negative $z$ gives a probability close to 0, and $z = 0$ gives exactly 0.5. The probability of the negative class is then simply $P(y = 0 \mid x, \theta) = 1 - \sigma(z)$.

Because there is no closed-form solution for $\hat{\theta}$, we use **Gradient Descent** — iteratively stepping in the direction that increases the log-likelihood until convergence. This makes logistic regression a direct application of the GD framework described above, just with a different objective function.

In the case of **stochastic mini-batch GD**, rather than summing over all $n$ training examples at each step, we sum the log-probabilities over only the samples in the current batch of size $B$:

$$\mathcal{L}(\theta) = \sum_{i \in \text{batch}} \log P(y_i \mid x_i, \theta)$$

We then compute the gradient of this batch log-likelihood and take one parameter update step. In the next iteration, a new batch is sampled and the process repeats. Each batch gives a noisy but unbiased estimate of the true full-dataset gradient. This is why mini-batch GD is faster per step than full-batch GD and often generalizes better, as the noise in the gradient acts as a form of regularization.

## Naive Bayes

For the foundational theory, see [Bayes' Rule](../../math/probability/bayes_rule.md), which covers how prior beliefs are updated with evidence, and [Continuous Distributions](../../math/probability/continuous_distributions.md) for the treatment of likelihood.

**Where does the name come from?**

- **Bayes**— the classifier is built on Bayes' theorem, which gives us a way to compute $P(\text{class} \mid \text{features})$— the probability of a class given the observed features. Bayes' theorem states:

$$P(C \mid x_1, x_2, \ldots, x_m) = \frac{P(x_1, x_2, \ldots, x_m \mid C) \cdot P(C)}{P(x_1, x_2, \ldots, x_m)}$$

where $P(C)$ is the **prior** (how common is the class?), $P(x_1, \ldots, x_m \mid C)$ is the **likelihood** (how probable are these features given the class?), and the left side is the **posterior** (what we want).

- **Naive**— computing $P(x_1, x_2, \ldots, x_m \mid C)$ exactly requires knowing the joint distribution of all features, which is intractable for high-dimensional data. The "naive" assumption is that all features are **conditionally independent** given the class:

$$P(x_1, x_2, \ldots, x_m \mid C) = \prod_{j=1}^{m} P(x_j \mid C)$$

This is almost never strictly true in practice (e.g. the words "New" and "York" in a document are not independent), but the classifier works surprisingly well despite this assumption. The denominator $P(x_1, \ldots, x_m)$ is constant across classes and can be ignored when comparing classes, so we classify by:

$$\hat{C} = \arg\max_{C} \; P(C) \prod_{j=1}^{m} P(x_j \mid C)$$

What differs between variants of Naive Bayes is the assumed form of $P(x_j \mid C)$.

### Multinomial Naive Bayes

Used when features are **counts**— most commonly word counts in text classification. The likelihood of feature $x_j$ given class $C$ follows a Multinomial distribution:

$$P(x_j \mid C) = \frac{N_{C,j} + \alpha}{N_C + \alpha m}$$

where $N_{C,j}$ is the count of feature $j$ in class $C$, $N_C$ is the total count of all features in class $C$, $m$ is the number of features, and $\alpha$ is a smoothing parameter (typically $\alpha = 1$, called Laplace smoothing) to avoid zero probabilities for unseen words.

### Gaussian Naive Bayes

Used when features are **continuous**. The likelihood of feature $x_j$ given class $C$ is assumed to follow a Normal distribution:

$$P(x_j \mid C) = \frac{1}{\sqrt{2\pi\sigma_{C,j}^2}} \exp\!\left(-\frac{(x_j - \mu_{C,j})^2}{2\sigma_{C,j}^2}\right)$$

where $\mu_{C,j}$ and $\sigma_{C,j}^2$ are the mean and variance of feature $j$ within class $C$, estimated directly from the training data. This is a natural choice when features like height, temperature, or sensor readings are roughly bell-shaped within each class.

### Exponential Naive Bayes

Used when features are **non-negative and right-skewed**— for example, time between events or document lengths. The likelihood is modelled with an Exponential distribution:

$$P(x_j \mid C) = \lambda_{C,j} \, e^{-\lambda_{C,j} x_j}, \quad x_j \geq 0$$

where $\lambda_{C,j} = 1 / \mu_{C,j}$ is the rate parameter estimated as the inverse of the mean of feature $j$ in class $C$. This is the right choice when features represent durations or inter-arrival times, which are better described by the Exponential than the Normal distribution.