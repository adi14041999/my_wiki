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

Statistics provides us with a set of tools to quantify the variation that we find in everything and, for the purposes of Machine Learning, helps us make predictions and quantify how confident we should be in those predictions.

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