# Regularization

Increasing the amount of training data is one way of reducing overfitting. Fortunately, there are other techniques which can reduce overfitting, even when we have a fixed network and fixed training data. These are known as regularization techniques.

## Cross-Entropy Loss with L2 Regularization

The complete loss function for a softmax classifier with L2 regularization combines the cross-entropy loss with a regularization term:

$$L = \frac{1}{N} \sum_{i=1}^{N} L_i + \lambda R(W)$$

where:

- $L_i = -\log\left(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}\right)$ is the cross-entropy loss for the $i$-th sample

- $R(W) = \sum_k \sum_l W_{k,l}^2$ is the L2 regularization term (sum of squared weights)

- $\lambda$ is the regularization strength hyperparameter

- $N$ is the number of training samples

Intuitively, the effect of regularization is to make it so the network prefers to learn small weights, all other things being equal. Put another way, regularization can be viewed as a way of compromising between finding small weights and minimizing the original loss function. The relative importance of the two elements of the compromise depends on the value of $\lambda$: when $\lambda$ is small we prefer to minimize the original loss function, but when is large we prefer small weights.

The compromise exists because the two objectives in our loss function can conflict with each other:

1. **Minimizing the original loss function**: This encourages the model to fit the training data as well as possible, which often requires large weights to capture complex patterns and noise in the data. To understand why large weights are often needed, consider what happens when we try to fit training data perfectly. Real training data often contains noise, outliers, or mislabeled examples. To achieve near zero training error, the model must learn to classify these noisy examples correctly. The model must thus be sensitive to noisy samples. Having large weights makes the model sensitive to variation in input data. The model might learn to recognize very specific features of individual training examples (like particular pixel patterns, lighting conditions, or background elements) rather than generalizable features. This often requires large weights to amplify these specific signals. In essence, the original loss function asks: "How can I classify every training example correctly?". The answer often involves learning very specific, complex patterns with noise that require large weights to implement.

2. **Minimizing the regularization term**: This encourages small weights, which constrains the model's capacity to fit complex patterns.

**Example: Which weights does the regularizer prefer?**

Consider the following example with input $x = [1, 1, 1, 1]$ and two different weight vectors:

- $w_1 = [1, 0, 0, 0]$ 
- $w_2 = [0.25, 0.25, 0.25, 0.25]$

Both weight vectors produce the same output: $w_1 \cdot x = 1$ and $w_2 \cdot x = 1$. However, the L2 regularizer strongly prefers $w_2$ over $w_1$.

**L2 regularization calculation:**

- $R(w_1) = \sum_{i} w_{1,i}^2 = 1^2 + 0^2 + 0^2 + 0^2 = 1$

- $R(w_2) = \sum_{i} w_{2,i}^2 = 0.25^2 + 0.25^2 + 0.25^2 + 0.25^2 = 4 \times 0.0625 = 0.25$

Even though both weight vectors produce identical predictions, the regularizer penalizes $w_1$ four times more heavily than $w_2$ because $w_1$ concentrates all its "weight" in a single dimension, while $w_2$ distributes the same total "influence" more evenly across all dimensions.

This example illustrates why regularization encourages **weight spreading** rather than **weight concentration**- it prefers solutions that use many small weights rather than a few large ones, even when both approaches achieve the same functional result.