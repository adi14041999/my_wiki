# Evaluating Generative Models

In any research field, evaluation drives progress. How do we evaluate generative models? The evaluation of discriminative models (classification, regression, etc.) is well understood because:

**Clear ground truth**: For discriminative tasks, we have access to labeled data that serves as ground truth. We can directly compare the model's predictions with the true labels.

**Simple Metrics**: Evaluation metrics are straightforward and interpretable:

   - **Classification**: Accuracy, precision, recall, F1-score, ROC-AUC

   - **Regression**: Mean squared error (MSE), mean absolute error (MAE), R²

   - **Ranking**: NDCG, MAP, MRR

**Domain-Agnostic**: These metrics work across different domains (computer vision, NLP, etc.) with minimal adaptation.

**Example**: For a binary classifier, we can compute accuracy as $\frac{\text{correct predictions}}{\text{total predictions}}$ and immediately understand how well the model performs.

Evaluating generative models is highly non-trivial. 
**Key question:** What is the task you care about? Density estimation- do you care about evaluating probabilities of images? Compression? Pure sampling/generation? Representation learning from unlabelled data? More than one task?

## Density Estimation or Compression

Likelihood as a metric is pretty good for Density Estimation.

   - Split data into train, validation and test sets.

   - Learn model $p_{\theta}$ using the train set.

   - Tune hyperparameters on the validation set.

   - Evaluate generalization with likelihood on test set: $\mathbb{E}_{p_{data}}[\log p_\theta]$

**Note**: This is the same as compression because, by Shannon's source coding theorem, the optimal code length for encoding data from distribution $p_{data}$ using model $p_\theta$ is $-\log p_\theta(x)$. The average number of bits needed to encode data from $p_{data}$ using model $p_\theta$ is:

$$\text{Average Code Length} = \mathbb{E}_{p_{data}}[-\log p_\theta(x)] = -\mathbb{E}_{p_{data}}[\log p_\theta(x)]$$

Therefore, maximizing $\mathbb{E}_{p_{data}}[\log p_\theta]$ is equivalent to minimizing the expected code length, which is the goal of compression. The intuition is that we assign short codes to frequent data points.

**Perplexity**: Another common metric for evaluating generative models is perplexity, defined as:

$$\text{Perplexity} = 2^{-\frac{1}{D}\mathbb{E}_{p_{data}}[\log p_\theta(x)]}$$

where $D$ is the dimension of the data. This normalizes the log-likelihood by the data dimension, making perplexity comparable across different dimensionalities.

Perplexity measures how "surprised" the model is by the data. Lower perplexity indicates better performance. For language models, perplexity represents the average number of choices the model has at each step when predicting the next token.

Not all generative models have tractable likelihoods. For models where exact likelihood computation is intractable, we need alternative evaluation approaches:

**VAEs**: We can compare models using the Evidence Lower BOund (ELBO):

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

While ELBO is a lower bound on the true likelihood, it provides a reasonable proxy for model comparison within the VAE framework.

**GANs**: GANs pose a unique challenge because they don't provide explicit likelihood estimates.

In general, unbiased estimation of probability density functions from samples is impossible.

## Sample Quality
   
