# Autoregressive models
We assume we are given access to a dataset:
$$
\mathcal{D} = \{ \mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_m \}
$$
where each datapoint is n-dimensional. For simplicity, we assume the datapoints are binary.
$$
x_i \in \{0,1\}^n
$$
## Representation
If you have n random variables:
$$
X_1, X_2, \dots, X_n
$$
then their joint probability can be written as a product of conditional probabilities:
$$
P(X_1, X_2, \dots, X_n) = P(X_1) \cdot P(X_2 \mid X_1) \cdot P(X_3 \mid X_1, X_2) \cdot \dots \cdot P(X_n \mid X_1, X_2, \dots, X_{n-1})
$$
In words:
> The probability of all n variables taking particular values equals:
> → the probability of the **first variable**,  
> → times the probability of the **second variable given the first**,  
> → times the probability of the **third variable given the first two**,  
> → and so on, until the n-th variable.

By this chain rule of probability, we can factorize the joint distribution over the n-dimensions as:
$$
p(\mathbf{x}) = \prod_{i=1}^n p(x_i \mid x_1, x_2, \dots, x_{i-1}) = \prod_{i=1}^n p(x_i \mid x_{<i})
$$
where
$$
x_{<i} = [x_1, x_2, \dots, x_{i-1}]
$$
denotes the vector of random variables with index less than i.