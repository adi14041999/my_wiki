# SVD based methods

For this class of methods to find word embeddings (otherwise known as word vectors), we first loop over a massive dataset and accumulate word co-occurrence counts in some form of a matrix $X$, and then perform Singular Value Decomposition on $X$ to get a $USV^T$ decomposition. We then use the rows of $U$ as the word embeddings for all words in our dictionary.


Let us discuss a few choices of $X$.

As our first attempt, we make the bold conjecture that words that are related will often appear in the same documents. For instance, "banks", "bonds", "stocks", "money", etc. are probably likely to appear together. But "banks", "octopus", "banana", and "hockey" would probably not consistently appear together. We use this fact to build a word-document matrix $X$ in the following manner: loop over billions of documents and for each time word $i$ appears in document $j$, we add one to entry $X_{ij}$. This is obviously a very large matrix and it scales with the number of documents. So perhaps we can try something better.

In another method we count the number of times each word appears inside a window of a particular size around the word of interest. We calculate this count for all the words in corpus. We display an example below.

![img](svd0.png)

We now perform SVD on $X$ to obtain $X = USV^T$, observe the singular values (the diagonal entries in the resulting $S$ matrix), and select the first $k$ singular vectors to form $U_k$ as described in the dimensionality reduction section above. The choice of $k$ is based on the desired percentage variance captured.

The Singular Value Decomposition (SVD) of a matrix $X \in \mathbb{R}^{m \times n}$ is given by:

$$X = USV^T$$

where:

- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix whose columns are the left singular vectors

- $S \in \mathbb{R}^{m \times n}$ is a diagonal matrix with non-negative singular values $\sigma_1 \geq 
\sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$ on the diagonal

- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix whose columns are the right singular vectors

The singular values are ordered from largest to smallest, and they represent the importance of each corresponding dimension.

To reduce the dimensionality of our word embeddings, we select the first $k$ singular vectors. This gives us a rank-$k$ approximation of $X$:

$$X \approx U_k S_k V_k^T$$

where:

- $U_k \in \mathbb{R}^{m \times k}$ contains the first $k$ columns of $U$ (the $k$ most important left singular vectors)

- $S_k \in \mathbb{R}^{k \times k}$ is a diagonal matrix containing the first $k$ singular values

- $V_k \in \mathbb{R}^{n \times k}$ contains the first $k$ columns of $V$ (the $k$ most important right singular vectors)

The rows of $U_k$ then serve as our $k$-dimensional word embeddings. This rank-$k$ approximation captures the most important information in $X$ while reducing the dimensionality from the original space (which could be $m$ or $n$ dimensions) down to $k$ dimensions. The choice of $k$ is typically based on the desired percentage of variance captured, which can be computed as:

$$\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}$$

Problems with this kind of approach are outlined below:

- **Dynamic dimensions**: The dimensions of the matrix change very often (new words are added very frequently and corpus changes in size).

- **Sparsity**: The matrix is extremely sparse since most words do not co-occur.

- **High dimensionality**: The matrix is very high dimensional in general ($10^6 \times 10^6$).

- **Computational cost**: Quadratic cost to train (i.e., to perform SVD).

- **Word frequency imbalance**: Requires the incorporation of some hacks on $X$ to account for the drastic imbalance in word frequency.

