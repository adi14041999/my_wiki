# Image Classification with Linear Classifiers

## The Image Classification Task

Image classification is a fundamental computer vision task where, given an image and a set of labels, the goal is to assign the image to one of the labels. This is essentially a supervised learning problem where we want to learn a mapping from input images to discrete class labels.

Formally given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ where:

- $x_i$ is an input image (typically represented as a high-dimensional vector of pixel values)

- $y_i$ is the corresponding class label from a predefined set of categories $\{1, 2, \ldots, C\}$

- The task is to learn a function $f: \mathbb{R}^d \rightarrow \{1, 2, \ldots, C\}$ that can accurately predict the class label for new, unseen images

**Key challenges**:

- **High dimensionality**: Images are typically represented as very high-dimensional vectors (e.g., a 224×224 RGB image has 150,528 dimensions)

- **Variability**: The same object can appear in different poses, lighting conditions, scales, backgrounds, camera movements, with background clutter, at different scales (zoom) within the image, with partial occlusion, deformation, and varying contextual information

- **Intra-class variation**: Objects within the same class can look very different

- **Inter-class similarity**: Objects from different classes can sometimes look very similar

## Machine Learning: Data-Driven Approach

The machine learning approach to image classification follows a systematic, data-driven methodology that can be broken down into three main steps:

**1. Collect a Dataset of Images and Labels:** The first step involves gathering a comprehensive dataset where each image is paired with its corresponding class label.

**2. Use ML Algorithms to Train a Classifier:** Once we have the dataset, we employ machine learning algorithms like Linear Regression, Support Vector Machines (SVM), or Logistic Regression to learn a mapping from images to class labels. The choice of algorithm depends on the complexity of the problem, dataset size, and computational constraints. 

**3. Evaluate the Classifier on New Images:** The final step is to assess how well the trained classifier performs on previously unseen images. This evaluation process includes measuring accuracy, precision, recall, and F1-score on the held-out test set. Also includes validation techniques like k-fold cross-validation to get robust performance estimates.

## k-Nearest Neighbors (k-NN) Algorithm

The k-Nearest Neighbors algorithm is one of the simplest and most intuitive machine learning algorithms for classification. It's a non-parametric, instance-based learning method that makes predictions based on the similarity of new examples to previously seen training examples.

**Training Phase**: k-NN is a "lazy learner" - it doesn't actually learn a model during training. Instead, it simply stores all the training examples and their labels.

**Prediction Phase**: For a new test image, k-NN:

1. Computes the distance between the test image and all training images

2. Identifies the k nearest training examples (neighbors)

3. Assigns the class label that appears most frequently among these k neighbors

The choice of distance metric is crucial for k-NN performance:

- **Euclidean Distance**: $d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$

- **Manhattan Distance**: $d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$

- **Cosine Similarity**: $d(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||}$

Understanding the computational efficiency of k-NN requires analyzing its time complexity using Big O notation.

**Big O Notation**: $O(f(n))$ describes how the runtime of an algorithm grows as the input size $n$ increases. It provides an upper bound on the worst-case performance, focusing on the dominant term and ignoring constants and lower-order terms.

**k-NN Time Complexity**:

- **Training Time**: $O(1)$ - k-NN doesn't perform any computation during training; it simply stores the training data

- **Prediction Time**: $O(N)$ - For each prediction, k-NN must compute distances to all $N$ training examples

In real-world applications, we typically want classifiers that are **fast at prediction** and can tolerate **slow training** because:

1. **Training happens once**: We train the model offline, often overnight or over days, so training time is less critical
2. **Prediction happens repeatedly**: Once deployed, the model makes thousands or millions of predictions per day
3. **Real-time requirements**: Many applications (autonomous vehicles, medical diagnosis, security systems) need immediate predictions
4. **Scalability**: As the dataset grows, k-NN prediction time grows linearly, making it impractical for large-scale systems

This is why we often prefer **parametric models** (like linear classifiers) that invest computational effort upfront during training to enable fast predictions later.

# Linear Classification

kNN has a number of disadvantages:

- The classifier must remember all of the training data and store it for future comparisons with the test data. This is space inefficient because datasets may easily be gigabytes in size.
- Classifying a test image is expensive since it requires a comparison to all training images.

We are now going to develop a more powerful approach to image classification that we will eventually naturally extend to Neural Networks. The approach will have two major components: a score function that maps the raw data to class scores, and a loss function that quantifies the agreement between the predicted scores and the ground truth labels. We will then cast this as an optimization problem in which we will minimize the loss function with respect to the parameters of the score function.

## Parameterized mapping from images to label scores

The first component of this approach is to define the score function that maps the pixel values of an image to confidence scores for each class. We will develop the approach with a concrete example. Let's assume a training dataset of images $x_i \in \mathbb{R}^D$, each associated with a label $y_i$. Here $i = 1 \ldots N$ and $y_i \in 1 \ldots K$. That is, we have $N$ examples (each with a dimensionality $D$) and $K$ distinct categories. For example, in CIFAR-10 we have a training set of $N = 50,000$ images, each with $D = 32 \times 32 \times 3 = 3072$ pixels, and $K = 10$, since there are 10 distinct classes (dog, cat, car, etc). We will now define the score function $f: \mathbb{R}^D \mapsto \mathbb{R}^K$ that maps the raw image pixels to class scores.

We will start out with arguably the simplest possible function, a linear mapping:

$$f(x_i, W, b) = Wx_i + b$$

In the above equation, we are assuming that the image $x_i$ has all of its pixels flattened out to a single column vector of shape $[D \times 1]$. The matrix $W$ (of size $[K \times D]$), and the vector $b$ (of size $[K \times 1]$) are the parameters of the function. In CIFAR-10, $x_i$ contains all pixels in the $i$-th image flattened into a single $[3072 \times 1]$ column, $W$ is $[10 \times 3072]$ and $b$ is $[10 \times 1]$, so 3072 numbers come into the function (the raw pixel values) and 10 numbers come out (the class scores). The parameters in $W$ are often called the **weights**, and $b$ is called the **bias vector** because it influences the output scores, but without interacting with the actual data $x_i$.

There are a few things to note:

1. **First**, note that the single matrix multiplication $Wx_i$ is effectively evaluating 10 separate classifiers in parallel (one for each class), where each classifier is a row of $W$.

2. **Notice also** that we think of the input data $(x_i, y_i)$ as given and fixed, but we have control over the setting of the parameters $W, b$. Our goal will be to set these in such way that the computed scores match the ground truth labels across the whole training set. We will go into much more detail about how this is done, but intuitively we wish that the correct class has a score that is higher than the scores of incorrect classes.

3. **An advantage** of this approach is that the training data is used to learn the parameters $W, b$, but once the learning is complete we can discard the entire training set and only keep the learned parameters. That is because a new test image can be simply forwarded through the function and classified based on the computed scores.

4. **Lastly**, note that classifying the test image involves a single matrix multiplication and addition, which is significantly faster than comparing a test image to all training images.

**Example: Binary Classification**

To make this more concrete, let's consider a simpler example with just two classes (e.g., "cat" vs "dog"). In this case:

- $K = 2$ (two classes)

- $W$ becomes a $[2 \times D]$ matrix

- $b$ becomes a $[2 \times 1]$ vector

- The output $f(x_i, W, b) = Wx_i + b$ gives us two scores: $[s_{\text{cat}}, s_{\text{dog}}]$

For a $32 \times 32 \times 3$ image, we'd have:

- $W$: $[2 \times 3072]$ matrix

- $b$: $[2 \times 1]$ vector  

- Input: $[3072 \times 1]$ flattened image

- Output: $[2 \times 1]$ vector with cat and dog scores

The class with the higher score is our prediction. For example, if $s_{\text{cat}} = 2.1$ and $s_{\text{dog}} = 0.8$, we predict "cat".

**Geometric Interpretation: What about 1 class?**

When we have just 1 class, the linear classifier becomes even simpler. In this case:

- $K = 1$ (one class)

- $W$ becomes a $[1 \times D]$ vector (a single row)

- $b$ becomes a scalar

- The output is a single score: $f(x_i, W, b) = Wx_i + b$ (scalar)

Geometrically, this means we're computing the **dot product** between the weight vector $W$ and the input image $x_i$, plus a bias term. The dot product $W \cdot x_i$ measures how "aligned" the input image is with the learned weight vector $W$.

- If $W \cdot x_i + b > 0$, we predict the positive class

- If $W \cdot x_i + b < 0$, we predict the negative class (or "not the class")

- The decision boundary is the hyperplane (a hyperplane is a flat, n-1 dimensional boundary that divides an n-dimensional space into two sections. In 2D, a hyperplane is a line, and in 3D, it's a plane. In machine learning, hyperplanes are used to categorize data by separating different classes) where $W \cdot x_i + b = 0$

**Why positive for positive class and negative for negative class?**

This is actually just a **convention** - we could have chosen the opposite! The key insight is that we need a way to distinguish between the two classes, and using the sign of the output is a natural way to do this.

The convention makes intuitive sense:
- **Positive score** → "Yes, this image belongs to the target class"
- **Negative score** → "No, this image does not belong to the target class"

The actual values of $W$ and $b$ are learned during training to make this distinction meaningful. The learning process will adjust $W$ and $b$ so that:
- Images of the target class tend to produce positive scores
- Images of other classes tend to produce negative scores

We could just as easily flip the convention and say "negative means positive class" - the math would work the same way, just with the signs flipped.

This is essentially a **linear separator** in the high-dimensional pixel space. The weight vector $W$ can be thought of as a "template" - it learns what patterns in the pixel space are most indicative of the target class.

**Geometric Visualization of the Hyperplane when K=1**

When K=1, we're working in a very high-dimensional space (e.g., 3072 dimensions for CIFAR-10 images). While we can't visualize 3072 dimensions directly, we can understand the geometry:

1. **The Hyperplane**: The equation $W \cdot x_i + b = 0$ defines a hyperplane in $\mathbb{R}^{3072}$. This hyperplane has dimension 3071 (one less than the ambient space).

2. **Normal Vector**: The weight vector $W$ is **perpendicular** to the hyperplane. This means $W$ points in the direction of maximum "positive class-ness" - the direction that most increases the score.

3. **Distance from Origin**: The bias term $b$ determines how far the hyperplane is from the origin. Specifically, the distance from the origin to the hyperplane is $\frac{|b|}{\|W\|}$.

4. **Two Half-Spaces**: The hyperplane divides the 3072-dimensional space into two regions:
   - **Positive half-space**: $W \cdot x_i + b > 0$ (predict positive class)
   - **Negative half-space**: $W \cdot x_i + b < 0$ (predict negative class)

5. **Intuitive Picture in 2D**: Imagine projecting this onto 2D - the hyperplane becomes a line, and $W$ is perpendicular to that line. Points on one side of the line are classified as positive, points on the other side as negative.

The key insight is that even though we're in 3072 dimensions, the decision boundary is still just a flat hyperplane - it's not curved or complex. This is what makes linear classifiers "linear"!