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

## Linear Classifiers

kNN has a number of disadvantages:

- The classifier must remember all of the training data and store it for future comparisons with the test data. This is space inefficient because datasets may easily be gigabytes in size.
- Classifying a test image is expensive since it requires a comparison to all training images.

We are now going to develop a more powerful approach to image classification that we will eventually naturally extend to Neural Networks. The approach will have two major components: a score function that maps the raw data to class scores, and a loss function that quantifies the agreement between the predicted scores and the ground truth labels. We will then cast this as an optimization problem in which we will minimize the loss function with respect to the parameters of the score function.

### Parameterized mapping from images to label scores

The first component of this approach is to define the score function that maps the pixel values of an image to confidence scores for each class. We will develop the approach with a concrete example. Let's assume a training dataset of images $x_i \in \mathbb{R}^D$, each associated with a label $y_i$. Here $i = 1 \ldots N$ and $y_i \in 1 \ldots K$. That is, we have $N$ examples (each with a dimensionality $D$) and $K$ distinct categories. For example, in CIFAR-10 we have a training set of $N = 50,000$ images, each with $D = 32 \times 32 \times 3 = 3072$ pixels, and $K = 10$, since there are 10 distinct classes (dog, cat, car, etc). We will now define the score function $f: \mathbb{R}^D \mapsto \mathbb{R}^K$ that maps the raw image pixels to class scores.

We will start out with arguably the simplest possible function, a linear mapping:

$$f(x_i, W, b) = Wx_i + b$$

In the above equation, we are assuming that the image $x_i$ has all of its pixels flattened out to a single column vector of shape $[D \times 1]$. The matrix $W$ (of size $[K \times D]$), and the vector $b$ (of size $[K \times 1]$) are the parameters of the function. In CIFAR-10, $x_i$ contains all pixels in the $i$-th image flattened into a single $[3072 \times 1]$ column, $W$ is $[10 \times 3072]$ and $b$ is $[10 \times 1]$, so 3072 numbers come into the function (the raw pixel values) and 10 numbers come out (the class scores). The parameters in $W$ are often called the **weights**, and $b$ is called the **bias vector** because it influences the output scores, but without interacting with the actual data $x_i$.

There are a few things to note:

1. **First**, note that the single matrix multiplication $Wx_i$ is effectively evaluating 10 separate classifiers in parallel (one for each class), where each classifier is a row of $W$.

2. **Notice also** that we think of the input data $(x_i, y_i)$ as given and fixed, but we have control over the setting of the parameters $W, b$. Our goal will be to set these in such way that the computed scores match the ground truth labels across the whole training set. We will go into much more detail about how this is done, but intuitively we wish that the correct class has a score that is higher than the scores of incorrect classes.

3. **An advantage** of this approach is that the training data is used to learn the parameters $W, b$, but once the learning is complete we can discard the entire training set and only keep the learned parameters. That is because a new test image can be simply forwarded through the function and classified based on the computed scores.

4. **Lastly**, note that classifying the test image involves a single matrix multiplication and addition, which is significantly faster than comparing a test image to all training images.

**Geometric Interpretation**

To understand linear classifiers geometrically, let's first consider the general form of a plane in ℝ³:

$$ax + by + cz = d$$

where $(a, b, c)$ is the **normal vector** to the plane and $d$ determines the plane's position in space.

- The normal vector $(a, b, c)$ is perpendicular to the plane

- The plane divides 3D space into two half-spaces

- Points on one side of the plane satisfy $ax + by + cz > d$

- Points on the other side satisfy $ax + by + cz < d$

- Points on the plane satisfy $ax + by + cz = d$

**Example**: The plane $2x - 3y + 4z = 12$ has normal vector $(2, -3, 4)$.

Now, let's see how this relates to linear classifiers. In a binary classification problem, our linear classifier computes:

$$f(x) = w^T x + b$$

where $w$ is the weight vector and $b$ is the bias.

**The Decision Boundary**: The equation $w^T x + b = 0$ defines a **hyperplane** in the input space that separates the two classes. This is exactly analogous to the plane equation $ax + by + cz = d$. This is how:

- $w$ is the **normal vector** to the hyperplane (just like $(a, b, c)$ in the plane equation)

- $b$ determines the **position** of the hyperplane (just like $d$ in the plane equation)

- The hyperplane divides the input space into two half-spaces

- Points in one half-space are classified as class 1 ($w^T x + b > 0$)

- Points in the other half-space are classified as class 2 ($w^T x + b < 0$)

**Multi-class extension**: For $K$ classes, we have $K$ hyperplanes, each defined by a row of the weight matrix $W$. 

In the multi-class case, we have $K$ score functions:

$$f_1(x) = w_1^T x + b_1$$

$$f_2(x) = w_2^T x + b_2$$

$$\vdots$$

$$f_K(x) = w_K^T x + b_K$$

The classifier predicts class $i$ if $f_i(x) > f_j(x)$ for all $j \neq i$.

The decision boundary between classes $i$ and $j$ is the set of points where the classifier is indifferent between the two classes, i.e., where $f_i(x) = f_j(x)$.

Starting with:

$$f_i(x) = f_j(x)$$

Substituting the score functions:

$$w_i^T x + b_i = w_j^T x + b_j$$

Rearranging terms:

$$w_i^T x - w_j^T x = b_j - b_i$$

Factoring out $x$:

$$(w_i - w_j)^T x = b_j - b_i$$

Moving all terms to one side:

$$(w_i - w_j)^T x + (b_i - b_j) = 0$$

The decision boundary between classes $i$ and $j$ is the hyperplane:

$$(w_i - w_j)^T x + (b_i - b_j) = 0$$

Note:

- The normal vector to this decision boundary is $(w_i - w_j)$

- The bias term is $(b_i - b_j)$

- Points on one side of this hyperplane are classified as class $i$

- Points on the other side are classified as class $j$

- Points on the hyperplane are exactly at the decision boundary

**Why this matters**: This geometric interpretation helps us understand that:

1. Linear classifiers create **linear decision boundaries**

2. The weight vector $w$ determines the **orientation** of the decision boundary

3. The bias $b$ determines the **position** of the decision boundary

4. The classifier's performance depends on how well the data can be separated by these linear boundaries

This connection between planes in ℝ³ and hyperplanes in linear classifiers provides an intuitive way to visualize and understand how linear classifiers work geometrically.

Coming back to CIFAR-10, we cannot visualize 3072-dimensional spaces, but if we imagine squashing all those dimensions into only two dimensions, then we can try to visualize what the classifier might be doing.

![Multi-class extension](linear_classifier.jpeg)

Above figure shows a cartoon representation of the image space, where each image is a single point, and three classifiers are visualized. For example, take the car classifier (in red), where the red line shows all points in the space that get a score of zero for the car class. The red arrow shows the direction of increase, so all points to the right of the red line have positive (and linearly increasing) scores, and all points to the left have a negative (and linearly decreasing) scores.

As we saw above, every row of $W$ is a classifier for one of the classes. The geometric interpretation of these numbers is that as we change one of the rows of $W$, the corresponding line in the pixel space will rotate in different directions. The biases $b$, on the other hand, allow our classifiers to translate the lines.

### Template Matching interpretation

Another interpretation for the weights $W$ is that each row of $W$ corresponds to a template (or sometimes also called a prototype) for one of the classes. The score of each class for an image is then obtained by comparing each template with the image using an inner product (or dot product) one by one to find the one that "fits" best. With this terminology, the linear classifier is doing template matching, where the templates are learned. Another way to think of it is that we are still effectively doing Nearest Neighbor, but instead of having thousands of training images we are only using a single image per class.

### Bias Trick

Recall that we defined the score function as:

$$f(x_i, W, b) = Wx_i + b$$

It is a little cumbersome to keep track of two sets of parameters (the biases $b$ and weights $W$) separately. A commonly used trick is to combine the two sets of parameters into a single matrix that holds both of them by extending the vector $x_i$ with one additional dimension that always holds the constant 1- a default bias dimension. With the extra dimension, the new score function will simplify to a single matrix multiply:

$$f(x_i, W) = Wx_i$$

With our CIFAR-10 example, $x_i$ is now $[3073 \times 1]$ instead of $[3072 \times 1]$ - (with the extra dimension holding the constant 1), and $W$ is now $[10 \times 3073]$ instead of $[10 \times 3072]$. The extra column that $W$ now corresponds to the bias $b$. An illustration might help clarify this transformation.

![Bias trick](bias_trick.jpeg)