# Self-Supervised Learning

## Introduction

What is the problem with large-scale training?
We need a lot of labelled data.
Is there a way we can train neural networks without the need for huge manually labelled datasets?

**Self-supervised learning** is a paradigm where models learn representations from unlabeled data by creating supervisory signals from the data itself. Instead of relying on manual annotations, the model generates its own learning objectives by exploiting the inherent structure, relationships, or transformations present in the data.

The key idea is to design a **pretext task**— a task that doesn't require labels but forces the model to learn useful representations that can later be transferred to downstream tasks (e.g., classification, detection).

![img](ss0.png)
![img](ss1.png)

Some examples of pretext tasks are:

1. **Masked Language Modeling (BERT)**: Predict masked words in a sentence from surrounding context. The model learns linguistic representations by understanding word relationships and context.

2. **Image Rotation Prediction**: Rotate images by 0°, 90°, 180°, or 270° and train the model to predict the rotation angle. This forces the model to understand object orientation and spatial structure.

3. **Jigsaw Puzzles**: Shuffle image patches and train the model to predict the correct spatial arrangement. This encourages learning of spatial relationships and object parts.

4. **Contrastive Learning (SimCLR, MoCo)**: Create two augmented views of the same image and train the model to make their representations similar, while making representations of different images dissimilar. This learns robust, view-invariant features.

5. **Temporal Consistency (Video)**: In video sequences, predict future frames or track objects across time. The model learns motion without explicit labels.

6. **Colorization**: Convert grayscale images to color by predicting color channels from luminance. This requires understanding of object semantics and typical color associations.

![img](ss2.png)
![img](ss3.png)

## Differences between Attention Maps of Supervised and Self-Supervised models

Attention maps reveal what parts of an image a model focuses on when making predictions. There are fundamental differences between supervised and self-supervised models in what they attend to.

Supervised models learn to attend primarily to **discriminative features**— parts of objects that help distinguish between classes. For example, when classifying birds, they might focus on beaks, wings, or distinctive markings that differentiate species. Attention is often concentrated on small, highly discriminative regions. The model learns to ignore background and irrelevant details, focusing only on what's needed for the classification task. The attention patterns are optimized for the specific labeled task (e.g., "is this a dog or cat?"), which can lead to overfitting to dataset-specific cues rather than learning general object structure. Attention maps tend to be sparse, highlighting only the most discriminative patches rather than the entire object.

Self-supervised models learn to attend to **entire object structures** rather than just discriminative parts. They develop a more complete understanding of object shape, boundaries, and spatial relationships. Attention maps often cover larger regions, including object boundaries, edges, and structural elements. This is because pretext tasks like rotation prediction or jigsaw puzzles require understanding the full spatial structure of objects. Without task-specific labels, self-supervised models learn more general, transferable features that capture object semantics rather than class-specific discriminative cues. Attention maps typically show more uniform coverage of objects, attending to the whole object rather than just key parts. This makes them more suitable for tasks like object detection or segmentation where understanding the full object is important.

![img](ss6.png)

## Pretext task: predict rotations

![img](ss4.png)
![img](ss5.png)

## Pretext Tasks: Rearrangement and Inpainting

![img](ss6_2.png)
![img](ss7.png)
![img](ss8.png)
![img](ss9.png)

## Reconstruction-Based Learning (MAE)

**Masked Autoencoders (MAE)** are a self-supervised learning approach inspired by BERT's masked language modeling, but applied to images. The core idea is simple: mask out a large portion of image patches (typically 75%) and train the model to reconstruct the missing patches from the visible ones.

### How MAE Works

**Patch-based masking**: The input image is divided into non-overlapping patches. A high percentage (e.g., 75%) of patches are randomly masked out.

**Encoder-decoder architecture**: 

   - The **encoder** processes only the visible (unmasked) patches, learning to extract meaningful representations from partial observations.

   - The **decoder** takes the encoded representations along with mask tokens and reconstructs the full image, including the masked patches.

![img](mae0.png)
![img](mae2.png)
![img](mae3.png)

**Reconstruction objective**: The model is trained to predict the pixel values of masked patches, typically using mean squared error (MSE) loss between predicted and original patches.

![img](mae4.png)

The high masking ratio (75%) is crucial—it prevents the model from relying on trivial interpolation between nearby patches. Instead, it must develop a deep understanding of object structure, semantics, and spatial relationships to successfully reconstruct masked regions. This makes MAE particularly effective for learning general visual representations that capture the essence of objects and scenes.

## Summary of pretext tasks

![img](mae5.png)
![img](mae6.png)
![img](mae7.png)

## Contrastive Learning

### Intuition

**Contrastive learning** is based on a simple but powerful principle: **learn representations by contrasting similar and dissimilar examples**. The core idea is that an image and its augmented versions (e.g., rotated, cropped, color-jittered) should have similar representations, while different images should have dissimilar representations.

The intuition is that good visual representations should be **invariant** to data augmentations (the same object under different views should map to similar embeddings) but **discriminative** across different objects (different objects should map to different embeddings). By learning to distinguish between "positive pairs" (augmented views of the same image) and "negative pairs" (different images), the model develops rich, semantically meaningful representations without any labels.

![img](cl0.png)
![img](cl1.png)

In contrastive learning, we start with a **reference sample** $x$ (an anchor image). **Positive samples** are augmented views of the same image $x$ (e.g., rotated, cropped, or color-jittered versions), which should have similar representations to $x$. **Negative samples** are different images from the dataset, which should have dissimilar representations to $x$. The model learns by pulling positive samples closer to the reference while pushing negative samples further away in the embedding space.

![img](cl2.png)

### Formulation

![img](cl3.png)
![img](cl4.png)
![img](cl5.png)
![img](cl6.png)

### SimCLR

![img](cl7.png)

In the above figure, $\tilde{x}_i$ and $\tilde{x}_j$ are **positive samples** created through data augmentation from the same original image. They represent two different augmented views (e.g., different random crops, color jittering, or rotations) of the same underlying image, and the model learns to make their representations similar.

![img](cl8.png)
![img](cl10.png)

#### Results

![img](cl9.png)

As discussed in the contrastive learning formulation, the InfoNCE loss provides a lower bound on mutual information (MI). The larger the negative sample size $N$, the tighter this bound becomes, meaning the loss $-\mathcal{L}$ more closely approximates the true mutual information between the representations of positive pairs. Thus, we need to use a large batch size in SimCLR.

![img](cl11.png)

### DINO

DINO takes an input image and creates two different augmented views: one goes to the student network and the other to the teacher network. Both networks output representations that pass through softmax to produce probability scores over a set of classes.

![img](dino0.png)

**How did we get classes in an unsupervised method?** While DINO doesn't use labeled data, it requires an estimate of the number of output classes. In the official implementation, the DINO head output dimensionality (default: 65,000) is much larger than ImageNet's 1,000 classes. This is due to **over-segmentation**. By using many classes, the model can learn to distinguish fine-grained parts (e.g., cat whiskers, ears, nose) rather than just whole objects.

![img](dino1.png)

The training uses knowledge distillation: the teacher's output serves as the ground truth, and the student is trained to match it using cross-entropy loss. 

![img](dino2.png)

Importantly, the teacher is not pre-trained. It trains simultaneously with the student. Both networks have identical architectures and parameters, yet the teacher should produce better representations. This raises two questions: (1) How do we train the teacher? (2) Why is the teacher better?

#### Multi-crop Training

**Why the teacher is better**

Multi-crop training uses two types of crops: **global views** (more than 50% of the image) and **local views** (less than 50% of the image). 

![img](dino3.png)
![img](dino4.png)

During training, the student receives both global and local views, while the teacher receives only global views. The teacher produces better representations because global views contain more information than local views. When the student receives global views, it can match the teacher's performance, but on average, the teacher outperforms because it consistently receives global information.

![img](dino5.png)

**How we train the teacher**

Researchers tried different update strategies: copying student weights every iteration (0.1% accuracy), after each iteration (no improvement), and after each epoch (66.6% accuracy on ImageNet). Copying after each epoch works because it provides stable ground truth for the entire epoch, allowing the student to converge. If weights are copied every iteration, the teacher's outputs (ground truth) change constantly, preventing convergence.

The optimal solution uses **exponential moving average (EMA)**: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$, where $\lambda$ follows a cosine schedule from 0.996 to 1. This slowly updates teacher weights, incorporating student information gradually. Throughout training, both networks improve, with the teacher initially performing better due to global views. At convergence, both achieve similar performance.

**Why attention focuses on objects**

The student with local views (e.g., cat whiskers and fur) must match the teacher's representation from global views (most of the cat plus background). This local-global view difference is why DINO's attention maps naturally segment objects.

However, implementing DINO without proper safeguards causes **mode collapse**, where the model outputs the same representation regardless of input.

#### Avoiding Collapse

Two collapse scenarios can occur.

1. **Uniform distribution collapse**: Both teacher and student output uniform distributions across all classes for any input (cat, dog, building, etc.). This minimizes cross-entropy loss (since identical distributions have zero loss) but provides no discriminative information.

![img](dino6.png)

2. **Single-dimension collapse**: The output is dominated by a single dimension regardless of input (e.g., always predicting "cat").

![img](dino7.png)

**Solution 1: Sharpening with temperature**: To prevent uniform distributions, we add a temperature parameter $\tau$ to the softmax: $P = \text{softmax}(\text{logits} / \tau)$. When logits are nearly equal, standard softmax ($\tau=1$) produces a nearly uniform distribution. Lower temperatures (e.g., $\tau=0.1$ or $0.05$) sharpen the distribution, making small differences in logits produce highly peaked probabilities. However, sharpening alone can cause single-dimension collapse.

![img](dino8.png)

**Solution 2: Centering**: To prevent single-dimension collapse, we apply centering to the teacher's logits: $\text{logits}_t \leftarrow \text{logits}_t - c$, where $c$ is an exponential moving average of the mean teacher logits across all batch samples. This bias term $c$ is initially the mean of all teacher logits in the batch. Centering smooths the distribution and prevents collapse to a single mode. By combining sharpening (on both student and teacher) with centering (on teacher), we mitigate both collapse scenarios.  

![img](dino9.png)