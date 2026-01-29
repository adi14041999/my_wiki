# Model Taxonomy

## Models as functions

In many ways, ML models behave like software functions: they have **parameters**, take **inputs**, and produce **outputs**. You can **compose** them (pipe the output of one into another) and **decompose** them (treat submodules as smaller functions). That analogy helps when reading code or designing systems— you can reason about models as black boxes with a clear signature.

The crucial difference is *how* that mapping from input to output is defined. A traditional function encodes **handcrafted logic**. A human writes rules, conditionals, and formulas, and the program executes them. A model, by contrast, is a **statistical estimator**. Its parameters are not written by hand but **trained on data**. The input–output behavior is learned from examples (e.g. supervised learning) or from interaction with an environment (e.g. reinforcement learning), rather than from explicit rules. So while the *interface* looks like a function, the *implementation* is a fitted statistical object.

## Three lenses for understanding the ML landscape

You can organize models and methods along three axes: how they learn, how big they are, and what kind of task it is supposed to do.

### 1. Learning paradigms

**Supervised:** The model learns from **labeled data**— each input is paired with a correct answer. Examples: image classification (image → class label), spam detection (email → spam/not spam), house price prediction (features → price). The model’s job is to approximate the input–output mapping implied by the labels.

**Self-supervised:** No human-provided labels; the **labels are derived from the data itself**. The model is trained to predict missing or corrupted parts of the input. Examples: language models that predict the next token or a masked word (e.g. BERT’s “fill in the blank”), vision models that predict rotated orientation or missing image patches. Self-supervised learning is widely used for pretraining before fine-tuning on a downstream task.

**Unsupervised:** No labels at all. The goal is to find **patterns, clusters, or structure** in raw data. Examples: k-means clustering (group points by similarity), topic modeling (discover themes in documents), anomaly detection (flag points that don’t fit the learned distribution). You don’t have a “correct” target; you explore the data’s structure.

**Reinforcement:** The model learns **actions** from **states** and **rewards** provided by an environment. It isn’t given correct input–output pairs; it discovers good behavior by trial and error. Examples: game-playing agents (AlphaGo, DQN), robotics control, recommendation systems that optimize long-term engagement. The “label” is implicit in the reward signal.

### 2. Model complexity

**Shallow learning:** Relatively **few parameters**, often **interpretable**, **fast and cheap** to train and run. Suited to **single, well-defined tasks** and smaller datasets. Examples: linear/logistic regression, decision trees, small feedforward nets, classical ML (SVMs, k-NN). You can often inspect weights or rules and reason about why the model behaved as it did.

**Deep learning:** **Millions to billions of parameters**, usually **black box**, **expensive** in compute and data. Can exhibit **emergent capabilities** (e.g. in-context learning, chain-of-thought) that weren’t explicitly programmed. Examples: ResNets for vision, transformers for language and multimodal tasks. Complexity buys **representational power**— the ability to capture intricate structure and transformations— but at the cost of interpretability and resource requirements.

### 3. Task types

Common ways to frame what a model is asked to do (only listed a few here). 

- **Regression:** Predict **continuous** numerical outputs. Examples: predicting temperature, stock price, or the probability of an event. Loss is often mean squared error (MSE) or similar.

- **Classification:** Assign inputs to **discrete classes**. Examples: spam vs not spam, digit recognition (0–9), object categories in images. Can be binary or multiclass; outputs are typically class labels or class probabilities.

- **Clustering (unsupervised):** **Group** data points by similarity without using labels. Examples: segmenting customers, organizing documents into themes, discovering modes in a dataset. Algorithms include k-means, hierarchical clustering, and density-based methods.

- **Dimensionality reduction:** **Compress** data (fewer dimensions) while preserving as much structure as possible. Examples: visualization (e.g. t-SNE, UMAP), noise reduction, preprocessing for other models. PCA is a common tool.

#### Caveat: task types are limiting

The same problem can often be solved with different tasks in mind— e.g. anomaly detection can be framed as **classification** (normal vs anomalous) or as **clustering** (flag points far from any cluster). Large models further blur the boundaries: LLMs do classification in the process of doing next-token prediction rather than as separate task-specific heads. Use task types as an **initial filtering heuristic** (e.g. “do I need a continuous output or discrete classes?”), not as a rigid categorization.

## Practical framework

When selecting models, ask:

1. **What task needs accomplishing?** (e.g. regression, classification, clustering, generation)
2. **What data is available?** Labeled vs unlabeled, volume, and modality (text, image, tabular, etc.)
3. **What complexity can you afford?** Computational resources, latency, cost.

Task types provide the fastest path from business need to model selection, even if not theoretically rigorous.