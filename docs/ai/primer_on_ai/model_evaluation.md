# Model Evaluation

Once a model has been trained, we need to measure how good it actually is. That question turns out to be surprisingly nuanced: the right answer depends on the learning paradigm, the task, and the cost of different kinds of mistakes in the real application.

## Evaluation strategy depends on learning type

The first question to ask is what kind of learning the model uses, because that determines what evaluation is even possible. For a full overview of learning paradigms, see [Model Taxonomy](model_taxonomy.md).

**Supervised learning** has ground truth. Every example in the test set comes with a correct answer, so we can directly compare predictions to those answers. The key is to go beyond a single summary number and understand *where* the model fails. A model that is 95% accurate overall can still be systematically wrong for a particular class or subgroup.

**Unsupervised learning** has no correct answers, which makes evaluation harder. We are asking whether the structure the model found is meaningful rather than whether predictions are right. For clustering, a common proxy is the **silhouette score**, which measures how tight a cluster is relative to how far it sits from neighboring clusters— higher separation suggests better-quality groupings.

## The accuracy limitation

For classification tasks it is tempting to report a single accuracy number, but accuracy treats every error as equally bad. In most real applications, different errors have very different costs.

A **confusion matrix** breaks predictions into four cells:

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True Positive (TP) | False Negative (FN) |
| **Actually negative** | False Positive (FP) | True Negative (TN) |

For a full breakdown of each cell and worked examples using spam filters and medical tests, see [Introduction to Machine Learning — Assessing Model Performance](introduction_to_machine_learning.md#assessing-model-performance).

From this we can compute metrics that capture the cost structure of the specific task:

- **Precision** $= \frac{TP}{TP + FP}$: of all the times the model predicted positive, how often was it right? This matters most when false positives are costly. For example, flagging a benign biopsy as cancerous triggers unnecessary treatment.
- **Recall** $= \frac{TP}{TP + FN}$: of all the true positives that exist, how many did the model catch? This matters most when false negatives are costly. For example, missing a fraudulent transaction means the fraud goes through.
- **F1 score** $= 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$: the harmonic mean, which balances precision and recall. Useful when both types of error matter and neither should dominate.

When the model outputs a probability score rather than a hard label, the choice of decision threshold shifts the precision–recall trade-off. The [ROC curve and AUC](introduction_to_machine_learning.md#roc-curve) let you visualise this trade-off across all thresholds and compare models at a glance. For imbalanced datasets (where one class is far rarer than the other) the [Precision-Recall curve](introduction_to_machine_learning.md#precision-recall-curve) is a more informative alternative, because the ROC curve can look deceptively good when true negatives vastly outnumber positives.

Choosing the right metric requires understanding the real-world consequences of each kind of mistake, not just optimizing for overall accuracy.

## Generalization vs. memorization

The purpose of a model is to generalize— to perform well on data it has never seen. A model that merely memorizes its training data is useless in practice.

- **Underfitting** occurs when the model is too simple to capture the relevant structure, leading to poor performance even on training data.
- **Overfitting** occurs when the model is too complex and memorizes noise in the training data, performing well on training examples but poorly on new ones.

The standard remedy is to evaluate the final model on a held-out **test set** that played no role in training or hyperparameter selection. [Introduction to Machine Learning](introduction_to_machine_learning.md#cross-validation) covers the three-way split (training, validation, test) in detail, including why the test set must remain untouched until the very end and how data leakage can make a model appear to generalize when it does not.

Model complexity is a direct lever on this trade-off. More parameters increase a model's capacity to fit training data, but at the cost of higher overfitting risk and greater computational demands.

## Task-specific evaluation approaches

Different task types call for different metrics. For a description of the task types themselves, see [Model Taxonomy — Task Types](model_taxonomy.md#3-task-types).

### Classification

Beyond precision and recall, two tools are particularly useful for understanding classifier behavior across operating conditions:

- **Confusion matrices** reveal per-class performance, which is essential in multi-class problems where a model might excel at some categories and fail at others.
- **ROC curves** (Receiver Operating Characteristic) plot the true positive rate against the false positive rate as the decision threshold varies. The area under the ROC curve (AUC-ROC) summarizes performance across all thresholds, making it useful when the threshold itself has not yet been decided.

### Regression

Regression models predict continuous outputs, so evaluation focuses on the magnitude of prediction errors:

- **Mean Squared Error (MSE)** $= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ penalizes large errors heavily and has smooth gradients, making it well-suited as a training objective.
- **Mean Absolute Error (MAE)** $= \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$ is more robust to outliers but produces non-smooth gradients that can slow gradient-based optimization.

The choice between them depends on whether outlier errors should be penalized disproportionately (MSE) or treated uniformly (MAE).

### Clustering

Because there are no labels, clustering evaluation relies on geometric properties of the discovered groups. The **silhouette score** for a data point $i$ is:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

where $a(i)$ is the mean distance from $i$ to all other points in its cluster, and $b(i)$ is the mean distance from $i$ to all points in the nearest neighboring cluster. A score near $+1$ means the point is well inside its own cluster; a score near $-1$ means it would fit better elsewhere.

### Generative models

Generative models (image generators, language models, and similar) produce non-deterministic outputs that cannot be compared against a single ground truth. Evaluation therefore relies on benchmarking approaches:

- The **Inception Score** uses a separate pre-trained classifier to rate output quality, measuring both variety (diversity of generated samples) and clarity (each sample looks unambiguously like something).
- For language models, downstream-task benchmarks are common proxies.
- Human evaluation remains the gold standard for subjective qualities like coherence or aesthetic appeal.

The key shift is from asking "is this correct?" to asking "how close is this to good?"

## Production considerations

A model that performs well in evaluation can still degrade over time. **Data drift**, the gradual shift of real-world inputs away from the training distribution, is one of the most common failure modes in deployed systems. Continuous automated evaluation pipelines that monitor live performance are essential, not optional.

**Human-in-the-loop** feedback is irreplaceable for subjective outputs. Mechanisms like thumbs-up/thumbs-down ratings and A/B testing let real users signal quality in ways that automated metrics cannot fully capture.

Beyond predictive performance, production evaluation also involves **practical metrics** such as computational efficiency (RAM usage, CPU or GPU time), model size and deployment constraints, and maintainability— how easily the model can be retrained as new data arrives.

## Emerging challenges

Several evaluation challenges have grown in importance as models have become more capable and more widely deployed.

**Fairness and bias:** A model can achieve high aggregate accuracy while performing systematically worse for particular demographic groups. Evaluating equity requires disaggregating metrics by subgroup and choosing fairness criteria appropriate to the application.

**Adversarial robustness:** Models can be deliberately attacked through input perturbations, data poisoning during training, or misuse of model outputs. Evaluation should include adversarial stress-testing rather than only clean-data benchmarks.

**Multimodal systems:** When a model handles text, images, audio, or other modalities jointly, benchmarking becomes more complex. Per-modality metrics may miss cross-modal failure modes, and new benchmarks that probe multimodal reasoning are an active area of research.

## Core decision framework

Three principles anchor sound evaluation practice:

1. **Match the metric to the task:** Supervised tasks support direct prediction comparison; unsupervised tasks require structural proxies. Within supervised learning, the right metric depends on the cost of each type of error.
2. **Understand error consequences:** Choose metrics that reflect what goes wrong in the real application, not just what is easy to compute.
3. **Plan for production:** Evaluation does not end at deployment. Treat models like production software. Set up monitoring pipelines, define retraining triggers, and expect that the distribution of incoming data will shift over time.
