# Introduction to Machine Learning

Machine Learning (ML) is a field of study in Artificial Intelligence concerned with the development and study of systems comprising of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit programming language instructions. These systems are generally called as Machine Learning Models, or ML models.

In ML, we model the relationship between independent variables (also called features) and a dependent variable we want to predict. Machine Learning consists of a lot of different statistical algorithms and methods. Deciding which method (example, k-means versus hierarchical clustering) to use often means just trying it and seeing how well it performs. By performance, we mean accuracy in predictions.

When a Machine Learning method fits the training data really well but makes poor predictions, we say that it is overfit to the training data.

## Cross Validation

When building a Machine Learning model, we often split the available data into three parts:

- The training set is used to fit the model.
- The validation set is used to tune hyperparameters and make design choices.
- The test set is held back until the end and used only once to estimate how well the final chosen model performs on truly unseen data.

### Stopping Training During Cross Validation

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
