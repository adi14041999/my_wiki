# Hyperparameter Tuning

Hyperparameter tuning is the process of selecting the best set of configuration variables (known as hyperparameters) before a machine learning model begins training. Unlike model parameters (such as neural network weights or linear regression coefficients) that are learned automatically from the data, hyperparameters must be set manually or via automated search to control how the learning algorithm behaves.

## Main tuning techniques

**Grid Search:** An exhaustive brute-force method that tests every single combination of user-defined hyperparameter values using cross-validation. It is thorough but computationally expensive for large search spaces. You can implement this via tools like [Scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html).

**Randomized Search:** A technique that randomly samples a fixed number of parameter combinations from the specified distributions instead of trying every option. It is faster and often matches grid search performance.

**Early Stopping / Successive Halving (e.g., Hyperband, ASHA):** Methods that evaluate multiple configurations concurrently and stop underperforming training runs early to save time and compute power.

**Bayesian optimization:** Bayesian optimization is a technique based on Bayes’ theorem, which describes how to update a belief about an event as new evidence arrives. The core idea is to treat the relationship between hyperparameter values and model performance as an unknown function— one that is expensive to evaluate because each evaluation requires a full training run.

Rather than sampling this function blindly (as grid or random search do), Bayesian optimization maintains a **surrogate model**- a cheap probabilistic approximation of the true performance function built from the results of evaluations seen so far.

The algorithm then uses an **acquisition function** to decide where to evaluate next. The acquisition function balances two competing goals:

- **Exploitation**: try hyperparameter values where the surrogate predicts high performance.
- **Exploration**: try regions of the search space where the surrogate is uncertain, because the true function might be much better there than expected.

This loop means each new evaluation is informed by everything that came before, so Bayesian optimization typically finds good hyperparameter configurations in far fewer trials than grid or random search (at the cost of the overhead of fitting and querying the surrogate model between runs).


