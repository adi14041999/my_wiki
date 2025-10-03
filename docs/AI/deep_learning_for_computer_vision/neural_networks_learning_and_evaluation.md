# Neural Networks: Learning and Evaluation

## Learning

### Gradient Checks

In theory, performing a gradient check is as simple as comparing the analytic gradient to the numerical gradient. In practice, the process is much more involved and error prone. Here are some tips, tricks, and issues to watch out for:

**Use the centered formula:** The formula you may have seen for the finite difference approximation when evaluating the numerical gradient looks as follows:

$$\frac{df(x)}{dx} = \frac{f(x+h) - f(x)}{h}$$

where $h$ is a small number, which in practice is approximately 1e-5 or so. In practice, it is often much better to use the centered formula:

$$\frac{df(x)}{dx} = \frac{f(x+h) - f(x-h)}{2h}$$

This requires you to evaluate the loss function twice to check every single dimension of the gradient (so it is about 2 times as expensive), but the gradient approximation turns out to be much more precise. To see this, you can use Taylor expansion of $f(x+h)$ and $f(x-h)$ and see that the first order terms cancel out.

**Use relative error for the comparison:** What are the details of comparing the numerical gradient $f'_n$ to the analytic gradient $f'_a$? You might be tempted to keep track of whether their difference is greater than some threshold (e.g. 1e-4). However, this is problematic. For example, consider the case where their difference is 1e-4, and if the analytic gradient is 1e-2 then we'd consider the quantities to be very close, and hence the gradient would be OK. But if we consider the case where the analytic gradient is 1e-5, then we'd consider 1e-4 to be a huge difference, and hence the gradient would be bad. It is more appropriate to consider the relative error:

$$\frac{|f'_a - f'_n|}{\max(|f'_a|, |f'_n|)}$$

which considers their ratio of the differences to the ratio of the absolute values of both gradients. Notice that normally the relative error formula only includes one of the two terms (either one), but I prefer to max (or add) both to make it symmetric and to prevent dividing by zero in the case where one of the two is zero (which can often happen, especially with ReLUs). However, one must explicitly keep track of the case where both are zero and pass the gradient check in that edge case. In practice:

- relative error > 1e-2 usually means the gradient is probably wrong
- 1e-2 > relative error > 1e-4 should make you feel uncomfortable
- 1e-4 > relative error is usually okay for objectives with kinks. But if there are no kinks (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high.
- 1e-7 and less you should be happy.

Also keep in mind that the deeper the network, the higher the relative errors will be. So if you are gradient checking the input data for a 10-layer network, a relative error of 1e-2 might be okay because the errors build up on the way. Conversely, an error of 1e-2 for a single differentiable function likely indicates incorrect gradient.

**Use double precision:** A common pitfall is using single precision floating point to compute gradient check. It is often that case that you might get high relative errors (as high as 1e-2) even with a correct gradient implementation. In my experience I've sometimes seen my relative errors plummet from 1e-2 to 1e-8 by switching to double precision.

**Stick around active range of floating point:** It's a good idea to read through ["What Every Computer Scientist Should Know About Floating-Point Arithmetic"](http://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html), as it may demystify your errors and enable you to write more careful code. For example, in neural nets it can be common to normalize the loss function over the batch. However, if your gradients per datapoint are very small, then *additionally* dividing them by the number of data points is starting to give very small numbers, which in turn will lead to more numerical issues. This is why I like to always print the raw numerical/analytic gradient, and make sure that the numbers you are comparing are not extremely small (e.g. roughly 1e-10 and smaller in absolute value is worrying). If they are you may want to temporarily scale your loss function up by a constant to bring them to a "nicer" range where floats are more dense - ideally on the order of 1e-3 to 1e-1.

**Kinks in the objective:** One source of inaccuracy to be aware of during gradient checking is the problem of *kinks*. Kinks refer to non-differentiable parts of an objective function, introduced by functions such as ReLU ($max(0,x)$), or the SVM loss, Maxout neurons, etc. Consider gradient checking at $x = -1e6$. Since $x < 0$, the analytic gradient at that point is exactly zero. However, the numerical gradient would suddenly compute a non-zero gradient because $f(x+h)$ might cross the kink (e.g. if $h > 1e-6$) and introduce a non-zero contribution. In practice this is often not a problem, but it is worth being aware of.

Note that it is possible to know if a kink was crossed in the evaluation of the loss. This can be done by keeping track of the identities of all "winners" in a function of form $max(x,y)$; That is, was $x$ or $y$ higher during the forward pass. If the identities of the winners change during the gradient check then a kink was crossed and the numerical gradient will not be exact.

**Use only few datapoints:** One fix to the above problem of kinks is to use fewer datapoints, since loss functions that contain kinks (e.g. due to use of ReLUs or margin losses etc.) will have fewer kinks with fewer datapoints, so it is less likely for you to cross one when you perform the finite different approximation. Moreover, if your gradcheck for only ~2 or 3 datapoints then you would almost certainly gradcheck for an entire batch. Using very few datapoints also makes your gradient check faster and more efficient.

**Be careful with the step size h:** It is not necessarily the case that smaller h is better, because when h is much smaller, you might start running into numerical precision issues. Sometimes when the gradient doesn't check, it is possible that you change h to be 1e-4 or 1e-6 and suddenly the gradient will pass. This has to do with numerical precision issues with the finite difference approximation.

**Gradient check in the "attentive" regime:** A subtlety to be aware of is that it is possible for your gradient check to pass even when your implementation is incorrect. Consider the case where you have a bug in the forward pass that causes the loss to be computed incorrectly (e.g. you're using the wrong data, or you're not applying the loss in the right place). It is possible that this buggy forward pass still produces a gradient that passes your gradient check, even though the forward pass is incorrect. This is a very insidious bug. An even more insidious version of this bug is if your forward pass is correct, but you're not actually using the correct loss function that you think you're using.

**Don't let the gradient check pass if you haven't checked the loss**. Make sure that the loss you're checking is actually the loss that you think you're checking. Some versions of gradient check will only check the gradient with respect to the input, but not with respect to the parameters (e.g. bias terms, or in the case of a neural network, the weights). This is a common mistake. It is also important to check the gradient with respect to all parameters that participate in the forward pass.

**Remember to turn off dropout/augmentations**. Turn off any non-deterministic effects in your network, such as dropout, random data augmentations, etc. Otherwise the gradient check will fail. The reason is that these effects will add noise to the numerically computed gradient, whose scale might be larger than 1e-2. For example, dropout might not be applied at all during the forward pass, but it might be applied (or not applied) randomly during the numerical gradient computation.

**Check only few dimensions**. Gradient checks can be expensive to run. If you have many parameters, it can be good practice to check only some of the dimensions of the gradient and assume that the others are correct. This is especially true for the bias terms, where it is common to only check a few of these dimensions, since each bias term only affects a single output neuron.

### Sanity Checks

Here are a few sanity checks you might consider running before you plunge into expensive optimization:

- **Look for correct loss at chance performance.** Make sure you're getting the loss you expect when you initialize with small parameters. It's best to first check the data loss alone (so set regularization strength to zero). For example, for CIFAR-10 with a Softmax classifier we would expect the initial loss to be 2.302, because we expect a diffuse probability of 0.1 for each class (since there are 10 classes), and Softmax loss is the negative log probability of the correct class so: -ln(0.1) = 2.302. For The Weston Watkins SVM, we expect all desired margins to be violated (since all scores are approximately zero), and hence expect a loss of 9 (since margin is 1 for each wrong class). If you're not seeing these losses there might be issue with initialization.
- As a second sanity check, increasing the regularization strength should increase the loss
- **Overfit a tiny subset of data**. Lastly and most importantly, before training on the full dataset try to train on a tiny portion (e.g. 20 examples) of your data and make sure you can achieve zero cost. For this experiment it's also best to set regularization to zero, otherwise this can prevent you from getting zero cost. Unless you pass this sanity check with a small dataset it is not worth proceeding to the full dataset. Note that it may happen that you can overfit very small dataset but still have an incorrect implementation. For instance, if your datapoints' features are random due to some bug, then it will be possible to overfit your small training set but you will never notice any generalization when you fold it your full dataset.

### Babysitting the learning process

There are multiple useful quantities you should monitor during training of a neural network. These plots are the window into the training process and should be utilized to get intuitions about different hyperparameter settings and how they should be changed for more efficient learning.

The x-axis of the plots below are always in units of epochs, which measure how many times every example has been seen during training in expectation (e.g. one epoch means that every example has been seen once). It is preferable to track epochs rather than iterations since the number of iterations depends on the arbitrary setting of batch size.

#### Loss function

The first quantity that is useful to track during training is the loss, as it is evaluated on the individual batches during the forward pass. Below is a cartoon diagram showing the loss over time, and especially what the shape might tell you about the learning rate:

![Learning rates](learningrates.jpeg)
![Loss function](loss.jpeg)

*Left: A cartoon depicting the effects of different learning rates. With low learning rates the improvements will be linear. With high learning rates they will start to look more exponential. Higher learning rates will decay the loss faster, but they get stuck at worse values of loss (green line). This is because there is too much "energy" in the optimization and the parameters are bouncing around chaotically, unable to settle in a nice spot in the optimization landscape. Right: An example of a typical loss function over time, while training a small network on CIFAR-10 dataset. This loss function looks reasonable (it might indicate a slightly too small learning rate based on its speed of decay, but it's hard to say), and also indicates that the batch size might be a little too low (since the cost is a little too noisy).*

The amount of "wiggle" in the loss is related to the batch size. When the batch size is 1, the wiggle will be relatively high. When the batch size is the full dataset, the wiggle will be minimal because every gradient update should be improving the loss function monotonically (unless the learning rate is set too high).

Some people prefer to plot their loss functions in the log domain. Since learning progress generally takes an exponential form shape, the plot appears as a slightly more interpretable straight line, rather than a hockey stick. Additionally, if multiple cross-validated models are plotted on the same loss graph, the differences between them become more apparent.

Sometimes loss functions can look funny [lossfunctions.tumblr.com](http://lossfunctions.tumblr.com/).

#### Train/Val accuracy

The second important quantity to track while training a classifier is the validation/training accuracy. This plot can give you valuable insights into the amount of overfitting in your model:

![Accuracies](accuracies.jpeg)

*The gap between the training and validation accuracy indicates the amount of overfitting. Two possible cases are shown in the diagram on the left. The blue validation error curve shows very small validation accuracy compared to the training accuracy, indicating strong overfitting (note, it's possible for the validation accuracy to even start to go down after some point). When you see this in practice you probably want to increase regularization (stronger L2 weight penalty, more dropout, etc.) or collect more data. The other possible case is when the validation accuracy tracks the training accuracy fairly well. This case indicates that your model capacity is not high enough: make the model larger by increasing the number of parameters.*

#### Weights:Updates ratio

The last quantity you might want to track is the ratio of the update magnitudes to the value magnitudes. Note: updates, not the raw gradients (e.g. in vanilla sgd this would be the gradient multiplied by the learning rate). You might want to evaluate and log the norm of the weights (or some subset of the weights), and the norm of the updates (or again, some subset). Looking at the ratio of these two quantities can be helpful. For a single weight this ratio should be somewhere around 1e-3. If it is much smaller than this then the learning rate might be too low. If it is much higher then the learning rate might be too high.

#### Activation/Gradient distributions per layer

An incorrect initialization can slow or even stall learning. It can be useful to dump the histograms of activations and gradients, per layer, at initialization and during training. For example, for tanh networks the activations should be small, centered around 0. If you see that the activations are saturated at -1 and 1, then the activations are saturated, and the gradients will be zero (because the gradient of tanh is zero at -1 and 1). This will lead to the gradients not flowing backward through the network.

#### Visualization

The final tool you might want to use for visualization is the first-layer weights. It is often useful to visualize the weights of the first layer in a ConvNet, because the first layer operates directly on the raw pixel data. In general, it might be hard to interpret the weights of layers that are deeper in the network. The weights in the first layer are usually interpretable and can help you get an intuition for the kinds of features that your network is trying to learn.

For example, if the first layer is a ConvNet, it might be interesting to plot the filters, as we did in the Linear Classification section. If the first layer is fully connected, it might be interesting to plot the weight matrix as a color map.

### Parameter updates

Once the analytic gradient is computed with backpropagation, the gradients are used to perform a parameter update. There are several approaches for performing the update, which we discuss next.

We note that optimization for deep networks is currently a very active area of research. In this section we highlight some established and common techniques you may see in practice, briefly describe their intuition, but leave a detailed analysis outside of the scope of the class. We provide some further pointers for an interested reader.

#### SGD and bells and whistles

**Vanilla update**. The simplest form of update is to change the parameters along the negative gradient direction (since the gradient indicates the direction of increase, but we usually wish to minimize a loss function). Assuming a vector of parameters `x` and the gradient `dx`, the simplest update has the form:

```python
# Vanilla update
x += - learning_rate * dx
```

where `learning_rate` is a hyperparameter - a fixed constant. When evaluated on the full dataset, and when the learning rate is low enough, this is guaranteed to make non-negative progress on the loss function.

**Momentum update** is another approach that almost always enjoys better converge rates on deep networks. This update can be motivated from a physical perspective of the optimization problem. In particular, the loss can be interpreted as the height of a hilly terrain (and therefore also to the potential energy since $U = mgh$). The optimization process can then be seen as simulating a particle sliding down the hilly landscape.

Since the force on the particle is related to the gradient of potential energy (i.e. $F = -\nabla U$), the force felt by the particle is precisely the (negative) gradient of the loss function. Moreover, $F = ma$ so the (negative) gradient is in this view proportional to the acceleration of the particle. Note that this is different from the SGD update, where the gradient directly integrates the position. Instead, the physics view suggests an update in terms of velocity, which in turn is integrated to give the position. Here is the momentum update:

```python
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```

Here we see an introduction of a `v` variable that is initialized at zero, and an additional hyperparameter (`mu`). As an unfortunate misnomer, this variable is in optimization referred to as *momentum* (its typical value is about 0.9), but its physical meaning is more consistent with the coefficient of friction. Effectively, this variable damps the velocity and reduces the kinetic energy of the system, or otherwise the particle would never come to a stop at the bottom of a hill. When cross-validated, this parameter is usually set to values such as [0.5, 0.9, 0.95, 0.99]. Similar to annealing schedules for learning rates (discussed later, below), optimization can sometimes benefit a little from momentum schedules, where the momentum is increased in later stages of learning. A typical setting is to start with momentum of about 0.5 and anneal it to 0.99 or so over multiple epochs.

> With Momentum update, the parameter vector will build up velocity in any direction that has consistent gradient.

**Nesterov Momentum** is a slightly different version of the momentum update that has recently been gaining popularity. It enjoys stronger theoretical converge guarantees for convex functions and in practice it also consistenly works slightly better than standard momentum.

The core idea behind Nesterov momentum is that when the current parameter vector is at some position `x`, then looking at the momentum update above, we know that the momentum term alone (i.e. ignoring the second term with the gradient) is about to nudge the parameter vector by `mu * v`. Therefore, if we are about to compute the gradient, we can treat the future approximate position `x + mu * v` as a "lookahead" - this is a point in the vicinity of where we are soon going to end up. Hence, it makes sense to compute the gradient at `x + mu * v` instead of at the "old/stale" position `x`.

![Nesterov momentum](nesterov.jpeg)

*Nesterov momentum. Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this "looked-ahead" position.*

That is, in a slightly awkward notation, we would like to do the following:

```python
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```

However, in practice people prefer to express the update to look as similar to vanilla SGD or to the previous momentum update as possible. This is possible to achieve by manipulating the update above with a variable transform `x_ahead = x + mu * v`, and then expressing the update in terms of `x_ahead` instead of `x`. That is, the parameter vector we are actually storing is always the ahead version. The equations in terms of `x_ahead` (but renaming it back to `x`) then become:

```python
v_prev = v # back this up
v = mu * v - learning_rate * dx # velocity update stays the same
x += -mu * v_prev + (1 + mu) * v # position update changes form
```

#### Annealing the learning rate

In training deep networks, it is usually helpful to anneal the learning rate over time. Good intuition to have in mind is that with a high learning rate, the system contains too much kinetic energy and the parameter vector bounces around chaotically, unable to settle down into deep, narrow valleys of the loss function (where the optimum might be). Knowing when to decay the learning rate can be tricky: Decay it slowly and you'll be wasting computation bouncing around chaotically with little improvement for a long time. But decay it too aggressively and the system will cool too quickly, unable to reach the best position it can.

There are three common types of implementing the learning rate decay:

**Step decay**: Reduce the learning rate by some factor every few epochs. Typical values might be reducing the learning rate by a half every 30 epochs, or by a factor of 0.1 every 10 epochs. These numbers depend heavily on the type of problem and the model. One heuristic you may see in practice is to watch the validation error while training with a base learning rate, and reduce the learning rate by a constant (e.g. 0.5) whenever the validation error stops improving.

**Exponential decay**: Has the mathematical form $\alpha = \alpha_0 e^{-kt}$, where $\alpha_0, k$ are hyperparameters and $t$ is the iteration number (but you can also use units of epochs).

**1/t decay**: Has the mathematical form $\alpha = \alpha_0/(1+kt)$ where $\alpha_0, k$ are hyperparameters and $t$ is the iteration number.

In practice, we find that the step decay is slightly preferable because the hyperparameters it involves (the fraction of decay and the step timings in units of epochs) are more interpretable than the hyperparameters involved in the other two methods. Moreover, if you use a good learning rate schedule, you can get away with a higher initial learning rate than you normally would, which can lead to faster training.

#### Second-order methods

A second, popular group of methods for optimization is based on Newton's method, which is an iterative approach to finding the zero of a function. To apply this to optimization, we seek the zero of the gradient. The update rule for Newton's method is:

$$x \leftarrow x - [H f(x)]^{-1} \nabla f(x)$$

where $H f(x)$ is the Hessian matrix, which is a square matrix of second-order partial derivatives of the function. The term $\nabla f(x)$ is the gradient vector, as seen before. Intuitively, the Hessian describes the local curvature of the loss function, which allows us to perform a more efficient update. In particular, multiplying by the inverse Hessian leads the optimization to take more aggressive steps in directions of shallow curvature and shorter steps in directions of steep curvature.

In practice, applying Newton's method with the exact Hessian is usually not feasible for the following reasons:

1. Computing the Hessian is expensive, in $O(n^2)$ time
2. The Hessian has $O(n^2)$ elements, so storing it requires $O(n^2)$ memory, which is infeasible for high-dimensional problems
3. Inverting the Hessian is computationally expensive, in $O(n^3)$ time

Instead of using the full Hessian, various quasi-Newton methods have been developed that approximate the inverse Hessian. Among these, the most popular is L-BFGS, which uses the information in the gradients over time to form the approximation to the inverse Hessian (the matrix that is actually "quasi-Newton").

However, even L-BFGS often isn't a great choice for batch optimization of neural networks. In particular, L-BFGS is a batch method, but neural networks are usually trained with mini-batches. L-BFGS would need to be modified to work with mini-batches, but even then, it isn't often a great choice because the mini-batches introduce a lot of noise into the gradient, which makes the approximation to the inverse Hessian very poor.

#### Per-parameter adaptive learning rates

All previous approaches we've discussed so far manipulated the learning rate in a dimension-independent way. The following methods adapt the learning rate on a per-parameter basis, and can give more fine-grained control.

**Adagrad** is an adaptive learning rate method originally proposed by [Duchi et al.](http://jmlr.org/papers/v12/duchi11a.html)

```python
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Notice that the variable `cache` has size equal to the gradient matrix, and keeps track of per-parameter sum of squared gradients. This is then used to normalize the parameter update step, element-wise. The net effect is that parameters that receive big gradients will have their effective learning rate reduced, while parameters that receive small or infrequent updates will have their effective learning rate increased. The nice thing about Adagrad is that it requires no manual tuning of the learning rate, and it "just works". However, Adagrad's main weakness is its accumulation of the squared gradients in the denominator: Since every added term is positive, the accumulated sum keeps growing during training. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge.

**RMSprop** is a very effective, but currently unpublished adaptive learning rate method. Amusingly, everyone who uses this method in their work currently cites slide 29 of [Lecture 6](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) of Geoff Hinton's Coursera class. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. In particular, it uses a moving average of squared gradients instead. Letting the running average be computed with discounting factor $\rho$:

```python
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Here, `decay_rate` is a hyperparameter and typical values are [0.9, 0.99, 0.999]. Notice that the `x+=` update is identical to Adagrad, but the `cache` variable is a "leaky". Hence, RMSProp still modulates the learning rate of each weight based on the magnitudes of its gradients, which has a beneficial equalizing effect, but unlike Adagrad the updates do not get monotonically smaller.

**Adam** is a recently proposed method that some say works well in practice and compares favorably to RMSProp. The full Adam update also includes a bias correction mechanism, but the information above should give you a reasonable idea of the method. For more details see the [paper](http://arxiv.org/abs/1412.6980), or the [slides](http://cs231n.github.io/neural-networks-3/#ada).

```python
# Adam update
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```

Notice that the update looks exactly like RMSProp update, except the "smooth" version of the gradient `m` is used instead of the raw gradient vector `dx`. The paper recommends to set `beta1=0.9`, `beta2=0.999`, and `eps=1e-8`. In practice Adam is currently recommended as the default algorithm to use, and often works slightly better than RMSProp. However, it is often also worth trying SGD+Nesterov Momentum as an alternative. The full Adam update also includes a bias correction mechanism, which we recommend you to read about in the paper if you are interested in learning more.

### Hyperparameter Optimization

As we've seen, neural networks can have many hyperparameters, and it can be challenging to find good settings. A common approach is to search over hyperparameter space by training different models with different hyperparameter settings. Here are a few additional tips and tricks for performing this search:

**Implementation**. Larger networks require more hyperparameter tuning than smaller networks, and hyperparameter tuning can take a very long time (days to weeks) depending on the size of the data and the complexity of the model. A common approach is to start with a coarse search over a large range of hyperparameters (e.g. 2 orders of magnitude), and then perform a finer search around the best settings found from the coarse search.

**Use random search, not grid search**. [Bergstra and Bengio](http://www.jmlr.org/papers/v13/bergstra12a.html) argue in their paper that random search is more efficient than grid search. This is also empirically verified. Intuitively, this makes sense because there are usually only a few hyperparameters that matter, but the same hyperparameter can have a very different impact depending on the other hyperparameter settings. With grid search, you're essentially wasting time on hyperparameter combinations that don't matter.

**Careful with the evaluation on the test set**. It is common to accidentally leak information from the test set into the training process, and then report overly optimistic results. A common pitfall is to pick hyperparameters that work well on the test set, and then report the test set performance. This is problematic because the test set should be completely unseen until the very end. Instead, you should split your data into three sets: a training set, a validation set, and a test set. You should pick hyperparameters based on performance on the validation set, and then do a single evaluation on the test set at the very end.

**Prefer random search over grid search**. As mentioned above, random search is more efficient than grid search. However, there are some cases where you might want to use grid search. For example, if you have a hyperparameter that you know is important (e.g. learning rate), and you want to explore it systematically, then grid search might make sense. But in general, random search is preferred.

## Evaluation

### Model Ensembles

In practice, one reliable approach to improving the performance of Neural Networks by a few percent is to train multiple independent models, and at test time average their predictions. As the number of models in the ensemble increases, the performance typically monotonically improves (though with diminishing returns). Moreover, the improvements are more dramatic with higher model variety in the ensemble. There are a few approaches to forming an ensemble:

- **Same model, different initializations**. Use cross-validation to determine the best hyperparameters, then train multiple models with the best set of hyperparameters but with different random initialization. The danger with this approach is that the variety is only due to initialization.
- **Top models discovered during cross-validation**. Use cross-validation to determine the best hyperparameters, then pick the top few (e.g. 10) models to form the ensemble. This improves the variety of the ensemble but has the danger of including suboptimal models. In practice, this can be easier to perform since it doesn't require additional retraining of models after cross-validation
- **Different checkpoints of a single model**. If training is very expensive, some people have had limited success in taking different checkpoints of a single network over time (for example after every epoch) and using those to form an ensemble. Clearly, this suffers from some lack of variety, but can still work reasonably well in practice. The advantage of this approach is that is very cheap.
- **Running average of parameters during training**. Related to the last point, a cheap way of almost always getting an extra percent or two of performance is to maintain a second copy of the network's weights in memory that maintains an exponentially decaying sum of previous weights during training. This way you're averaging the state of the network over last several iterations. You will find that this "smoothed" version of the weights over last few steps almost always achieves better validation error. The rough intuition to have in mind is that the objective is bowl-shaped and your network is jumping around the mode, so the average has a higher chance of being somewhere nearer the mode.

One disadvantage of model ensembles is that they take longer to evaluate on test example. An interested reader may find the recent work from Geoff Hinton on ["Dark Knowledge"](https://www.youtube.com/watch?v=EK61htlw8hY) inspiring, where the idea is to "distill" a good ensemble back to a single model by incorporating the ensemble log likelihoods into a modified objective.

## Additional References

- [SGD](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) tips and tricks from Leon Bottou
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (pdf) from Yann LeCun
- [Practical Recommendations for Gradient-Based Training of Deep Architectures](http://arxiv.org/pdf/1206.5533v2.pdf) from Yoshua Bengio