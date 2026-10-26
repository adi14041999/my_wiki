# Multi-armed Bandits

## A k-armed Bandit Problem

Consider the following learning problem. You are faced repeatedly with a choice among $k$ different options, or actions, $a \in \{1, 2, \dots, k\}$. On each time step $t$ you select an action $A_t$, and after each choice you receive a numerical reward $R_t$ chosen from a stationary probability distribution that depends on the action you selected:

$$
R_t \sim p(r \mid A_t), \qquad \text{where } p(r \mid a) \text{ does not change over time.}
$$

Your objective is to maximize the expected total reward over some time period, for example, over $T = 1000$ action selections, or time steps:

$$
\max \; \mathbb{E}\left[ \sum_{t=1}^{T} R_t \right].
$$

This is the original form of the $k$-armed bandit problem, so named by analogy to a slot machine, or “one-armed bandit,” except that it has $k$ levers instead of one. Each action selection $A_t$ is like a play of one of the slot machine’s levers, and the rewards $R_t$ are the payoffs for hitting the jackpot. Through repeated action selections you are to maximize your winnings by concentrating your actions on the best levers.

In our $k$-armed bandit problem, each of the $k$ actions has an expected or mean reward given that that action is selected; let us call this the **value** of that action. We denote the action selected on time step $t$ as $A_t$, and the corresponding reward as $R_t$. The value then of an arbitrary action $a$, denoted $q_*(a)$, is the expected reward given that $a$ is selected:

$$
q_*(a) \doteq \mathbb{E}\left[ R_t \mid A_t = a \right].
$$

If you knew the value of each action, then it would be trivial to solve the $k$-armed bandit problem: you would always select the action with highest value, $A_t = \arg\max_a q_*(a)$. We assume that you do not know the action values with certainty, although you may have estimates. We denote the estimated value of action $a$ at time step $t$ as $Q_t(a)$. We would like $Q_t(a)$ to be close to $q_*(a)$:

$$
Q_t(a) \approx q_*(a).
$$

If you maintain estimates of the action values, then at any time step $t$ there is at least one action whose estimated value is greatest. We call these the **greedy actions**:

$$
A_t^{\text{greedy}} \in \arg\max_a Q_t(a).
$$

When you select one of these actions, we say that you are **exploiting** your current knowledge of the values of the actions. If instead you select one of the nongreedy actions, $a \notin \arg\max_a Q_t(a)$, then we say you are **exploring**, because this enables you to improve your estimate $Q_t(a)$ of the nongreedy action’s value. Exploitation is the right thing to do to maximize the expected reward $\mathbb{E}[R_t]$ on the one step, but exploration may produce the greater total reward $\sum_{t=1}^{T} R_t$ in the long run.

For example, suppose a greedy action’s value is known with certainty, while several other actions are estimated to be nearly as good but with substantial uncertainty. The uncertainty is such that at least one of these other actions probably is actually better than the greedy action, i.e. $q_*(a) > q_*(A_t^{\text{greedy}})$ for some nongreedy $a$, but you don’t know which one. If you have many time steps ahead on which to make action selections, then it may be better to explore the nongreedy actions and discover which of them are better than the greedy action. Reward is lower in the short run, during exploration, but higher in the long run because after you have discovered the better actions, you can exploit them many times. Because it is not possible both to explore and to exploit with any single action selection, one often refers to the “conflict” between exploration and exploitation.

## Action-value Methods

We begin by looking more closely at methods for estimating the values of actions and for using the estimates to make action selection decisions, which we collectively call **action-value methods**.

Every action-value method therefore has two parts. The first is an **estimation rule**, which uses the rewards observed so far to compute estimates $Q_t(a) \approx q_*(a)$. The second is a **selection rule**, which uses those estimates to choose the next action $A_t$, deciding on each step whether to exploit (pick an action in $\arg\max_a Q_t(a)$) or explore.

Recall that the true value $q_*(a)$ of an action is the mean reward when that action is selected. One natural way to estimate this is by averaging the rewards actually received:

$$
Q_t(a) \doteq \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbf{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbf{1}_{A_i = a}},
$$

where $\mathbf{1}_{\text{predicate}}$ denotes the random variable that is $1$ if the predicate is true and $0$ if it is not. If the denominator is zero, then we instead define $Q_t(a)$ as some default value, such as $Q_t(a) = 0$. As the denominator goes to infinity, by the law of large numbers, $Q_t(a)$ converges to $q_*(a)$:

$$
\sum_{i=1}^{t-1} \mathbf{1}_{A_i = a} \to \infty \implies Q_t(a) \to q_*(a).
$$

We call this the **sample-average method** for estimating action values because each estimate is an average of the sample of relevant rewards. Of course this is just one way to estimate action values, and not necessarily the best one.

The simplest action selection rule is to select one of the actions with the highest estimated value, that is, one of the greedy actions. If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly. We write this **greedy action selection** method as

$$
A_t \doteq \arg\max_a Q_t(a),
$$

where $\arg\max_a$ denotes the action $a$ for which the expression that follows is maximized (with ties broken arbitrarily).

A simple alternative is to behave greedily most of the time, but every once in a while, say with small probability $\varepsilon$, instead select randomly from among all the actions with equal probability, independently of the action-value estimates:

$$
A_t = \begin{cases} \arg\max_a Q_t(a) & \text{with probability } 1 - \varepsilon, \\ \text{a uniformly random action} & \text{with probability } \varepsilon. \end{cases}
$$

We call methods using this near-greedy action selection rule **$\varepsilon$-greedy methods**. An advantage of these methods is that, in the limit as the number of steps increases, every action will be sampled an infinite number of times, thus ensuring that all the $Q_t(a)$ converge to their respective $q_*(a)$.

These are just asymptotic guarantees, however, and say little about the practical effectiveness of the methods.

## Incremental Implementation

To simplify notation we concentrate on a single action. Let $R_i$ now denote the reward received after the $i$th selection of this action, and let $Q_n$ denote the estimate of its action value after it has been selected $n - 1$ times, which we can now write simply as

$$
Q_n \doteq \frac{R_1 + R_2 + \cdots + R_{n-1}}{n - 1}.
$$

The obvious implementation would be to maintain a record of all the rewards and then perform this computation whenever the estimated value was needed. However, if this is done, then the memory and computational requirements would grow over time as more rewards are seen. Each additional reward would require additional memory to store it and additional computation to compute the sum in the numerator.

As you might suspect, this is not really necessary. It is easy to devise **incremental formulas** for updating averages with small, constant computation required to process each new reward. Given $Q_n$ and the $n$th reward, $R_n$, the new average of all $n$ rewards can be computed by

$$
\begin{aligned}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^{n} R_i \\
&= \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right) \\
&= \frac{1}{n} \left( R_n + (n - 1) \frac{1}{n - 1} \sum_{i=1}^{n-1} R_i \right) \\
&= \frac{1}{n} \Big( R_n + (n - 1) Q_n \Big) \\
&= \frac{1}{n} \Big( R_n + n Q_n - Q_n \Big) \\
&= Q_n + \frac{1}{n} \Big[ R_n - Q_n \Big],
\end{aligned}
$$

which holds even for $n = 1$, obtaining $Q_2 = R_1$ for arbitrary $Q_1$:

$$
Q_2 = Q_1 + \frac{1}{1} \Big[ R_1 - Q_1 \Big] = R_1.
$$

This implementation requires memory only for $Q_n$ and $n$, and only the small computation of the update above for each new reward.

This update rule has a form that occurs frequently throughout reinforcement learning. The general form is

$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \Big[ \text{Target} - \text{OldEstimate} \Big].
$$

The expression $\big[ \text{Target} - \text{OldEstimate} \big]$ is an **error** in the estimate. It is reduced by taking a step toward the “Target.” The target is presumed to indicate a desirable direction in which to move, though it may be noisy. In the case above, for example, the target is the $n$th reward, $R_n$:

$$
\underbrace{Q_{n+1}}_{\text{NewEstimate}} = \underbrace{Q_n}_{\text{OldEstimate}} + \underbrace{\frac{1}{n}}_{\text{StepSize}} \Big[ \underbrace{R_n}_{\text{Target}} - \underbrace{Q_n}_{\text{OldEstimate}} \Big].
$$

Note that the step-size parameter ($\text{StepSize}$) used in the incremental method changes from time step to time step. In processing the $n$th reward for action $a$, the method uses the step-size parameter $\frac{1}{n}$.

Pseudocode for a complete bandit algorithm using incrementally computed sample averages and $\varepsilon$-greedy action selection is shown in the box below. The function $\text{bandit}(a)$ is assumed to take an action and return a corresponding reward.

!!! note "A simple bandit algorithm"
    **Initialize**, for $a = 1$ to $k$:

    $$
    \begin{aligned}
    Q(a) &\leftarrow 0 \\
    N(a) &\leftarrow 0
    \end{aligned}
    $$

    **Loop forever:**

    $$
    \begin{aligned}
    A &\leftarrow \begin{cases} \arg\max_a Q(a) & \text{with probability } 1 - \varepsilon \quad \text{(breaking ties randomly)} \\ \text{a random action} & \text{with probability } \varepsilon \end{cases} \\
    R &\leftarrow \text{bandit}(A) \\
    N(A) &\leftarrow N(A) + 1 \\
    Q(A) &\leftarrow Q(A) + \frac{1}{N(A)} \Big[ R - Q(A) \Big]
    \end{aligned}
    $$

## Tracking a Nonstationary Problem

The averaging methods discussed so far are appropriate for **stationary** bandit problems, that is, for bandit problems in which the reward probabilities do not change over time. As noted earlier, we often encounter reinforcement learning problems that are effectively **nonstationary**. In such cases it makes sense to give more weight to recent rewards than to long-past rewards. One of the most popular ways of doing this is to use a constant step-size parameter. For example, the incremental update rule for updating an average $Q_n$ of the $n - 1$ past rewards is modified to be

$$
Q_{n+1} \doteq Q_n + \alpha \Big[ R_n - Q_n \Big],
$$

where the step-size parameter $\alpha \in (0, 1]$ is constant. This results in $Q_{n+1}$ being a weighted average of past rewards and the initial estimate $Q_1$:

$$
\begin{aligned}
Q_{n+1} &= Q_n + \alpha \Big[ R_n - Q_n \Big] \\
&= \alpha R_n + (1 - \alpha) Q_n \\
&= \alpha R_n + (1 - \alpha) \big[ \alpha R_{n-1} + (1 - \alpha) Q_{n-1} \big] \\
&= \alpha R_n + (1 - \alpha) \alpha R_{n-1} + (1 - \alpha)^2 Q_{n-1} \\
&= \alpha R_n + (1 - \alpha) \alpha R_{n-1} + (1 - \alpha)^2 \alpha R_{n-2} + \cdots + (1 - \alpha)^{n-1} \alpha R_1 + (1 - \alpha)^n Q_1 \\
&= (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n-i} R_i.
\end{aligned}
$$

We call this a **weighted average** because the sum of the weights is

$$
(1 - \alpha)^n + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n-i} = 1.
$$

To check this, substitute $j = n - i$ in the sum. As $i$ goes from $1$ to $n$, $j$ goes from $n - 1$ down to $0$, so the sum is a geometric series with ratio $1 - \alpha$:

$$
\sum_{i=1}^{n} \alpha (1 - \alpha)^{n-i} = \alpha \sum_{j=0}^{n-1} (1 - \alpha)^j = \alpha \cdot \frac{1 - (1 - \alpha)^n}{1 - (1 - \alpha)} = 1 - (1 - \alpha)^n.
$$

Adding the weight $(1 - \alpha)^n$ on $Q_1$ gives exactly $1$.

Note that the weight, $\alpha (1 - \alpha)^{n-i}$, given to the reward $R_i$ depends on how many rewards ago, $n - i$, it was observed. The quantity $1 - \alpha$ is less than $1$, and thus the weight given to $R_i$ decreases as the number of intervening rewards increases. In fact, the weight decays exponentially according to the exponent on $1 - \alpha$. Accordingly, this is sometimes called an **exponential recency-weighted average**.

For example, with $\alpha = 0.1$ the weights on the most recent rewards are

| Reward | Rewards ago, $n - i$ | Weight $\alpha (1 - \alpha)^{n-i}$ |
|---|---|---|
| $R_n$ | $0$ | $0.1$ |
| $R_{n-1}$ | $1$ | $0.09$ |
| $R_{n-2}$ | $2$ | $0.081$ |
| $R_{n-10}$ | $10$ | $\approx 0.035$ |
| $R_{n-50}$ | $50$ | $\approx 0.0005$ |

So a reward from $50$ steps ago counts about $200$ times less than the latest one.

## Gradient Bandit Algorithms

So far in this chapter we have considered methods that estimate action values and use those estimates to select actions. This is often a good approach, but it is not the only one possible. In this section we consider learning a numerical **preference** for each action $a$, which we denote $H_t(a) \in \mathbb{R}$. The larger the preference, the more often that action is taken, but the preference has no interpretation in terms of reward. Only the relative preference of one action over another is important; if we add $1000$ to all the action preferences there is no effect on the action probabilities, which are determined according to a **soft-max distribution** (i.e., Gibbs or Boltzmann distribution) as follows:

$$
\Pr\{A_t = a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} \doteq \pi_t(a),
$$

where here we have also introduced a useful new notation, $\pi_t(a)$, for the probability of taking action $a$ at time $t$. Initially all action preferences are the same (e.g., $H_1(a) = 0$, for all $a$) so that all actions have an equal probability of being selected, $\pi_1(a) = \frac{1}{k}$.

To see why adding a constant has no effect, add $c$ to every preference. The common factor $e^c$ cancels:

$$
\frac{e^{H_t(a) + c}}{\sum_{b=1}^{k} e^{H_t(b) + c}} = \frac{e^{c} \, e^{H_t(a)}}{e^{c} \sum_{b=1}^{k} e^{H_t(b)}} = \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} = \pi_t(a).
$$

There is a natural learning algorithm for soft-max action preferences based on the idea of stochastic gradient ascent. On each step, after selecting action $A_t$ and receiving the reward $R_t$, the action preferences are updated by:

$$
\begin{aligned}
H_{t+1}(A_t) &\doteq H_t(A_t) + \alpha \big( R_t - \bar{R}_t \big) \big( 1 - \pi_t(A_t) \big), && \text{and} \\
H_{t+1}(a) &\doteq H_t(a) - \alpha \big( R_t - \bar{R}_t \big) \, \pi_t(a), && \text{for all } a \neq A_t,
\end{aligned}
$$

where $\alpha > 0$ is a step-size parameter, and $\bar{R}_t \in \mathbb{R}$ is the average of the rewards up to but not including time $t$ (with $\bar{R}_1 \doteq R_1$), which can be computed incrementally. The $\bar{R}_t$ term serves as a **baseline** with which the reward is compared. If the reward is higher than the baseline, $R_t > \bar{R}_t$, then the probability of taking $A_t$ in the future is increased, and if the reward is below baseline, $R_t < \bar{R}_t$, then the probability is decreased. The non-selected actions move in the opposite direction.

Both updates can be written as one rule using the indicator $\mathbf{1}_{a = A_t}$, which is $1$ for the selected action and $0$ for the rest:

$$
H_{t+1}(a) \doteq H_t(a) + \alpha \big( R_t - \bar{R}_t \big) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big), \qquad \text{for all } a.
$$

!!! note "Why $1 - \pi_t(A_t)$ for the selected action and $\pi_t(a)$ for the others?"
    Each term has a simple meaning.

    - $1 - \pi_t(A_t)$ is **how much room the selected action has left to grow**. If $A_t$ is already taken with probability $0.99$, there is little left to gain, so its preference moves only a little. If it had probability $0.1$, it moves a lot.
    - $\pi_t(a)$ means each non-selected action **gives up probability in proportion to how much it has**. An action with probability $0.4$ is pushed down a lot, while one with $0.01$ barely moves. So the probability that $A_t$ gains comes mostly from the actions that currently hold the most.

!!! note "Why do the selected and non-selected actions move in opposite directions?"
    The probabilities $\pi_t(a)$ must always sum to $1$, so one action cannot become more likely without the others becoming less likely. Raising $H_t(A_t)$ while lowering the rest is the most direct way to shift probability toward $A_t$.

    The direction is not fixed, though. It depends on the sign of $R_t - \bar{R}_t$: if the reward was **below** the baseline, the selected action goes **down** and the others go **up**.

    In fact, the changes to the preferences always sum to zero:

    $$
    \sum_{a} \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big) = 1 - \sum_{a} \pi_t(a) = 1 - 1 = 0.
    $$

    So the update never shifts all preferences up or down together, which would have no effect anyway, since adding a constant to every preference leaves $\pi_t$ unchanged. Every step goes entirely into changing their relative sizes.

    For example, take $k = 3$ with $\pi_t = (0.5, 0.3, 0.2)$. Action $1$ is selected, and suppose $\alpha (R_t - \bar{R}_t) = 1$. The preferences change by

    $$
    \Delta H = (1 - 0.5, \; -0.3, \; -0.2) = (+0.5, \; -0.3, \; -0.2), \qquad \text{sum} = 0,
    $$

    and the new probabilities are about $(0.68, \; 0.18, \; 0.14)$. Action $1$ went from $0.5$ to $0.68$. Action $2$ lost $0.12$ and action $3$ lost $0.06$, so the action holding more probability gave up more.

!!! note "Isn't there a risk of a runaway effect, where one action gets picked again and again?"
    Two things keep this from happening.

    1. **The baseline tracks the rewards.** A single reward can land below the baseline, $R_t < \bar{R}_t$, and then the selected action's preference goes down, so no action is pushed up on every step. And as one action comes to be picked most of the time, the baseline $\bar{R}_t$ moves toward that action's typical reward. Its rewards then land above and below the baseline about equally often, so the ups and downs cancel out and there is no steady climb. This only works because the baseline moves.
    2. **The step shrinks near certainty.** The selected action's increase is scaled by $1 - \pi_t(A_t)$, which goes to $0$ as $\pi_t(A_t) \to 1$.

Later, when we compare this update with exact gradient ascent, we will need to tell the two apart. So we will sometimes write the result of this algorithm's update as $H_{t+1}^{\text{alg}}(a)$:

$$
H_{t+1}^{\text{alg}}(a) \doteq H_t(a) + \alpha \big( R_t - \bar{R}_t \big) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big).
$$

The label "alg" just means "the update the gradient bandit algorithm actually performs". It is the same rule as above. Because it depends on the randomly selected action $A_t$ and the sampled reward $R_t$, $H_{t+1}^{\text{alg}}(a)$ is a random quantity: running the same step again from the same $H_t$ could give a different result.

### The Bandit Gradient Algorithm as Stochastic Gradient Ascent

Three different action symbols appear in the derivation below, so it helps to keep them apart:

| Symbol | What it is | Role |
|---|---|---|
| $A_t$ | random variable | the action actually selected at step $t$ |
| $x$ | sum index | runs over all possible actions $1, \dots, k$ |
| $a$ | fixed action | whichever action's preference we're computing the update for |

The definition of expectation connects $A_t$ and $x$: to average any quantity $f(A_t)$, we sum over every possible action $x$, weighted by its probability,

$$
\mathbb{E}\big[ f(A_t) \big] = \sum_{x=1}^{k} \Pr\{A_t = x\} \, f(x) = \sum_{x=1}^{k} \pi_t(x) \, f(x).
$$

Now, let's get to the derivation.

In exact **gradient ascent**, each action preference $H_t(a)$ would be incremented in proportion to the **increment’s effect on performance**:

$$
H_{t+1}^{\text{exact}}(a) \doteq H_t(a) + \alpha \frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)},
$$

where the **measure of performance here is the expected reward**:

$$
\mathbb{E}[R_t] = \sum_x \pi_t(x) \, q_*(x),
$$

**The measure of the increment’s effect is the partial derivative of this performance measure with respect to the action preference.** Of course, it is not possible to implement gradient ascent exactly in our case because by assumption we do not know the $q_*(x)$.

Recall that $q_*(x) = \mathbb{E}[R_t \mid A_t = x]$ is the true mean reward of action $x$: a fixed real number for each action, set by the environment and unknown to the agent. Unlike the reward $R_t$ it is not random, and unlike $\pi_t(x)$ it does not change with $t$. So $\mathbb{E}[R_t]$ is the average of these true values, weighted by how often the agent currently selects each action.

Any single update of the gradient bandit algorithm,

$$
H_{t+1}^{\text{alg}}(a) = H_t(a) + \alpha \big( R_t - \bar{R}_t \big) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big),
$$

is noisy. It is based on one randomly selected action $A_t$ and one sampled reward $R_t$, not on the unknown true values $q_*(x)$, so it does not point exactly along the true gradient.

!!! note "What we want to prove"
    Starting from the same preferences $H_t$, we compare two updates:

    - $H_{t+1}^{\text{exact}}(a)$, the exact gradient ascent update. It is a **fixed number**: given $H_t$ there is only one possible result. But we cannot compute it, because it needs the unknown true values $q_*(x)$.
    - $H_{t+1}^{\text{alg}}(a)$, the gradient bandit update. It is a **random variable**: its value depends on which action $A_t$ is selected and which reward $R_t$ comes back. But we can compute it, because it only needs $A_t$ and $R_t$.

    We want to show that the algorithm is right **on average**:

    $$
    \mathbb{E}\big[ H_{t+1}^{\text{alg}}(a) \big] = H_{t+1}^{\text{exact}}(a), \qquad \text{for all } a,
    $$

    where the expectation is over $A_t$ and $R_t$, with $H_t$ and $\bar{R}_t$ held fixed. Subtracting $H_t(a)$ from both sides and dividing by $\alpha$, this is the same as showing

    $$
    \mathbb{E}\Big[ \big( R_t - \bar{R}_t \big) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big) \Big] = \frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)},
    $$

    which is what the steps below establish.

    This does **not** mean $H_{t+1}^{\text{alg}}(a) = H_{t+1}^{\text{exact}}(a)$. A single update is usually off, sometimes even in the wrong direction; it is only its average that matches. Like a fair die, whose average roll is $3.5$ even though no single roll is ever $3.5$.

First we take a closer look at the exact performance gradient ($a$ is the action whose preference we're computing the update for):

$$
\begin{aligned}
\frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)} \left[ \sum_x \pi_t(x) \, q_*(x) \right] \\
&= \sum_x \frac{\partial}{\partial H_t(a)} \Big[ \pi_t(x) \, q_*(x) \Big] && \text{(derivative of a sum)} \\
&= \sum_x \left[ q_*(x) \frac{\partial \pi_t(x)}{\partial H_t(a)} + \pi_t(x) \frac{\partial q_*(x)}{\partial H_t(a)} \right] && \text{(product rule)} \\
&= \sum_x q_*(x) \frac{\partial \pi_t(x)}{\partial H_t(a)} && \left( \tfrac{\partial q_*(x)}{\partial H_t(a)} = 0 \right) \\
&= \sum_x \big( q_*(x) - B_t \big) \frac{\partial \pi_t(x)}{\partial H_t(a)},
\end{aligned}
$$

The term with $\frac{\partial q_*(x)}{\partial H_t(a)}$ vanishes because $q_*(x)$ is set by the environment, not by the agent: changing the preferences changes the probabilities $\pi_t(x) = \frac{e^{H_t(x)}}{\sum_{y=1}^{k} e^{H_t(y)}}$, but not the true mean rewards.

Here, $B_t$ is called the **baseline**. It can be any scalar that does not depend on $x$. We can include a baseline here without changing the equality because the gradient sums to zero over all the actions, $\sum_x \frac{\partial \pi_t(x)}{\partial H_t(a)} = 0$. As $H_t(a)$ is changed, some actions’ probabilities go up and some go down, but the sum of the changes must be zero because the sum of the probabilities is always one.

Next we multiply each term of the sum by $\pi_t(x) / \pi_t(x)$:

$$
\frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)} = \sum_x \pi_t(x) \big( q_*(x) - B_t \big) \frac{\partial \pi_t(x)}{\partial H_t(a)} \Big/ \pi_t(x).
$$

The equation is now in the form of an expectation, summing over all possible values $x$ of the random variable $A_t$, then multiplying by the probability of taking those values. Thus:

$$
\frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E}\left[ \big( q_*(A_t) - B_t \big) \frac{\partial \pi_t(A_t)}{\partial H_t(a)} \Big/ \pi_t(A_t) \right].
$$

To shorten the next few steps, write $g(A_t) = \frac{\partial \pi_t(A_t)}{\partial H_t(a)} \Big/ \pi_t(A_t)$, which depends only on the action $A_t$ (taken at timestamp $t$). We now choose the baseline $B_t = \bar{R}_t$. This is allowed because $\bar{R}_t$ is computed only from rewards before time $t$, so it does not depend on $A_t$. Then, splitting the expectation by linearity,

$$
\begin{aligned}
\frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)} &= \mathbb{E}\Big[ \big( q_*(A_t) - \bar{R}_t \big) \, g(A_t) \Big] \\
&= \mathbb{E}\big[ q_*(A_t) \, g(A_t) \big] - \bar{R}_t \, \mathbb{E}\big[ g(A_t) \big].
\end{aligned}
$$

Next we replace the unknown $q_*(A_t)$ with the observed reward $R_t$. How can we do that? The next few steps show it.

Consider $\mathbb{E}\big[ R_t \, g(A_t) \big]$.

We can compute the average of $R_t \, g(A_t)$ in two stages. First, find the average for each action separately. Then, combine those averages, weighting each by how often that action is picked.

For example, take two actions, each picked half the time, $\pi_t = (0.5, 0.5)$, with $g(1) = 2$ and $g(2) = -1$:

- Action $1$ always pays $1$.
- Action $2$ pays $0$ or $6$, each with probability $\tfrac{1}{2}$.

*Stage 1: average for each action on its own.*

| Action $x$ | Possible $R_t \, g(x)$ | Average given $A_t = x$ |
|---|---|---|
| $1$ | $1 \cdot 2 = 2$ | $2$ |
| $2$ | $0 \cdot (-1) = 0$ or $6 \cdot (-1) = -6$ | $\tfrac{0 + (-6)}{2} = -3$ |

*Stage 2: combine, weighting by how often each action is picked.*

$$
\mathbb{E}\big[ R_t \, g(A_t) \big] = \underbrace{0.5}_{\pi_t(1)} \cdot \underbrace{2}_{\text{avg. for } 1} + \underbrace{0.5}_{\pi_t(2)} \cdot \underbrace{(-3)}_{\text{avg. for } 2} = -0.5.
$$

That is all the **law of total expectation** says. In general:

$$
\mathbb{E}\big[ R_t \, g(A_t) \big] = \sum_x \underbrace{\Pr\{A_t = x\}}_{\text{how often } x \text{ is picked}} \cdot \underbrace{\mathbb{E}\big[ R_t \, g(A_t) \mid A_t = x \big]}_{\text{average when } x \text{ is picked}}.
$$

$$
\begin{aligned}
\mathbb{E}\big[ R_t \, g(A_t) \big] &= \sum_x \Pr\{A_t = x\} \, \mathbb{E}\big[ R_t \, g(A_t) \mid A_t = x \big] && \text{(law of total expectation)} \\
&= \sum_x \pi_t(x) \, \mathbb{E}\big[ R_t \, g(A_t) \mid A_t = x \big] && \left( \Pr\{A_t = x\} = \pi_t(x) \right) \\
&= \sum_x \pi_t(x) \, \mathbb{E}\big[ R_t \, g(x) \mid A_t = x \big] && \text{(given } A_t = x \text{, } g(A_t) = g(x)\text{)} \\
&= \sum_x \pi_t(x) \, g(x) \, \mathbb{E}\big[ R_t \mid A_t = x \big] && \text{($g(x)$ is a fixed number once $x$ is fixed)} \\
&= \sum_x \pi_t(x) \, g(x) \, q_*(x) && \left( \mathbb{E}[R_t \mid A_t = x] = q_*(x) \right) \\
&= \mathbb{E}\big[ q_*(A_t) \, g(A_t) \big]. && \text{(definition of expectation)}
\end{aligned}
$$

So $R_t$ can stand in for $q_*(A_t)$ inside the expectation. Substituting this in and recombining,

$$
\begin{aligned}
\frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)} &= \mathbb{E}\big[ R_t \, g(A_t) \big] - \bar{R}_t \, \mathbb{E}\big[ g(A_t) \big] \\
&= \mathbb{E}\Big[ \big( R_t - \bar{R}_t \big) \, g(A_t) \Big] \\
&= \mathbb{E}\left[ \big( R_t - \bar{R}_t \big) \frac{\partial \pi_t(A_t)}{\partial H_t(a)} \Big/ \pi_t(A_t) \right].
\end{aligned}
$$

Shortly we will establish that $\frac{\partial \pi_t(A_t)}{\partial H_t(a)} = \pi_t(A_t) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big)$, where $\mathbf{1}_{a = A_t}$ is defined to be $1$ if $a = A_t$, else $0$.

**Assuming that is true for now**, we have

$$
\begin{aligned}
&= \mathbb{E}\Big[ \big( R_t - \bar{R}_t \big) \, \pi_t(A_t) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big) \big/ \pi_t(A_t) \Big] \\
&= \mathbb{E}\Big[ \big( R_t - \bar{R}_t \big) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big) \Big].
\end{aligned}
$$

Thus,

$$
\mathbb{E}\Big[ \big( R_t - \bar{R}_t \big) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big) \Big] = \frac{\partial \, \mathbb{E}[R_t]}{\partial H_t(a)},
$$

Now, it remains only to show that $\frac{\partial \pi_t(A_t)}{\partial H_t(a)} = \pi_t(A_t) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big)$, as we assumed. Recall the standard quotient rule for derivatives:

$$
\frac{\partial}{\partial u} \left[ \frac{f(u)}{g(u)} \right] = \frac{\frac{\partial f(u)}{\partial u} g(u) - f(u) \frac{\partial g(u)}{\partial u}}{g(u)^2}.
$$

Using this, we can write

$$
\begin{aligned}
\frac{\partial \pi_t(A_t)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)} \pi_t(A_t) \\
&= \frac{\partial}{\partial H_t(a)} \left[ \frac{e^{H_t(A_t)}}{\sum_{y=1}^{k} e^{H_t(y)}} \right] \\
&= \frac{\frac{\partial e^{H_t(A_t)}}{\partial H_t(a)} \sum_{y=1}^{k} e^{H_t(y)} - e^{H_t(A_t)} \frac{\partial \sum_{y=1}^{k} e^{H_t(y)}}{\partial H_t(a)}}{\left( \sum_{y=1}^{k} e^{H_t(y)} \right)^2} && \text{(by the quotient rule)} \\
&= \frac{\mathbf{1}_{a = A_t} \, e^{H_t(A_t)} \sum_{y=1}^{k} e^{H_t(y)} - e^{H_t(A_t)} e^{H_t(a)}}{\left( \sum_{y=1}^{k} e^{H_t(y)} \right)^2} && \left( \text{because } \tfrac{\partial e^u}{\partial u} = e^u \right) \\
&= \frac{\mathbf{1}_{a = A_t} \, e^{H_t(A_t)}}{\sum_{y=1}^{k} e^{H_t(y)}} - \frac{e^{H_t(A_t)} e^{H_t(a)}}{\left( \sum_{y=1}^{k} e^{H_t(y)} \right)^2} \\
&= \mathbf{1}_{a = A_t} \, \pi_t(A_t) - \pi_t(A_t) \, \pi_t(a) \\
&= \pi_t(A_t) \big( \mathbf{1}_{a = A_t} - \pi_t(a) \big). && \text{Q.E.D.}
\end{aligned}
$$

**We have just shown that the expected update of the gradient bandit algorithm is equal to the gradient of expected reward, and thus that the algorithm is an instance of stochastic gradient ascent. This assures us that the algorithm has robust convergence properties.**

Note that we did not require any properties of the reward baseline other than that it does not depend on the selected action. For example, we could have set it to zero, or to $1000$, and the algorithm would still be an instance of stochastic gradient ascent.

## Exercises

**1.** In $\varepsilon$-greedy action selection, for the case of two actions and $\varepsilon = 0.5$, what is the probability that the greedy action is selected?

**Solution:** There are two ways the greedy action can be selected:

1. We exploit, which happens with probability $1 - \varepsilon$, and then we always pick the greedy action.
2. We explore, which happens with probability $\varepsilon$, and then we pick one of the $k$ actions uniformly at random. The greedy action is one of them, so it is picked with probability $\frac{1}{k}$.

Adding the two cases,

$$
P(A_t = \text{greedy}) = (1 - \varepsilon) + \varepsilon \cdot \frac{1}{k}.
$$

With $k = 2$ and $\varepsilon = 0.5$,

$$
P(A_t = \text{greedy}) = (1 - 0.5) + 0.5 \cdot \frac{1}{2} = 0.5 + 0.25 = 0.75.
$$

So the greedy action is selected $75\%$ of the time, not $50\%$, because exploration picks the greedy action by chance half the time.

**2.** Consider a $k$-armed bandit problem with $k = 4$ actions, denoted $1, 2, 3$, and $4$. Consider applying to this problem a bandit algorithm using $\varepsilon$-greedy action selection, sample-average action-value estimates, and initial estimates of $Q_1(a) = 0$, for all $a$. Suppose the initial sequence of actions and rewards is $A_1 = 1, R_1 = -1, A_2 = 2, R_2 = 1, A_3 = 2, R_3 = -2, A_4 = 2, R_4 = 2, A_5 = 3, R_5 = 0$. On some of these time steps the $\varepsilon$ case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?

**Solution:** On each step, we write down the estimates $Q_t(a)$ *before* the action is taken, find the greedy actions $\arg\max_a Q_t(a)$, and check whether $A_t$ is one of them. After each step, only the estimate of the action that was taken changes: it becomes the average of all rewards received for that action so far.

| $t$ | $Q_t(1)$ | $Q_t(2)$ | $Q_t(3)$ | $Q_t(4)$ | Greedy actions | $A_t$ | $R_t$ | Greedy? |
|---|---|---|---|---|---|---|---|---|
| 1 | $0$ | $0$ | $0$ | $0$ | $\{1, 2, 3, 4\}$ | $1$ | $-1$ | yes |
| 2 | $-1$ | $0$ | $0$ | $0$ | $\{2, 3, 4\}$ | $2$ | $1$ | yes |
| 3 | $-1$ | $1$ | $0$ | $0$ | $\{2\}$ | $2$ | $-2$ | yes |
| 4 | $-1$ | $-\tfrac{1}{2}$ | $0$ | $0$ | $\{3, 4\}$ | $2$ | $2$ | **no** |
| 5 | $-1$ | $\tfrac{1}{3}$ | $0$ | $0$ | $\{2\}$ | $3$ | $0$ | **no** |

Each estimate is the average of the rewards received for that action so far, or the default $0$ if the action has not been taken yet. Only the action that was taken changes from one step to the next.

After step $1$ (action $1$ was taken):

$$
Q_2(1) = \frac{R_1}{1} = \frac{-1}{1} = -1.
$$

After steps $2$, $3$ and $4$ (action $2$ was taken each time):

$$
Q_3(2) = \frac{R_2}{1} = \frac{1}{1} = 1, \qquad Q_4(2) = \frac{R_2 + R_3}{2} = \frac{1 + (-2)}{2} = -\frac{1}{2}, \qquad Q_5(2) = \frac{R_2 + R_3 + R_4}{3} = \frac{1 + (-2) + 2}{3} = \frac{1}{3}.
$$

After step $5$ (action $3$ was taken), the estimates for all four actions are

$$
\begin{aligned}
Q_6(1) &= \frac{R_1}{1} = \frac{-1}{1} = -1, \\
Q_6(2) &= \frac{R_2 + R_3 + R_4}{3} = \frac{1 + (-2) + 2}{3} = \frac{1}{3}, \\
Q_6(3) &= \frac{R_5}{1} = \frac{0}{1} = 0, \\
Q_6(4) &= 0 \quad \text{(never taken, so it keeps the default value).}
\end{aligned}
$$

So after these five steps, the greedy action is $\arg\max_a Q_6(a) = 2$.

On which time steps did the $\varepsilon$ case definitely occur? On steps $t = 4$ and $t = 5$. On both, the action taken was not greedy. A greedy step can only pick a greedy action, so these must have been random picks.

On which time steps could it possibly have occurred? On every step, $t = 1, \dots, 5$. When exploring, we pick uniformly from **all** $k$ actions, greedy ones included. So on steps $1$, $2$ and $3$, the action taken could have come from either a greedy step or a random pick that happened to land on a greedy action. Looking at the actions alone, we can't tell which.

**3. Varying Step Sizes:** If the step-size parameters, $\alpha_n$, are not constant, then the estimate $Q_n$ is a weighted average of previously received rewards with a weighting different from that given by the constant-$\alpha$ formula above. What is the weighting on each prior reward for the general case, analogous to that formula, in terms of the sequence of step-size parameters?

**Solution:** Now the step size used for the $n$th reward is $\alpha_n$, so the update is

$$
Q_{n+1} = Q_n + \alpha_n \Big[ R_n - Q_n \Big] = \alpha_n R_n + (1 - \alpha_n) Q_n.
$$

We unroll it exactly as in the constant case, except that each step keeps its own $\alpha$:

$$
\begin{aligned}
Q_{n+1} &= \alpha_n R_n + (1 - \alpha_n) Q_n \\
&= \alpha_n R_n + (1 - \alpha_n) \big[ \alpha_{n-1} R_{n-1} + (1 - \alpha_{n-1}) Q_{n-1} \big] \\
&= \alpha_n R_n + (1 - \alpha_n) \alpha_{n-1} R_{n-1} + (1 - \alpha_n)(1 - \alpha_{n-1}) Q_{n-1} \\
&= \alpha_n R_n + (1 - \alpha_n) \alpha_{n-1} R_{n-1} + \cdots + (1 - \alpha_n) \cdots (1 - \alpha_2) \, \alpha_1 R_1 + (1 - \alpha_n) \cdots (1 - \alpha_1) \, Q_1 \\
&= \left[ \prod_{j=1}^{n} (1 - \alpha_j) \right] Q_1 + \sum_{i=1}^{n} \left[ \alpha_i \prod_{j=i+1}^{n} (1 - \alpha_j) \right] R_i.
\end{aligned}
$$

So the weight on each prior reward $R_i$ is

$$
w_i = \alpha_i \prod_{j=i+1}^{n} (1 - \alpha_j),
$$

and the weight on the initial estimate $Q_1$ is $\prod_{j=1}^{n} (1 - \alpha_j)$.

Two checks:

1. **Constant step size**, $\alpha_j = \alpha$ for all $j$. Every factor is $1 - \alpha$, and there are $n - i$ of them, so $w_i = \alpha (1 - \alpha)^{n-i}$ and the weight on $Q_1$ is $(1 - \alpha)^n$. This is the constant-$\alpha$ formula.
2. **Sample averages**, $\alpha_j = \frac{1}{j}$. Then $1 - \alpha_j = \frac{j-1}{j}$, and the product telescopes:

    $$
    w_i = \frac{1}{i} \cdot \frac{i}{i+1} \cdot \frac{i+1}{i+2} \cdots \frac{n-1}{n} = \frac{1}{n}.
    $$

    Every reward gets the same weight $\frac{1}{n}$, and since $1 - \alpha_1 = 0$, the initial estimate $Q_1$ gets weight $0$. This is the plain average $Q_{n+1} = \frac{1}{n} \sum_{i=1}^{n} R_i$, as it should be.

**Do the weights always sum to 1?** Yes, for any sequence of step sizes. Each update $Q_{n+1} = \alpha_n R_n + (1 - \alpha_n) Q_n$ gives weight $\alpha_n$ to the new reward and $1 - \alpha_n$ to everything that was already in $Q_n$, and $\alpha_n + (1 - \alpha_n) = 1$. If the old weights summed to $1$, the new ones do too. The very first estimate $Q_1$ starts with weight $1$, so at every step

$$
\prod_{j=1}^{n} (1 - \alpha_j) + \sum_{i=1}^{n} \alpha_i \prod_{j=i+1}^{n} (1 - \alpha_j) = 1.
$$

What the step sizes decide is how that total weight of $1$ is shared out. For example, take $Q_1 = 0$ and three rewards $R_1 = 2$, $R_2 = 4$, $R_3 = 6$.

*Example 1: sample averages, $\alpha_n = \frac{1}{n}$* (the incremental implementation). Running the update,

$$
Q_2 = 0 + 1 \cdot (2 - 0) = 2, \qquad Q_3 = 2 + \tfrac{1}{2} (4 - 2) = 3, \qquad Q_4 = 3 + \tfrac{1}{3} (6 - 3) = 4.
$$

The weights are

| | $Q_1$ | $R_1$ | $R_2$ | $R_3$ | Sum |
|---|---|---|---|---|---|
| Weight | $0$ | $\tfrac{1}{3}$ | $\tfrac{1}{3}$ | $\tfrac{1}{3}$ | $1$ |

so $Q_4 = \tfrac{1}{3}(2 + 4 + 6) = 4$, the plain average. $Q_1$ gets weight $0$ because $1 - \alpha_1 = 0$: the first reward completely replaces the initial guess.

*Example 2: constant step size, $\alpha = 0.5$.* Running the update,

$$
Q_2 = 0 + 0.5 (2 - 0) = 1, \qquad Q_3 = 1 + 0.5 (4 - 1) = 2.5, \qquad Q_4 = 2.5 + 0.5 (6 - 2.5) = 4.25.
$$

The weights are $\alpha (1 - \alpha)^{n-i}$ for the rewards and $(1 - \alpha)^n$ for $Q_1$:

| | $Q_1$ | $R_1$ | $R_2$ | $R_3$ | Sum |
|---|---|---|---|---|---|
| Weight | $0.125$ | $0.125$ | $0.25$ | $0.5$ | $1$ |

so $Q_4 = 0.125 \cdot 0 + 0.125 \cdot 2 + 0.25 \cdot 4 + 0.5 \cdot 6 = 4.25$, matching the update. The most recent reward gets half the weight, each older one half as much as the next, and $Q_1$ never quite disappears.
