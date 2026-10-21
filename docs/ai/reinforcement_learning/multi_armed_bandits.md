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

**Example:** In $\varepsilon$-greedy action selection, for the case of two actions and $\varepsilon = 0.5$, what is the probability that the greedy action is selected?

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
