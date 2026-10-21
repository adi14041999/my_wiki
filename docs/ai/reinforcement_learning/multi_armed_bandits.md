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
