# Temporal-Difference Learning

## Background: Dynamic Programming and Monte Carlo

TD learning is easiest to understand as a mix of two older ideas, **dynamic programming (DP)** and **Monte Carlo (MC)** methods. All three solve the same problem, called **prediction** or **policy evaluation**: given a policy $\pi$, estimate its value function $v_\pi(s)$. They differ in what they need and in what they use as the "target" to move an estimate toward.

### Dynamic programming: use the model, look one step ahead

DP assumes we **know the model** $p(s', r \mid s, a)$ of the environment. It turns the [Bellman equation for $v_\pi$](finite_markov_decision_processes.md#policies-and-value-functions) into an update rule: start from any guess $V(s)$, and repeatedly replace each state's value with the right-hand side of the Bellman equation,

$$
V(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \Big[ r + \gamma \, V(s') \Big].
$$

Repeating this for every state, over and over, makes $V$ converge to $v_\pi$. Two features of this update matter for TD:

- **It averages over every possible next step**, using the known probabilities. This is called an *expected update*.
- **It updates a guess from other guesses:** the new $V(s)$ is built from the current estimates $V(s')$ of the next states, not from actual returns. This is called **bootstrapping**.

### Monte Carlo: no model, learn from complete episodes

MC methods need **no model**. They learn from experience: run an episode with policy $\pi$, wait until it ends, and then compute the actual return $G_t$ that followed each visited state. Since $v_\pi(s)$ is the *expected* return from $s$, averaging many observed returns from $s$ estimates it. Done incrementally, as in the [bandit chapter](multi_armed_bandits.md#incremental-implementation), this is

$$
V(S_t) \leftarrow V(S_t) + \alpha \Big[ G_t - V(S_t) \Big].
$$

- **It uses one actual sample**, not an average over all possibilities. This is called a *sample update*.
- **It does not bootstrap:** the target $G_t$ is a real return, so it must **wait until the episode ends** before $G_t$ is known.

### Where TD learning fits

TD learning takes one idea from each:

- like **MC**, it learns from sampled experience, without a model;
- like **DP**, it bootstraps, so it can update after a single step, without waiting for the episode to end.

| | Needs a model $p$? | Bootstraps? | Target for $V(S_t)$ | When it can update |
|---|---|---|---|---|
| Dynamic programming | yes | yes | $\sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma V(s') \big]$ | any time (no experience needed) |
| Monte Carlo | no | no | $G_t$, the actual return | at the end of the episode |
| TD learning | no | yes | $R_{t+1} + \gamma V(S_{t+1})$ | after one step |

The TD target in the last row is the MC idea, "use what actually happened", applied to just one step, with the rest of the return replaced by the DP idea, "use the current estimate of the next state's value". It comes straight from $G_t = R_{t+1} + \gamma G_{t+1}$: TD keeps the real reward $R_{t+1}$ and replaces the unknown remainder $G_{t+1}$ with the estimate $V(S_{t+1})$.