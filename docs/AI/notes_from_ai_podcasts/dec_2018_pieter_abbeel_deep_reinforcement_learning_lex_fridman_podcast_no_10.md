# Dec 2018. [Pieter Abbeel- Deep Reinforcement Learning | Lex Fridman Podcast no. 10](https://www.youtube.com/watch?v=l-mYLq6eZPY&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
16 Dec 2018.

## A very short introduction to Reinforcement Learning (RL)

Reinforcement Learning (RL) is a branch of machine learning where an autonomous agent learns to make decisions by interacting with an environment to maximize its cumulative reward.

### Core Components of RL

Every reinforcement learning system is built around five foundational elements:

1. Agent: The autonomous AI entity or decision-maker being trained.
2. Environment: The external world or simulation system the agent interacts with.
3. State ($S$): The current configuration or situation of the environment.
4. Action ($A$): All possible moves or decisions available to the agent.
5. Reward ($R$): Feedback from the environment evaluating the last action.

### The Feedback Loop

The learning process repeats sequentially across a standard cycle:

1. The agent observes the current State ($S_{t}$).
2. The agent selects and executes an Action ($A_{t}$) based on its current strategy.
3. The Environment changes and transitions to a new state ($S_{t+1}$).
4. The environment delivers a Reward ($R_{t+1}$) indicating the success of that move.

### Policy, value, exploration, and methods

**Policy** ($\pi$): the agent's strategy— a rule for choosing actions from states. It can be deterministic ($a = \pi(s)$) or stochastic ($\pi(a \mid s)$ = probability of action $a$ in state $s$).

**Value function**: an estimate of how good it is to be in a state (or to take an action in a state), measured as **expected cumulative future reward**. Common forms are $V(s)$ (state value function; “How good is it to be in state $s$?”) and $Q(s, a)$ (action value function or $Q$ function; “How good is it to take action $a$ in state $s$?”).

**Exploration** vs **exploitation**: the agent must balance trying new actions to discover better options (exploration) with using what it already knows pays off (exploitation). Too much exploitation → miss better strategies; too much exploration → waste reward on bad moves.

**Value-based methods**: learn $V$ or $Q$, then act by picking actions that look best (e.g. $\arg\max_a Q(s, a)$). Examples include Q-learning and DQN.

**Policy-based methods**: learn the policy $\pi$ directly, often by adjusting it to make rewarded actions more likely. **Policy-gradient** methods are the main example.

### ReLUs in deep RL networks

In **deep RL**, the policy (and often the value function) is not a lookup table but a **neural network**. State in → action probabilities or Q-values out. Hidden layers almost always use **ReLU** activations.

A **ReLU** (rectified linear unit) outputs the input if it is positive, and zero otherwise:

$$\mathrm{ReLU}(x) = \max(0, x).$$

So each neuron is either **on** (passing signal) or **off** (silent). Stacking many ReLUs gives a function that is **piecewise linear**: made of many flat segments rather than one smooth curve.

Why this matters for RL:

- The network is not memorizing “state 47,842 → turn left.” It learns **simple local rules** that hold across regions of similar states (e.g. “if tilted left, steer right”).
- Those rules are much like **feedback control** (small corrections based on what you sense), which is why simple policy-gradient updates can still learn surprisingly fast.

## Can robots have emotion?

After watching a computer play Go or solve some task, people ask whether it can *really* have emotion or relate to us. Yet spending time around robots already begins to produce that feeling. 

If a system uses **reinforcement learning** to optimize an objective, that objective could just as well be tied to how much a person enjoys interacting with it. An agent could optimize for *being fun to be around* and would then naturally grow more interactive and more pet- or person-like, acquiring those qualities automatically, provided the objective can be formalized.

The catch is the **reward**. How is it obtained from a human? Explicit numeric rewards are hard for a person to assign ("was that a 1.5 or a 0.7?"). A **comparison** is far easier. Saying that the last five minutes were nicer than the previous five. Using exactly this **preference feedback**, Paul Christiano and collaborators at OpenAI trained a one-legged MuJoCo Hopper to do a backflip. It was never told the goal, only which of two behaviors was better, and inferred from the comparisons what the person wanted. Interactive robots could likewise discover over time which behaviors are appreciated.

## Intuition behind RL

Why does RL need so many samples, and why does it work at all under sparse, delayed rewards? With a sparse reward, an agent might take a hundred actions and receive a single score, say, $3$, try again and get $2$, and now know only that the second sequence was somehow worse than the first, without knowing which of the hundred actions made the difference. That ambiguity is why **many experiences** are needed. Given enough of them, RL teases the signal apart by asking what is consistently present when the reward is high versus low. The **policy-gradient** update then makes the actions present when things went well more likely, and the actions present when things went poorly less likely.

!!! note "Beginner intuition: ReLUs and why simple updates work"
    Policy-gradient RL looks naive on paper (“reward good actions, punish bad ones”), yet it often learns faster than that story suggests. One intuition: the policy is usually a **ReLU network**, which is not one arbitrary function but a **patchwork of simple rules**.

    **Linear feedback control (thermostat analogy):** Keep a room at 20°C: too cold → heat up a bit; too hot → heat down a bit. That proportional correction is linear feedback control. It is simple and works well for many systems.

    **What ReLUs do:** A ReLU neuron is either on (passes signal) or off (zero). The network is many **flat pieces** glued together— **piecewise linear**. In each piece it acts like a small linear controller: “in this kind of situation, nudge the action this way.”

    **Tiling:** Think of all situations (states) as a floor covered with **tiles**. Turning one ReLU on or off changes behavior along **one direction** at a time, so neighboring regions are similar, not unrelated. Neighboring tiles **share** most of the same weights, so learning in one situation helps nearby ones. You are not memorizing every state from scratch.

## Time scales and credit assignment

Where the real world gets tricky, compared with the tasks RL has handled well, is **time scale**. A high-level decision, say, a student choosing to do a PhD, plays out as a long sequence of muscle-fiber contractions and relaxations. The decision to do a PhD is enormously abstract relative to those low-level actions, and bridging them is a [**credit assignment**](oct_2018_yoshua_bengio_deep_learning_lex_fridman_podcast_no_4.md#credit-assignment) problem far beyond any current RL algorithm. Closing that gap requires **hierarchical reasoning** at a level not yet available.

## Hierarchical reasoning approaches

Systems from 20–30 years ago could already reason over **long horizons**, but they were **not grounded** in the real world. Humans hand-designed logical, dynamical models with no tie to perception or real objects. Deep learning now fills that gap (sensors → understanding what is in the world), so it is a natural moment to reunite the two traditions.

### Deep learning + classical planning

One route **bolts** deep learning onto traditional planners via end-to-end training. Perception learns a representation that feeds a dynamical model used for planning. This is the direction of work such as **Causal InfoGAN** (e.g. Aviv Tamar and collaborators).

### High-level actions as latent variables

A more **information-theoretic** view: a high-level action is a **latent variable** that constrains (narrows down) the future. Choosing to drive to the gas station largely fixes the outcome (arriving there) long before the low-level steering and braking finish the trip. Learning this kind of hierarchy in practice is very difficult.

### Meta-learning hierarchy (RL²)

What is hierarchy actually *for*? Mainly **better credit assignment**, which in turn means **faster learning**. The **RL²** line of work ("learning to reinforcement-learn," led by Rocky Duan) skips hand-designed hierarchies and **meta-learns** whatever structure delivers that benefit. It does not yet scale to real-world settings, but it suggests hierarchical structure might be **learned** rather than engineered.

## Modularity

Vladimir Vapnik speaks of an “$E = mc^2$ for learning”— a general theory of intelligence. 

Evidence from neuroscience points to **modularity**: the brain is not a single undifferentiated blob.

People who are blind can repurpose cortex normally used for **vision** for other tasks; after some injuries, regions **rewire** and support new functions. Not every area is interchangeable, but much of the **neocortex** looks modular in this sense— similar structure reused for different skills.

A plausible AI analogue: build systems with the same kind of modularity, so capability grows mainly by **scaling size** (adding more of the same reusable building blocks).

## Math versus empirical trial and error

RL research lives in two modes at once: **proofs** (convergence, bounds) and **empiricism** (“try it and see”). **Math is preferable** when it works— a good formalization can **leapfrog** years of trial-and-error experiments. RL itself is a slow research loop for the same reason: many failed runs before a success.

In practice, deep RL has advanced mostly step by step: one experiment suggests the next tweak, and only later does math sometimes catch up and explain the pattern. An equation that predicts two years of lab work upfront remains rare; the hope is that enough experiments eventually expose patterns math can then formalize.

## Self-play

**Self-play** attacks RL's core difficulty: **sparse reward signal**. If the agent never succeeds, it often gets almost no useful feedback. In self-play the agent plays **both sides**: whenever one side wins, the other loses, so every match yields a clear contrast.

That means each self-play game produces learning signal, so problems cast as self-play can learn **much faster** than typical RL environments. Success so far is mostly in **games with a built-in opponent** (chess, Go, etc.). The open question is broader tasks (for example, a robot learning to build a simple hut).

A general recipe that turns *any* RL problem into a self-play formulation would be a major step. It would be another case where math could **leapfrog** years of experimentation (see [Math versus empirical trial and error](#math-versus-empirical-trial-and-error)).

## Imitation Learning

Many tasks cannot be turned into self-play. One option is to hand-design dense, progress-based rewards, but that can be as labor-intensive as solving part of the problem manually. At that point, it is often simpler to provide a **demonstration**.

The most direct version is **teleoperation**: a person controls the robot through the task, and the robot records what successful behavior looks like from its own sensors and actions. This gives high-signal data. A basic manipulation skill, such as picking up a bottle and placing it on a target from different starting positions, can sometimes be taught in minutes rather than through long trial-and-error RL.

The harder and more interesting version is **third-person imitation learning**. Here the robot watches a human perform the task and must translate that demonstration into its own body and action space. A human hand and a robot gripper are not the same, so the robot must infer the underlying intent rather than copy joint motions directly. Work led by Chelsea Finn frames this almost like **machine translation for demonstrations**: convert what the human did into what the robot should do.

## Simulation

Simulation means training or testing an AI agent inside a computer-generated version of the world instead of the real physical world.

For robotics, that might mean a virtual robot arm, virtual objects, simulated gravity, friction, contact forces, camera images, and so on. The robot can practice picking things up, walking, driving, or navigating without risking real hardware or people.

Much of imitation learning and self-play benefits from **simulation**. Simulators keep improving, so more training can happen in silicon before touching hardware.

The hard part is the sim-to-real gap: the simulated world is never exactly the real world. A policy that works in simulation might fail on a real robot because friction, lighting, object shapes, sensor noise, or contact dynamics differ slightly. A lot of robotics research is about making simulation useful despite that gap.
Maybe we should go beyond chasing one **perfect** simulator for sim-to-real transfer. The simulator only needs to be somewhat representative: train across an ensemble of simulators, so a policy that works across many imperfect worlds is more likely to transfer to the real one.

## What is missing for robots

Robots and self-driving systems could be tested far more often than humans (replicas, identical weights, repeated scenarios), but robotics still lacks the equivalent of **unit tests** or rigorous regression suites. An open problem: after a software update claims a better stack, how to verify **strict improvement on everything** that mattered before, with no new failure modes **creeping in**. There is no fully satisfying answer yet.

Humans rely on a coarse gate (pass the test, then drive) and still achieve very low accident rates per mile (on the order of millions of miles between serious incidents) relative to the brevity of the exam. That gap between a short test and long-run performance is what robotic safety research still needs to formalize.