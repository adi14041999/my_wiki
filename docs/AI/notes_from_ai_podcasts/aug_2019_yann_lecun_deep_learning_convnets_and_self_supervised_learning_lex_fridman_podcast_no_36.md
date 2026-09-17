# Aug 2019. [Yann LeCun- Deep Learning, ConvNets, and Self-Supervised Learning | Lex Fridman Podcast no. 36](https://www.youtube.com/watch?v=SGSOCuByo24&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
31 Aug 2019.

## HAL 9000 and objective functions

LeCun's office is decorated entirely with stills from *2001: A Space Odyssey*.

Asked whether HAL was flawed or evil, he says neither. There is no notion of evil in that context. It is an example of **value misalignment**: you give a machine an objective, the machine strives to achieve it, and if you have not constrained the objective (don't kill people, don't do things like this) a machine with power will do damaging things in service of it.

His point is that this is not a new problem. We already do exactly this with humans, and have for millennia.

The legal code is called *code*, which he thinks tells you something. So the idea that designing objective functions aligned with the common good is a new idea wrong.

!!! note "Related"
    [Stuart Russell](dec_2018_stuart_russell_long_term_future_of_artificial_intelligence_lex_fridman_podcast_no_9.md#the-control-problem) develops the same concern at greater length, including the observation that [corporations and governments are already misaligned optimizers](dec_2018_stuart_russell_long_term_future_of_artificial_intelligence_lex_fridman_podcast_no_9.md#misaligned-optimizers-corporations-and-governments).

He is also clear that these are not questions worth asking today: we have intelligent machines only in a very specialized sense, trained to do one thing rather than satisfying an objective, and until we have a design for a full-fledged autonomous intelligent system, asking how to design its objective is too abstract.

## The most surprising thing in deep learning

Asked for the most beautiful or surprising idea he has encountered, LeCun offers not an idea but an **empirical fact**: that you can take gigantic neural nets, train them with stochastic gradient descent on relatively small amounts of data, and it works.

This breaks everything in every pre-deep-learning textbook. You were told you need fewer parameters than data samples; that a non-convex objective gives you no guarantee of convergence; etc..

!!! note "The other side of this argument"
    [Vladimir Vapnik](nov_2018_vladimir_vapnik_statistical_learning_lex_fridman_podcast_no_5.md#the-admissible-set-of-functions) is the person who formalized the textbook position LeCun is describing— that you need a set of functions with small capacity to learn from few observations. The two episodes read well against each other.

Before he knew anything, it was **obvious** this was a good idea.

The intuition behind the original obviousness is an argument from existence. It is like the people in the late 19th century who proved that heavier-than-air flight was impossible— and meanwhile, there are birds. On the face of it the proof is wrong as an empirical matter. Likewise, we know the brain works; we do not know how, but we know it is a large network of neurons interacting, and that learning happens by changing the connections.

The second conviction, held since before he was an undergraduate, is that **intelligence is inseparable from learning**. Creating an intelligent machine by programming was a non-starter for him from the beginning, because every intelligent entity we know of arrives at its intelligence through learning.

## What is learning, reasoning, and can networks reason?

Reasoning, for LeCun, is a consequence of learning like other functions of the brain. The real question is **how you make reasoning compatible with gradient-based learning**.

He thinks neural nets can be made to reason. The open questions are how much prior structure has to be put in for something like human reasoning to emerge. He believes discrete logic is incompatible with gradient-based learning.

## What a reasoning system would need

Two ingredients.

**Working memory:** A subsystem that can store a reasonably large amount of factual, episodic information for a reasonable amount of time. LeCun sketches the three kinds of memory in the brain:

1. **The state of your cortex**— gone within about 20 seconds to a minute if nothing else holds it.
2. **The hippocampus**— short-to-medium term. You walked into this building and you remember where the exit and the elevators are; you have a map of the building.
3. **The synapses**— long-term.

What a reasoning system needs is the hippocampus-like thing. That is what memory networks were attempting, and what [transformers](../natural_language_processing/the_basic_transformer.md) approximate through their self-attention.

**Recurrence:** A network that can access that memory, retrieve information, crunch on it, and do this **iteratively**, because a chain of reasoning is a process of repeatedly updating your knowledge of the state of the world.

This is where he thinks transformers fall short. A transformer has a fixed number of layers, which caps the number of steps and therefore its representation; recurrence would let knowledge be built on and expanded. Asked whether this might simply emerge with scale, he is blunt: **no**. It is not clear how to read from and write into an associative memory efficiently.

### Reasoning as energy minimization

A second, more classical form. You have an [energy function](../deep_generative_models/energy_based_models.md) representing quality— energy goes up when things are bad and down when they are good.

Suppose you want to work out what gestures you need to grab an object or walk out the door. With a good model of your own body and of the environment, energy minimization lets you plan. In optimal control this is **model predictive control**: you have a model of what happens in the world as a consequence of your actions, and minimizing energy gives you the sequence of actions that optimizes an objective (minimize collisions, minimize energy spent on the gesture, and so on).

## Knowledge representation

On expert systems and knowledge graphs: too rigid and too brittle. Variables with values and constraints between them, represented by rules.

The main issue is **knowledge acquisition**. How do you reduce a pile of data into a network or graph of any kind? It relies on a human expert to encode the knowledge, which is essentially impractical.

The second issue is the compatibility problem again: symbols manipulated by logic are incompatible with learning. The suggestion Hinton has advocated for decades is to **replace symbols by vectors** and **replace logic by continuous functions**. That is compatible.

LeCun points to Léon Bottou's paper *From Machine Learning to Machine Reasoning* (Bottou was at Facebook at the time), whose idea is that a learning system should be able to manipulate objects in a space and put the result back into the same space (the working-memory idea again).

### Causal inference

He then makes the case that humans are not especially good at this either:

- **Physicists arguably do not believe in causality at all**, because the laws of microphysics are time-reversible; there is no arrow of time. It appears only in macroscopic systems, and whether it is emergent or fundamental is a genuine mystery.
- **Children get it backwards.** He cites Seymour Papert (who studied with Piaget, and who co-authored *Perceptrons* with Minsky) on asking a small child what causes wind. A four- or five-year-old will often say it is the branches of the trees moving. Their intuitive physics is not good enough yet.
- **Humanity has been deluded for millennia.** Anything unexplained gets attributed to a deity, which is a way of saying *I don't know the cause*.

## Why neural nets fell out of favour in the 1990s

They were not called deep learning then; they were just neural nets, and interest collapsed around 1995. The neural net community always existed, but it became disconnected from mainstream machine learning— essentially electrical engineering kept at it while computer science gave up.

LeCun is careful to say he was too close to it to analyze it without bias, but offers guesses. The central one is that **they were very hard to make work**:

- You implemented backprop in your favourite language, and that language was not Python or MATLAB, because those did not exist. It was Fortran or C.
- You would make basic mistakes— badly initialized weights, a network made too small because the textbook told you not to have too many parameters.
- You trained on XOR, because you had no other dataset to try.
- You used batch gradient, which is not sufficient.

There was a **bag of tricks** you had to know or reinvent, and many people simply could not make it work. Underlying all of this was the absence of software platforms: something flexible enough to build convolutional architectures, and good enough at displaying things that you could develop intuition about why training failed.

## Benchmarks and reducing ideas to practice

LeCun has advised: do not be fooled by people who claim to have an AGI system, or a system that works just like the human brain, or to have figured out how the brain works— **ask them what error rate they get on MNIST or ImageNet**.

He allows the specific benchmarks are dated but holds the philosophy. What matters is not that the task be *practical* but that it be something **the community has accepted as a standard**.

What struck him was how many people (particularly people with money to invest) would be taken in by someone claiming to have the algorithm of the cortex and asking for $50 million.

On the objection that genuinely new ideas may not yet have a benchmark, he agrees, and adds that **establishing the benchmark is part of the process**.

Where he sees benchmarks going is **interactive environments**. The classical supervised paradigm assumes samples are statistically independent and that the order you see them in does not matter. That breaks as soon as a machine can take an action that influences what it sees next: a robot goes somewhere and the next room is different, and consecutive samples are dependent because you can only move through space continuously. So people are building artificial environments (simulated houses with objects to interact with, simulated robots, games, etc.) and that is where the field is going.

## Why "general" is the wrong word

LeCun dislikes the term AGI because it implies human intelligence is general, and he thinks human intelligence is **nothing like general— it is very, very specialized**. We only feel general because we can only conceive of tasks within our comprehension.

His demonstration is a thought experiment about the optic nerve. You have about a million nerve fibres coming out of one eye; treat them as binary, so the input to your visual cortex is a million bits. Those fibres connect to a cortex whose connections are **local in space**, a little like a convolutional net.

Now imagine cutting the optic nerve and inserting a device that applies a **fixed random permutation** to all the fibres. What reaches your brain is a randomly shuffled version of the pixels. LeCun's claim is that even if this were done in infancy, you would never learn vision to anything like the same quality— because two pixels adjacent in the world now land in very different places in the cortex, and those neurons have no connections to each other.

The hardware is built to exploit the **locality of the real world**. That is specialization.

He then makes it quantitative. Suppose you want to classify patterns over those million bits. That is a Boolean function with a million binary inputs, so there are $2^{2^{1{,}000{,}000}}$ such functions— an unimaginable number. How many can your visual cortex actually compute? A vanishingly tiny sliver.

!!! note "Where $2^{2^{n}}$ comes from"
    The count happens in two stages, and the doubling in the exponent is the whole trick.

    **Stage 1: how many input patterns are there?** Each of the $n$ fibres is on or off, so there are $2^n$ distinct patterns the eye can deliver. With $n = 1{,}000{,}000$, that is $2^{1{,}000{,}000}$ possible images— the number of *rows* in the lookup table.

    **Stage 2**: Assigning a $0$ or a $1$ to every pattern (there are a million patterns) is the same act as **splitting the set of patterns into two groups**. For example, one such split results in cat and not-cat. Picking a function is therefore dividing the $2^n$ patterns into two groups. And the number of ways to divide a set (into two groups) of size $R$ is $2^R$, since each element is independently in or out. A classifier over a million-pixel retina is nothing more than **one particular way of dividing the space of all possible images into two piles**, and $2^{2^{1{,}000{,}000}}$ is how many such divisions exist.

    That framing is what makes LeCun's point land. Almost every one of those divisions is pure noise— an arbitrary scattering of images between the two piles, with visually identical images landing on opposite sides. The divisions we care about are the vanishingly rare ones where **nearby images land in the same pile**, and that is precisely the structure a cortex wired for locality, or a convolutional net, is built to capture. Shuffle the optic nerve and the target division leaves that rare family, which is why the shuffled brain cannot learn it.

He extends the point to physics. Take a container of gas: you can know pressure, temperature, volume, and write $PV = nRT$. That is a tiny number of bits compared to the full state— the position and momentum of every molecule. What you do not know about it is the **entropy**, and you interpret it as heat. It is entirely possible that there is strong structure in how those molecules move which we are simply not wired to perceive.

## Self-supervised learning

This is the only thing LeCun says he is interested in at the moment.

He calls it **self-supervised** because the underlying algorithms are the same supervised algorithms— what changes is what you train them to predict. Not a category provided by human labellers, but **a piece of the input that has been masked out**:

For example, show a window of a thousand words, remove 15% of them, and train the machine to predict the missing words.

!!! note "Related"
    The wiki's [Self-Supervised Learning](../deep_learning_for_computer_vision/self_supervised_learning.md) page covers the vision side of this— [pretext tasks](../deep_learning_for_computer_vision/self_supervised_learning.md#summary-of-pretext-tasks), [masked autoencoders](../deep_learning_for_computer_vision/self_supervised_learning.md#reconstruction-based-learning-mae), and [contrastive learning](../deep_learning_for_computer_vision/self_supervised_learning.md#contrastive-learning).

### Why it works for language and not for vision

This is the sharpest technical idea in the conversation, and the answer is **how you represent uncertainty in the prediction**.

For language, the possibility space is small and discrete. There are maybe 100,000 words in the lexicon, and the machine outputs a big probability vector. We know how to do that with computers, so representing uncertainty is easy. In LeCun's opinion that is *why* these techniques work for NLP.

Block out part of an image and ask a system to reconstruct it, and there are many possible answers, all perfectly legitimate.

The failure is visible and specific. LeCun is sitting still but might turn his head left or right. Train a predictor with least squares to minimize error against what actually happens, and what you get is a blurry image of himself in all possible future positions at once, which is not a useful prediction. And you cannot plan with blurry predictions: a perfect world model lets you run a hypothetical sequence of actions in your head and predict the outcome, but an imperfect one gives you nothing to plan with.

Asked whether self-play or simulation solves this, he says no. If the game is deterministic or quasi-deterministic, predicting the next few frames works fine. The problem comes precisely from the fact that the real world is not entirely predictable.

## The sample-efficiency argument

LeCun's case for why a model of the environment's dynamics matters.

- The best **model-free** deep RL takes about **80 hours** of training to reach the level a human reaches in about **15 minutes** on Atari games. It eventually beats humans, but it takes a long time.
- **AlphaStar** plays a single map with a single character type at better-than-human level, at the cost of roughly **200 years** of self-play.

Now apply those algorithms to driving. The car would have to drive millions of hours, kill thousands of pedestrians, run into thousands of trees, and **run off a cliff multiple times before working out it was a bad idea**.

Humans learn to drive in 20 or 30 hours without ever crashing. Driving next to a cliff, we know that turning the wheel right ends badly, because we have a good model of intuitive physics. Babies learn around eight or nine months that objects fall rather than float, which is why an eight-month-old in a highchair will systematically throw every toy on the ground while watching. They are not annoying you; **they are running the experiment**.

So what we bring is a predictive model of the world that keeps us from doing stupid things, and it is transferable, because stupid things are the same everywhere. The main problem, then, is **how to learn models of the world**, and that is what self-supervised learning is for.

### The benchmark he would want

Transfer learning works.

What he wants is a protocol:

1. A very large amount of **unlabelled** data (video clips, images) which you are **not allowed to label**.
2. Do whatever self-supervised training you like on it: frame prediction, masking a piece out, anything.
3. Then train on a supervised task such as ImageNet or MNIST, and **measure how the error falls as you increase the number of labelled samples**.

What you want to see is the error dropping much faster than training from random weights— reaching the level a purely supervised system achieves with far fewer labelled examples. He frames why this is the crucial question in practical terms: if you work on medical image analysis and know you need a million samples to reach a given error rate, **can self-supervised pre-training reduce that to a hundred?**

## The architecture of an autonomous intelligent system

LeCun's metaphor for progress: a series of mountains to climb, where you can see the first one but do not know whether there are fifty behind it.

He can name the first peak, which is what he is working on: **self-supervised learning— getting machines to learn models of the world by observation**.

The full architecture he sketches has four parts:

1. A **hardwired objective**— (in humans, the basal ganglia) the level of contentment or discontentment.
2. A **predictive model of the world**— given the state of the world at time $T$ and an action, what is the state at $T+1$. This is not a single answer, which is where the uncertainty problem bites. This is the world model.
3. An **objective predictor**— you do not put your hand in a fire because you predict it will hurt. This component predicts the result when you interact with the world model. The predictive model of the world helps you with this.
4. A **policy**— a module that figures out the best course of action to optimize the objective given the world model.

### Emotions

Emotions fall out of the same structure. The basal ganglia computes contentment; the objective predictor anticipates it. **Fear is the anticipation of bad things that may happen**— the inkling that something bad might occur. Uncertainty is what creates fear.

So the punchline is that we will not have autonomous intelligence without emotions, whatever emotions turn out to be.