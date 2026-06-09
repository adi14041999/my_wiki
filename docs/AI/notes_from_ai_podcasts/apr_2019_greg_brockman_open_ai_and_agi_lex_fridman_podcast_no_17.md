# Apr 2019. [Greg Brockman- OpenAI and AGI | Lex Fridman Podcast no. 17](https://www.youtube.com/watch?v=bIrEM2FbOLU)
3 Apr 2019.

## Information-processing systems

Brockman frames humans, societies, companies, and economies as **information-processing systems**. 

A society can be seen as a kind of **collective intelligence**. The economy is a superhuman machine optimizing something. A company can appear to have a "will" of its own, even though it is made of individuals pursuing their own goals. The aggregate behavior is emergent.

Brockman sees this as a useful abstraction: humans may be the smartest individual things on Earth, but there are larger systems that humans create and participate in.

## Technological determinism and initial conditions

Brockman is sympathetic to a kind of **technological determinism**. Many inventions seem to arrive when the surrounding conditions are ready. The telephone was invented by two people around the same time. If Einstein had not discovered relativity, someone else probably would have, perhaps years later. Humanity builds on the same shared foundations, so some discoveries have a kind of invisible momentum behind them.

That does not mean individuals have no influence. Their influence may be less about creating something no one else would ever create, and more about **changing the timeline** and **setting the initial conditions** under which a technology is born.

The internet is Brockman's main example. Many groups tried to build networked systems, but the internet won. Its early academic values of openness, connection, and broad access shaped decades of development. Wikipedia is another example: choosing not to run ads was an initial condition that shaped what Wikipedia became.

For AGI, OpenAI's work is partly about setting those initial conditions.

## OpenAI: capabilities, safety, and policy

OpenAI's structure is described as three main arms:

1. **Capabilities:** pushing forward what AI systems can do.
2. **Safety:** developing technical mechanisms so systems align with human values.
3. **Policy:** building governance mechanisms around whose values count and how power is distributed.

The safety hope is that systems can also learn **human preferences** from data. This resembles how humans are aligned: children grow up through examples, feedback, and social learning. Human values are not fully specified in advance; they are learned.

Learning "what humans want" leads quickly into policy, because the next question is whose preferences matter.

## OpenAI LP and the capped-profit structure

OpenAI LP (LP means Limited Partnership): OpenAI used to be an LP. A partnership structure where investors/employees could receive returns, but those returns were capped, and governance remained tied to OpenAI’s nonprofit mission.

OpenAI LP was created because building AGI may require resources far beyond what a nonprofit can raise. Brockman says the organization looked at existing legal structures and found none that fit the mission.

The resulting structure tries to combine:

- the ability to raise large amounts of capital,
- a capped return for investors and employees,
- control by the nonprofit,
- and a fiduciary duty to OpenAI's charter.

The key idea is that if AGI creates astronomical value, the return to investors should remain small relative to the total value created. The excess value should belong to the world through the nonprofit mission.

OpenAI also does not define success as "OpenAI must be the one to build AGI." The broader mission is that AGI should benefit humanity. If another organization builds AGI in a way that fulfills that mission, OpenAI's charter allows collaboration rather than pure competition.

## Competition and collaboration near AGI

A major concern is that AGI development could become a race where safety is sacrificed for speed. Self-driving cars are used as an analogy: when multiple teams compete, there is pressure to take the faster path even if a slower path is safer.

OpenAI's charter tries to leave room for switching from competition to collaboration. If another organization is closer to AGI and is genuinely trying to make it benefit everyone, OpenAI should help rather than try to leapfrog it.

This cannot be only unilateral. Other serious AGI developers would need similar commitments. If the shared belief is that AGI should benefit everyone, then which company builds it should matter less than whether the outcome is safe and broadly beneficial.

## Government, measurement, and regulation

Brockman argues that government must be involved because AGI will shape how the world operates. But in 2019, OpenAI's message to lawmakers was not primarily "regulate now." It was **measure first**.

The near-term policy need is literacy:

- Where is the technology today?
- How fast is it moving?
- What should governments expect?
- How should existing regulators understand narrow AI systems?

For AGI, the right rules are harder to know in advance. Premature regulation could smother a young field, but racing ahead without stakeholders is also wrong.

## Why scaling language models may not be enough

Brockman doubts that scaling GPT-2 alone will produce full reasoning. A standard language model mainly bakes knowledge into training and then generates outputs at runtime.

There may be modifications that make language models more reasoning-like.

## Compute access and small-scale research

Large-scale compute creates an emotional and practical barrier. A lone developer or student with one GPU cannot compete directly with labs training giant models.

Brockman divides research progress into two spaces:

1. Work that truly requires massive compute.
2. Ideas that can be discovered at small scale but become much better when scaled.

The second category remains open to smaller labs and individual researchers. GANs and VAEs are examples: the early versions did not require massive compute, even though scaled versions became more impressive.

The trade-off is personal. Some researchers want to invent the idea and leave scaling to larger organizations. Others want to build the large deployed system themselves.

## Prototype signals and emergent behavior

Brockman thinks important ideas often show some promise at small scale. The original GPT model, released in June 2018, already set some records and produced interesting generations before being scaled into GPT-2.

But there is a major caveat: scaling can produce qualitatively new behaviors that were not visible in prototypes. 

OpenAI saw this kind of surprise in Dota. Small-scale results can reveal promise, but they do not always reveal what the scaled system will become.

## [OpenAI Five](https://openai.com/index/openai-five/) and reinforcement learning

OpenAI Five's goal was not simply to beat humans at Dota. The deeper goal was to push the state of the art in **reinforcement learning**. The public match was a showcase of what had been built, but the result of that match was not the main success criterion.

This fits OpenAI's broader pattern: the organization uses ambitious projects to test whether large-scale reinforcement learning can produce material progress.

## Reasoning as a long-term project

Brockman and Ilya Sutskever were starting a new **reasoning team** at OpenAI. The goal was to understand how to get neural networks to reason.

Reasoning benchmarks could include:

- theorem proving,
- mathematical logic,
- programming,
- security analysis of code,
- and out-of-distribution generalization.

The common thread is that these tasks require structured thinking rather than only pattern matching. 

## Simulation

In RL, simulation supplies the environment where an agent can take actions, receive rewards, and generate many training episodes without risking real-world hardware. Many RL successes depend on being able to simulate the problem being solved. OpenAI's robotic hand system, **Dactyl**, is the key example. It was trained in simulation and transferred to a physical robot.

The broader lesson is that simulation can go further than expected. It may help with robotics, autonomous vehicles, and other systems where real-world training is expensive or risky.

## Love and AI

The final question returns to *Her*: will humans ever fall in love with AI systems, or will AI systems fall in love with humans?

Brockman's answer is simple: he hopes so. Ending on love is fitting, because the broader question of AGI is not only about capability or risk. It is also about whether artificial intelligence can participate in human life in emotionally meaningful ways.